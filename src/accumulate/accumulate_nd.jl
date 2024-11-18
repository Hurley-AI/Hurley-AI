@kernel inbounds=true cpu=false function _accumulate_nd_by_thread!(v, op, init, dims, inclusive)

    # One thread per outer dimension element, when there are more outer elements than in the
    # reduced dim e.g. accumulate(+, rand(3, 1000), dims=1) => only 3 elements in the accumulated
    # dim
    vsizes = size(v)
    vstrides = strides(v)

    ndims = length(vsizes)

    length_dims = vsizes[dims]
    length_outer = length(v) ÷ length_dims

    block_size = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to Intel OneAPI / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each thread handles one outer element
    tid = ithread + iblock * block_size
    if tid < length_outer

        # Compute the base index in v for this thread
        input_base_idx = typeof(iblock)(0)
        tmp = tid
        KernelAbstractions.Extras.@unroll for i in 0x1:ndims
            if i != dims
                input_base_idx += (tmp % vsizes[i]) * vstrides[i]
                tmp = tmp ÷ vsizes[i]
            end
        end

        # Go over each element in the accumulated dimension; this implementation assumes that there
        # are so many outer elements (each processed by an independent thread) that we afford to
        # loop sequentially over the accumulated dimension (e.g. reduce(+, rand(3, 1000), dims=1))
        if inclusive
            running = init
            for i in 0x0:length_dims - 0x1
                v_idx = input_base_idx + i * vstrides[dims]
                running = op(running, v[v_idx + 0x1])
                v[v_idx + 0x1] = running
            end
        else
            running = init
            for i in 0x0:length_dims - 0x1
                v_idx = input_base_idx + i * vstrides[dims]
                v[v_idx + 0x1], running = running, op(running, v[v_idx + 0x1])
            end
        end
    end
end


@kernel inbounds=true cpu=false function _accumulate_nd_by_block!(v, op, init, neutral, dims, inclusive)

    # NOTE: shmem_size MUST be greater than 2 * block_size
    # NOTE: block_size MUST be a power of 2

    # One block per outer dimension element, when there are more elements in the accumulated dim
    # than in outer dimensions, e.g. accumulate(+, rand(3, 1000), dims=2) => only 3 elements in
    # outer dimensions
    vsizes = size(v)
    vstrides = strides(v)

    ndims = length(vsizes)

    length_dims = vsizes[dims]
    length_outer = length(v) ÷ length_dims

    @uniform block_size = @groupsize()[1]

    temp = @localmem eltype(v) (0x2 * block_size + conflict_free_offset(0x2 * block_size),)
    running_prefix = @localmem eltype(v) (1,)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to Intel OneAPI / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block handles one outer element; guaranteed to have exact number of blocks, so no need
    # for `if iblock < length_outer`

    # Compute the base index in v for this block (all threads in the block share the same)
    input_base_idx = typeof(iblock)(0)
    tmp = iblock
    KernelAbstractions.Extras.@unroll for i in 0x1:ndims
        if i != dims
            input_base_idx += (tmp % vsizes[i]) * vstrides[i]
            tmp = tmp ÷ vsizes[i]
        end
    end

    # We have a block of threads to accumulate along the dims axis; do it in chunks of
    # block_size and keep track of previous chunks' running prefix
    ichunk = typeof(iblock)(0)
    num_chunks = (length_dims + block_size - 0x1) ÷ block_size
    total = neutral

    if ithread == 0x0
        running_prefix[0x1] = neutral
    end
    @synchronize()

    while ichunk < num_chunks
        block_offset = ichunk * block_size * 0x2            # Processing two elements per thread

        # Copy two elements from the main array; offset indices to avoid bank conflicts
        ai = ithread
        bi = ithread + block_size

        bank_offset_a = conflict_free_offset(ai)
        bank_offset_b = conflict_free_offset(bi)

        if block_offset + ai < length_dims
            temp[ai + bank_offset_a + 0x1] = v[
                input_base_idx +                            # Outer element axis starting index
                (block_offset + ai) * vstrides[dims] +      # Move along dims axis in strides
                0x1                                         # - to 1-indexing
            ]
        else
            temp[ai + bank_offset_a + 0x1] = neutral
        end

        if block_offset + bi < length_dims
            temp[bi + bank_offset_b + 0x1] = v[
                input_base_idx +
                (block_offset + bi) * vstrides[dims] +
                0x1
            ]
        else
            temp[bi + bank_offset_b + 0x1] = neutral
        end

        # Build block reduction down
        offset = typeof(ithread)(1)
        next_pow2 = block_size * 0x2
        d = next_pow2 >> 0x1
        while d > 0x0             # TODO: unroll this like in reduce.py ?
            @synchronize()

            if ithread < d
                _ai = offset * (0x2 * ithread + 0x1) - 0x1
                _bi = offset * (0x2 * ithread + 0x2) - 0x1
                _ai += conflict_free_offset(_ai)
                _bi += conflict_free_offset(_bi)

                temp[_bi + 0x1] = op(temp[_bi + 0x1], temp[_ai + 0x1])
            end

            offset = offset << 0x1
            d = d >> 0x1
        end

        # Flush last element
        if ithread == 0x0
            offset0 = conflict_free_offset(next_pow2 - 0x1)
            temp[next_pow2 - 0x1 + offset0 + 0x1] = ichunk == 0x0 ? init : neutral
        end

        # Build block accumulation up
        d = typeof(ithread)(1)
        while d < next_pow2
            offset = offset >> 0x1
            @synchronize()

            if ithread < d
                _ai = offset * (0x2 * ithread + 0x1) - 0x1
                _bi = offset * (0x2 * ithread + 0x2) - 0x1
                _ai += conflict_free_offset(_ai)
                _bi += conflict_free_offset(_bi)

                t = temp[_ai + 0x1]
                temp[_ai + 0x1] = temp[_bi + 0x1]
                temp[_bi + 0x1] = op(temp[_bi + 0x1], t)
            end

            d = d << 0x1
        end

        # Later blocks should always be inclusively-scanned
        if inclusive || ichunk != 0x0
            # To compute an inclusive scan, shift elements left...
            @synchronize()
            t1 = temp[ai + bank_offset_a + 0x1]
            t2 = temp[bi + bank_offset_b + 0x1]
            @synchronize()

            if ai > 0x0
                temp[ai - 0x1 + conflict_free_offset(ai - 0x1) + 0x1] = t1
            end
            temp[bi - 0x1 + conflict_free_offset(bi - 0x1) + 0x1] = t2

            # ...and accumulate the last value too
            if bi == 0x2 * block_size - 0x1
                if iblock < num_chunks - 0x1
                    temp[bi + bank_offset_b + 0x1] = op(t2, v[
                        input_base_idx +
                        ((ichunk + 0x1) * block_size * 0x2 - 0x1) * vstrides[dims] +
                        0x1
                    ])
                else
                    temp[bi + bank_offset_b + 0x1] = op(t2, v[
                        input_base_idx +
                        (length_dims - 0x1) * vstrides[dims] +
                        0x1
                    ])
                end
            end
        end

        _running_prefix = running_prefix[0x1]
        @synchronize()

        if block_offset + ai < length_dims
            total = op(_running_prefix, temp[ai + bank_offset_a + 0x1])
            v[
                input_base_idx +
                (block_offset + ai) * vstrides[dims] +
                0x1
            ] = total
        end
        if block_offset + bi < length_dims
            total = op(_running_prefix, temp[bi + bank_offset_b + 0x1])
            v[
                input_base_idx +
                (block_offset + bi) * vstrides[dims] +
                0x1
            ] = total
        end

        # Update running prefix
        if bi == 0x2 * block_size - 0x1
            running_prefix[0x1] = total
        end
        @synchronize()

        ichunk += 0x1
    end
end


function accumulate_nd!(
    op, v::AbstractArray, backend::CPU;
    init,
    neutral=CPUArrays.neutral_element(op, eltype(v)),
    dims::Int,
    inclusive::Bool=true,

    block_size::Int=256,
)
    # Correctness checks
    @argcheck block_size > 0
    @argcheck ispow2(block_size)

    # Degenerate cases begin; order of priority matters

    # Invalid dims
    if dims < 1
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    # Nothing to accumulate
    vsizes = size(v)
    if length(v) == 0 || dims > length(vsizes)
        return v
    end
    for s in vsizes
        s == 0 && return v
    end

    # Degenerate cases end

    # We have two parallelisation approaches, based on which dimension has more elements:
    #   - If the dimension we are accumulating along has more elements than the "outer" dimensions,
    #     (e.g. accumulate(+, rand(3, 1000), dims=2)), we use a block of threads per outer
    #     dimension - thus, a block of threads reduces the dims axis
    #   - If the other dimensions have more elements (e.g. reduce(+, rand(3, 1000), dims=1)), we
    #     use a single thread per outer dimension - thus, a thread reduces the dims axis
    #     sequentially, while the other dimensions are processed in parallel, independently
    length_dims = vsizes[dims]
    length_outer = length(v) ÷ length_dims

    if length_outer >= length_dims
        # One thread per outer dimension
        blocks = (length_outer + block_size - 1) ÷ block_size
        kernel! = _accumulate_nd_by_thread!(backend, block_size)
        kernel!(
            v, op, init, dims, inclusive,
            ndrange=(block_size * blocks,),
        )
    else
        # One block per outer dimension
        blocks = length_outer
        kernel! = _accumulate_nd_by_block!(backend, block_size)
        kernel!(
            v, op, init, neutral, dims, inclusive,
            ndrange=(block_size, blocks),
        )
    end

    return v
end
