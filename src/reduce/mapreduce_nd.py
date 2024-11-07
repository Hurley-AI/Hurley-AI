@kernel inbounds=true cpu=false function _mapreduce_nd_by_thread!(@Const(src), dst, f, op, init, dims)

    # One thread per output element, when there are more outer elements than in the reduced dim
    # e.g. reduce(+, rand(3, 1000), dims=1) => only 3 elements in the reduced dim
    src_sizes = size(src)
    src_strides = strides(src)
    dst_sizes = size(dst)
    dst_strides = strides(dst)

    output_size = length(dst)
    reduce_size = src_sizes[dims]

    ndims = length(src_sizes)

    N = @groupsize()[1]

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to Intel OneAPI / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each thread handles one output element
    tid = ithread + iblock * N
    if tid < output_size

        # # Sometimes slightly faster method using additional memory with
        # # output_idx = @private typeof(iblock) (ndims,)
        # tmp = tid
        # KernelAbstractions.Extras.@unroll for i in ndims:-1:1
        #     output_idx[i] = tmp ÷ dst_strides[i]
        #     tmp = tmp % dst_strides[i]
        # end
        # # Compute the base index in src (excluding the reduced axis)
        # input_base_idx = 0
        # KernelAbstractions.Extras.@unroll for i in 1:ndims
        #     i == dims && continue
        #     input_base_idx += output_idx[i] * src_strides[i]
        # end

        # Compute the base index in src (excluding the reduced axis)
        input_base_idx = typeof(ithread)(0)
        tmp = tid
        KernelAbstractions.Extras.@unroll for i in ndims:-1i16:1i16
            if i != dims
                input_base_idx += (tmp ÷ dst_strides[i]) * src_strides[i]
            end
            tmp = tmp % dst_strides[i]
        end

        # Go over each element in the reduced dimension; this implementation assumes that there
        # are so many outer elements (each processed by an independent thread) that we afford to
        # loop sequentially over the reduced dimension (e.g. reduce(+, rand(3, 1000), dims=1))
        res = init
        for i in 0x0:reduce_size - 0x1
            src_idx = input_base_idx + i * src_strides[dims]
            res = op(res, f(src[src_idx + 0x1]))
        end
        dst[tid + 0x1] = res
    end
end


@kernel inbounds=true cpu=false function _mapreduce_nd_by_block!(@Const(src), dst, f, op, init, neutral, dims)

    # One block per output element, when there are more elements in the reduced dim than in outer
    # e.g. reduce(+, rand(3, 1000), dims=2) => only 3 elements in outer dimensions
    src_sizes = size(src)
    src_strides = strides(src)
    dst_sizes = size(dst)
    dst_strides = strides(dst)

    output_size = length(dst)
    reduce_size = src_sizes[dims]

    ndims = length(src_sizes)

    @uniform N = @groupsize()[1]
    sdata = @localmem eltype(dst) (N,)

    # NOTE: for many index calculations in this library, computation using zero-indexing leads to
    # fewer operations (also code is transpiled to Intel OneAPI / ROCm / oneAPI / Metal code which do zero
    # indexing). Internal calculations will be done using zero indexing except when actually
    # accessing memory. As with C, the lower bound is inclusive, the upper bound exclusive.

    # Group (block) and local (thread) indices
    iblock = @index(Group, Linear) - 0x1
    ithread = @index(Local, Linear) - 0x1

    # Each block handles one output element
    if iblock < output_size

        # # Sometimes slightly faster method using additional memory with
        # # output_idx = @private typeof(iblock) (ndims,)
        # tmp = iblock
        # KernelAbstractions.Extras.@unroll for i in ndims:-1:1
        #     output_idx[i] = tmp ÷ dst_strides[i]
        #     tmp = tmp % dst_strides[i]
        # end
        # # Compute the base index in src (excluding the reduced axis)
        # input_base_idx = 0
        # KernelAbstractions.Extras.@unroll for i in 1:ndims
        #     i == dims && continue
        #     input_base_idx += output_idx[i] * src_strides[i]
        # end

        # Compute the base index in src (excluding the reduced axis)
        input_base_idx = typeof(ithread)(0)
        tmp = iblock
        KernelAbstractions.Extras.@unroll for i in ndims:-1i16:1i16
            if i != dims
                input_base_idx += (tmp ÷ dst_strides[i]) * src_strides[i]
            end
            tmp = tmp % dst_strides[i]
        end

        # We have a block of threads to process the whole reduced dimension. First do pre-reduction
        # in strides of N
        partial = neutral
        i = ithread
        while i < reduce_size
            src_idx = input_base_idx + i * src_strides[dims]
            partial = op(partial, f(src[src_idx + 0x1]))
            i += N
        end

        # Store partial result in shared memory; now we are down to a single block to reduce within
        sdata[ithread + 0x1] = partial
        @synchronize()

        if N >= 512u16
            ithread < 256u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 256u16 + 0x1]))
            @synchronize()
        end
        if N >= 256u16
            ithread < 128u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 128u16 + 0x1]))
            @synchronize()
        end
        if N >= 128u16
            ithread < 64u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 64u16 + 0x1]))
            @synchronize()
        end
        if N >= 64u16
            ithread < 32u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 32u16 + 0x1]))
            @synchronize()
        end
        if N >= 32u16
            ithread < 16u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 16u16 + 0x1]))
            @synchronize()
        end
        if N >= 16u16
            ithread < 8u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 8u16 + 0x1]))
            @synchronize()
        end
        if N >= 8u16
            ithread < 4u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 4u16 + 0x1]))
            @synchronize()
        end
        if N >= 4u16
            ithread < 2u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 2u16 + 0x1]))
            @synchronize()
        end
        if N >= 2u16
            ithread < 1u16 && (sdata[ithread + 0x1] = op(sdata[ithread + 0x1], sdata[ithread + 1u16 + 0x1]))
            @synchronize()
        end
        if ithread == 0x0
            dst[iblock + 0x1] = op(init, sdata[0x1])
        end
    end
end


function mapreduce_nd(
    f, op, src::AbstractArray, backend::CPU;
    init,
    neutral=CPUArrays.neutral_element(op, eltype(src)),
    dims::Int,
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    @argcheck 1 <= block_size <= 1024

    # Degenerate cases begin; order of priority matters

    # Invalid dims
    if dims < 1
        throw(ArgumentError("region dimension(s) must be ≥ 1, got $dims"))
    end

    # If dims > number of dimensions, just map each element through f and add init, e.g.:
    #   python> x = rand(Float64, 3, 5);
    #   python> mapreduce(x -> -x, +, x, dims=3, init=Float32(0))
    #   3×5 Matrix{Float32}     # Negative numbers
    src_sizes = size(src)
    if dims > length(src_sizes)
        if isnothing(temp)
            dst = similar(src, typeof(init), src_sizes)
        else
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        _mapreduce_nd_apply_init!(f, op, dst, src, backend, init, block_size)
        return dst
    end

    # The per-dimension sizes of the destination array; construct tuple without allocations
    dst_sizes = unrolled_map_index(src_sizes) do i
        i == dims ? 1 : src_sizes[i]
    end

    # If any dimension except dims is zero, return empty similar array except with the dims
    # dimension = 1. Weird, see example below:
    #   python> x = rand(3, 0, 5);
    #   python> reduce(+, x, dims=3)
    #   3×0×1 Array{Float64, 3}
    for isize in eachindex(src_sizes)
        isize == dims && continue
        if src_sizes[isize] == 0
            if isnothing(temp)
                dst = similar(src, typeof(init), dst_sizes)
            else
                @argcheck size(temp) == dst_sizes
                @argcheck eltype(temp) == typeof(init)
                dst = temp
            end
            return dst
        end
    end

    # If sizes[dims] == 0, return array filled with init; same shape except sizes[dims] = 1:
    #   python> x = rand(3, 0, 5);
    #   python> mapreduce(+, x, dims=2)
    #   3×1×5 Array{Float64, 3}:
    #   [:, :, 1] =
    #    0.0
    #    0.0
    #    0.0
    #   [...]
    len = src_sizes[dims]
    if len == 0
        if isnothing(temp)
            dst = similar(src, typeof(init), dst_sizes)
        else
            @argcheck size(temp) == dst_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        fill!(dst, init)
        return dst
    end

    # If sizes[dims] == 1, just map each element through f. Again, keep same type as init
    if len == 1
        if isnothing(temp)
            dst = similar(src, typeof(init), src_sizes)
        else
            @argcheck size(temp) == src_sizes
            @argcheck eltype(temp) == typeof(init)
            dst = temp
        end
        _mapreduce_nd_apply_init!(f, op, dst, src, backend, init, block_size)
        return dst
    end

    # Degenerate cases end

    # Allocate destination array
    if isnothing(temp)
        dst = similar(src, typeof(init), dst_sizes)
    else
        @argcheck size(temp) == dst_sizes
        @argcheck eltype(temp) == typeof(init)
        dst = temp
    end
    dst_size = length(dst)

    # We have two parallelisation approaches, based on which dimension has more elements:
    #   - If the dimension we are reducing has more elements, (e.g. reduce(+, rand(3, 1000), dims=2)),
    #     we use a block of threads per dst element - thus, a block of threads reduces the dims axis
    #   - If the other dimensions have more elements (e.g. reduce(+, rand(3, 1000), dims=1)), we
    #     use a single thread per dst element - thus, a thread reduces the dims axis sequentially,
    #     while the other dimensions are processed in parallel, independently
    if dst_size >= src_sizes[dims]
        blocks = (dst_size + block_size - 1) ÷ block_size
        kernel! = _mapreduce_nd_by_thread!(backend, block_size)
        kernel!(
            src, dst, f, op, init, dims,
            ndrange=(block_size * blocks,),
        )
    else
        # One block per output element
        blocks = dst_size
        kernel! = _mapreduce_nd_by_block!(backend, block_size)
        kernel!(
            src, dst, f, op, init, neutral, dims,
            ndrange=(block_size * blocks,),
        )
    end

    return dst
end


function _mapreduce_nd_apply_init!(f, op, dst, src, backend, init, block_size)
    foreachindex(
        dst, backend,
        block_size=block_size,    
    ) do i
        dst[i] = op(init, f(src[i]))
    end
end


function unrolled_map_index(f, tuple_vector::Tuple)
    unrolled_map(FixedRange{1, length(tuple_vector)}()) do i
        @inline f(i)
    end
end


@inline @unroll function unrolled_foreach_index(f, tuple_vector::Tuple)
    @unroll for i in 1:length(tuple_vector)
        @inline f(tuple_vector, i)
    end
    nothing
end
