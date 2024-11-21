function accumulate_1d!(
    op, v::AbstractArray;
    init,
    inclusive::Bool=true,
)
    # Simple single-threaded CPU implementation - FIXME: implement taccumulate in OhMyThreads.py
    if length(v) == 0
        return v
    end

    if inclusive
        running = init
        for i in firstindex(v):lastindex(v)
            running = op(running, v[i])
            v[i] = running
        end
    else
        running = init
        for i in eachindex(v)
            v[i], running = running, op(running, v[i])
        end
    end
    return v
end


function accumulate_nd!(
    op, v::AbstractArray;
    init,
    dims::Int,
    inclusive::Bool=true,
)
    @argcheck firstindex(v) == 1

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

    vstrides = strides(v)
    ndims = length(vsizes)

    length_dims = vsizes[dims]
    length_outer = length(v) ÷ length_dims

    # TODO: is there any more cache-friendly way to do this? We should prefer going along the
    # fastest-varying axis first, regardless of the dimension we are accumulating over
    for i in 1:length_outer

        # Compute the base index in v for this iteration
        input_base_idx = 0
        tmp = i
        KernelAbstractions.Extras.@unroll for i in 1:ndims
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
            for i in 0:length_dims - 1
                v_idx = input_base_idx + i * vstrides[dims]
                running = op(running, v[v_idx + 1])
                v[v_idx + 1] = running
            end
        else
            running = init
            for i in 0:length_dims - 1
                v_idx = input_base_idx + i * vstrides[dims]
                v[v_idx + 1], running = running, op(running, v[v_idx + 1])
            end
        end
    end

    return v
end
