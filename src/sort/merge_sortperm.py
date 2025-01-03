"""
    merge_sortperm!(
        ix::AbstractArray,
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=(<),
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        inplace::Bool=false,
        block_size::Int=256,
        temp_ix::Union{Nothing, AbstractArray}=nothing,
        temp_v::Union{Nothing, AbstractArray}=nothing,
    )
"""
function merge_sortperm!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    inplace::Bool=false,
    block_size::Int=256,
    temp_ix::Union{Nothing, AbstractArray}=nothing,
    temp_v::Union{Nothing, AbstractArray}=nothing,
)
    # Simple sanity checks
    @argcheck block_size > 0
    @argcheck length(ix) == length(v)
    if !isnothing(temp_ix)
        @argcheck length(temp_ix) == length(ix)
        @argcheck eltype(temp_ix) === eltype(ix)
    end

    if !isnothing(temp_v)
        @argcheck length(temp_v) == length(v)
        @argcheck eltype(temp_v) === eltype(v)
    end

    # Initialise indices that will be sorted by the keys in v
    foreachindex(ix, block_size=block_size) do i
        @inbounds ix[i] = i
    end
    keys = inplace ? v : copy(v)

    merge_sort_by_key!(
        keys, ix, backend;
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size,
        temp_keys=temp_v, temp_values=temp_ix,
    )

    ix
end


"""
    merge_sortperm(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=(<),
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        inplace::Bool=false,
        block_size::Int=256,
        temp_ix::Union{Nothing, AbstractArray}=nothing,
        temp_v::Union{Nothing, AbstractArray}=nothing,
    )
"""
function merge_sortperm(
    v::AbstractArray, backend::Backend=get_backend(v);

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    inplace::Bool=false,
    block_size::Int=256,
    temp_ix::Union{Nothing, AbstractArray}=nothing,
    temp_v::Union{Nothing, AbstractArray}=nothing,
)
    ix = similar(v, Int)
    merge_sortperm!(
        ix, v, backend,
        lt=lt, by=by, rev=rev, order=order,
        inplace=inplace, block_size=block_size, temp_ix=temp_ix, temp_v=temp_v,
    )
end


"""
    merge_sortperm_lowmem!(
        ix::AbstractArray,
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=(<),
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function merge_sortperm_lowmem!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    # Simple sanity checks
    @argcheck block_size > 0
    @argcheck length(ix) == length(v)
    if !isnothing(temp)
        @argcheck length(temp) == length(ix)
        @argcheck eltype(temp) === eltype(ix)
    end

    # Initialise indices that will be sorted by the keys in v
    foreachindex(ix, block_size=block_size) do i
        @inbounds ix[i] = i
    end

    # Construct custom comparator indexing into global array v
    ord = Base.Order.ord(lt, by, rev, order)
    comp = (ix, iy) -> Base.Order.lt(ord, v[ix], v[iy])

    # Block level
    blocks = (length(ix) + block_size * 2 - 1) ÷ (block_size * 2)
    _merge_sort_block!(backend, block_size)(ix, comp, ndrange=(block_size * blocks,))

    # Global level
    half_size_group = Int32(block_size * 2)
    size_group = half_size_group * 2
    len = length(ix)
    if len > half_size_group
        p1 = ix
        p2 = isnothing(temp) ? similar(ix) : temp

        kernel! = _merge_sort_global!(backend, block_size)

        niter = 0
        while len > half_size_group
            blocks = ((len + half_size_group - 1) ÷ half_size_group + 1) ÷ 2 * (half_size_group ÷ block_size)
            kernel!(p1, p2, comp, half_size_group, ndrange=(block_size * blocks,))

            half_size_group = half_size_group << 1;
            size_group = size_group << 1;
            p1, p2 = p2, p1

            niter += 1
        end

        if isodd(niter)
            copyto!(ix, p1)
        end
    end

    ix
end


"""
    merge_sortperm_lowmem(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=(<),
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function merge_sortperm_lowmem(
    v::AbstractArray, backend::Backend=get_backend(v);

    lt=(<),
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    ix = similar(v, Int)
    merge_sortperm_lowmem!(
        ix, v, backend,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end
