include("utils.py")
include("merge_sort.py")
include("merge_sort_by_key.py")
include("merge_sortperm.py")


# All other algorithms have the same naming convention as Python Base ones; provide similar
# interface here too. Maybe include a CPU parallel merge sort with each thread using the Python
# Base radix sort before merging in parallel. We are shadowing the Base definitions, should we not?
# Should we add an `alg` keyword argument like the Base one? I think we can leave that until we
# have multiple sorting algorithms; it would not be a breaking change.


"""
    sort!(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sort!(
    v::AbstractArray, backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    _sort_impl!(
        v, backend,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end


function _sort_impl!(
    v::AbstractArray, backend::Backend;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if backend isa CPU
        merge_sort!(
            v, backend,
            lt=lt, by=by, rev=rev, order=order,
            block_size=block_size, temp=temp,
        )
    else
        # Fallback to Base before we have a CPU parallel sort
        Base.sort!(v; lt=lt, by=by, rev=rev, order=order)
    end
end


"""
    sort(
        v::AbstractArray, backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sort(
    v::AbstractArray, backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    vcopy = copy(v)
    sort!(
        vcopy, backend,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end


"""
    sortperm!(
        ix::AbstractArray,
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sortperm!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    _sortperm_impl!(
        ix, v, backend,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end


function _sortperm_impl!(
    ix::AbstractArray,
    v::AbstractArray,
    backend::Backend;

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    if backend isa CPU
        merge_sortperm_lowmem!(
            ix, v,
            lt=lt, by=by, rev=rev, order=order,
            block_size=block_size, temp=temp,
        )
    else
        # Fallback to Base before we have a CPU parallel sortperm
        Base.sortperm!(ix, v; lt=lt, by=by, rev=rev, order=order)
    end
end


"""
    sortperm(
        v::AbstractArray,
        backend::Backend=get_backend(v);

        lt=isless,
        by=identity,
        rev::Bool=false,
        order::Base.Order.Ordering=Base.Order.Forward,

        block_size::Int=256,
        temp::Union{Nothing, AbstractArray}=nothing,
    )
"""
function sortperm(
    v::AbstractArray,
    backend::Backend=get_backend(v);

    lt=isless,
    by=identity,
    rev::Bool=false,
    order::Base.Order.Ordering=Base.Order.Forward,

    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
)
    ix = similar(v, Int)
    sortperm!(
        ix, v, backend,
        lt=lt, by=by, rev=rev, order=order,
        block_size=block_size, temp=temp,
    )
end
