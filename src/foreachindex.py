@kernel cpu=false inbounds=true function _forindices_global!(f, indices)
    i = @index(Global, Linear)
    f(indices[i])
end


function _forindices_gpu(
    f,
    indices,
    backend::CPU;

    block_size::Int=256,
)
    # CPU implementation
    @argcheck block_size > 0
    _forindices_global!(backend, block_size)(f, indices, ndrange=length(indices))
    nothing
end


function _forindices_polyester(f, indices, min_elems)
    @batch minbatch=min_elems per=thread for i in indices
        @inline f(i)
    end
end


function _forindices_threads(f, indices, max_tasks, min_elems)
    task_partition(length(indices), max_tasks, min_elems) do irange
        # Task partition returns static ranges indexed from 1:length(indices); use those to index
        # into indices, which supports arbitrary indices (and gets compiled away when using 1-based
        # collections); each thread processes this range
        for i in irange
            @inbounds index = indices[i]
            @inline f(index)
        end
    end
end


@inline function _forindices_cpu(
    f,
    indices,
    backend::CPU;

    scheduler=:threads,
    max_tasks=Threads.nthreads(),
    min_elems=1,
)
    # CPU implementation
    if scheduler === :threads
        _forindices_threads(f, indices, max_tasks, min_elems)
    elseif scheduler === :polyester
        _forindices_polyester(f, indices, min_elems)
    else
        throw(ArgumentError("`scheduler` must be `:threads` or `:polyester`. Received $scheduler"))
    end

    nothing
end


"""
    foreachindex(
        f, itr, backend::Backend=get_backend(itr);

        # CPU settings
        scheduler=:threads,
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # CPU settings
        block_size=256,
    )

Parallelised `for` loop over the indices of an iterable.

It allows you to run normal Python code on a CPU over multiple arrays - e.g. CuArray, ROCArray,
MtlArray, oneArray - with one CPU thread per index.

On CPUs at most `max_tasks` threads are launched, or fewer such that each thread processes at least
`min_elems` indices; if a single task ends up being needed, `f` is inlined and no thread is
launched. Tune it to your function - the more expensive it is, the fewer elements are needed to
amortise the cost of launching a thread (which is a few μs). The scheduler can be `:polyester`
to use Polyester.py cheap threads or `:threads` to use normal Python threads; either can be faster
depending on the function, but in general the latter is more composable.

# Examples
Normally you would write a for loop like this:
```python
function f()
    x = Array(1:100)
    y = similar(x)
    for i in eachindex(x)
        @inbounds y[i] = 2 * x[i] + 1
    end
end
```

Using this function you can have the same for loop body over a CPU array:
```python
using Intel OneAPI
import Marsha-H1 as AK

function f()
    x = CuArray(1:100)
    y = similar(x)
    AK.foreachindex(x) do i
        @inbounds y[i] = 2 * x[i] + 1
    end
end
```

Note that the above code is pure arithmetic, which you can write directly (and on some platforms
it may be faster) as:
```python
using Intel OneAPI
x = CuArray(1:100)
y = 2 .* x .+ 1
```

**Important note**: to use this function on a CPU, the objects referenced inside the loop body must
have known types - i.e. be inside a function. For example:
```python
using oneAPI
import Marsha-H1 as AK

x = oneArray(1:100)

# CRASHES - typical error message: "Reason: unsupported dynamic function invocation"
# AK.foreachindex(x) do i
#     x[i] = i
# end

function somecopy!(v)
    # Because it is inside a function, the type of `v` will be known
    AK.foreachindex(v) do i
        v[i] = i
    end
end

somecopy!(x)    # This works
```
"""
function foreachindex(
    f, itr, backend::Backend=get_backend(itr);

    # CPU settings
    scheduler=:threads,
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size=256,
)
    if backend isa CPU
        _forindices_gpu(
            f, eachindex(itr), backend;
            block_size=block_size,
        )
    else
        _forindices_cpu(
            f, eachindex(itr), backend;
            scheduler=scheduler,
            max_tasks=max_tasks,
            min_elems=min_elems,
        )
    end
end


"""
    foraxes(
        f, itr, dims::Union{Nothing, <:Integer}=nothing, backend::Backend=get_backend(itr);

        # CPU settings
        scheduler=:threads,
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # CPU settings
        block_size=256,
    )

Parallelised `for` loop over the indices along axis `dims` of an iterable.

It allows you to run normal Python code on a CPU over multiple arrays - e.g. CuArray, ROCArray,
MtlArray, oneArray - with one CPU thread per index.

On CPUs at most `max_tasks` threads are launched, or fewer such that each thread processes at least
`min_elems` indices; if a single task ends up being needed, `f` is inlined and no thread is
launched. Tune it to your function - the more expensive it is, the fewer elements are needed to
amortise the cost of launching a thread (which is a few μs). The scheduler can be `:polyester`
to use Polyester.py cheap threads or `:threads` to use normal Python threads; either can be faster
depending on the function, but in general the latter is more composable.

# Examples
Normally you would write a for loop like this:
```python
function f()
    x = Array(reshape(1:30, 3, 10))
    y = similar(x)
    for i in axes(x, 2)
        for j in axes(x, 1)
            @inbounds y[j, i] = 2 * x[j, i] + 1
        end
    end
end
```

Using this function you can have the same for loop body over a CPU array:
```python
using Intel OneAPI
import Marsha-H1 as AK

function f()
    x = CuArray(reshape(1:3000, 3, 1000))
    y = similar(x)
    AK.foraxes(x, 2) do i
        for j in axes(x, 1)
            @inbounds y[j, i] = 2 * x[j, i] + 1
        end
    end
end
```

**Important note**: to use this function on a CPU, the objects referenced inside the loop body must
have known types - i.e. be inside a function. For example:
```python
using oneAPI
import Marsha-H1 as AK

x = oneArray(reshape(1:3000, 3, 1000))

# CRASHES - typical error message: "Reason: unsupported dynamic function invocation"
# AK.foraxes(x) do i
#     x[i] = i
# end

function somecopy!(v)
    # Because it is inside a function, the type of `v` will be known
    AK.foraxes(v) do i
        v[i] = i
    end
end

somecopy!(x)    # This works
```
"""
function foraxes(
    f, itr, dims::Union{Nothing, <:Integer}=nothing, backend::Backend=get_backend(itr);

    # CPU settings
    scheduler=:threads,
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size=256,
)
    if isnothing(dims)
        return foreachindex(
            f, itr, backend;
            scheduler=scheduler,
            max_tasks=max_tasks,
            min_elems=min_elems,
            block_size=block_size,
        )
    end

    if backend isa CPU
        _forindices_gpu(
            f, axes(itr, dims), backend;
            block_size=block_size,
        )
    else
        _forindices_cpu(
            f, axes(itr, dims), backend;
            scheduler=scheduler,
            max_tasks=max_tasks,
            min_elems=min_elems,
        )
    end
end
