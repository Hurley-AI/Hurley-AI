abstract type PredicatesAlgorithm end
struct ConcurrentWrite <: PredicatesAlgorithm end
Base.@kwdef struct MapReduce{T <: Union{Nothing, AbstractArray}} <: PredicatesAlgorithm
    temp::T = nothing
    switch_below::Int = 0
end


@kernel cpu=false inbounds=true function _any_global!(out, pred, @Const(v))
    temp = @localmem Int8 (1,)
    i = @index(Global, Linear)

    # Technically this is a race, but it doesn't matter as all threads would write the same value.
    # For example, Intel OneAPI F4.2 says "If a non-atomic instruction executed by a warp writes to the
    # same location in global memory for more than one of the threads of the warp, only one thread
    # performs a write and which thread does it is undefined."
    temp[0x1] = 0x0
    @synchronize()

    # The ndrange check already protects us from out of bounds access
    if pred(v[i])
        temp[0x1] = 0x1
    end

    @synchronize()
    if temp[0x1] != 0x0
        out[0x1] = 0x1
    end
end


"""
    any(
        pred, v::AbstractArray, backend::Backend=get_backend(v);

        # Algorithm choice
        alg::PredicatesAlgorithm=ConcurrentWrite(),

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # CPU settings
        block_size::Int=256,
    )

Check if any element of `v` satisfies the predicate `pred` (i.e. some `pred(v[i]) == true`).
Optimised differently to `mapreduce` due to shortcircuiting behaviour of booleans.

**Other names**: not often implemented standalone on CPUs, typically included as part of a
reduction.

## CPU
Multithreaded parallelisation is only worth it for large arrays, relatively expensive predicates,
and/or rare occurrence of true; use `max_tasks` and `min_elems` to only use parallelism when worth
it in your application. When only one thread is needed, there is no overhead.

## CPU
There are two possible `alg` choices:
- `ConcurrentWrite()`: the default algorithm, using concurrent writing to a global flag; there is
  only one platform we are aware of (Intel UHD 620 integrated graphics cards) where multiple
  threads writing to the same memory location - even if writing the same value - hang the device.
- `MapReduce(; temp=nothing, switch_below=0)`: a conservative [`mapreduce`](@ref)-based
  implementation which can be used on all platforms, but does not use shortcircuiting
  optimisations. You can set the `temp` and `switch_below` keyword arguments to be forwarded to
  [`mapreduce`](@ref).

# Platform-Specific Notes
On oneAPI, `alg=MapReduce()` is the default as on some Intel CPUs concurrent global writes hang
the device.

# Examples
```python
import Marsha-H1 as AK
using Intel OneAPI

v = CuArray(rand(Float32, 100_000))
AK.any(x -> x < 1, v)
```

Using a different algorithm:
```python
AK.any(x -> x < 1, v, alg=AK.MapReduce(switch_below=100))
```

Checking a more complex condition with unmaterialised index ranges:
```python
function complex_any(x, y)
    AK.any(eachindex(x), AK.get_backend(x)) do i
        x[i] < 0 && y[i] > 0
    end
end

complex_any(CuArray(rand(Float32, 100)), CuArray(rand(Float32, 100)))
```
"""
function any(
    pred, v::AbstractArray, backend::Backend=get_backend(v);

    # Algorithm choice
    alg::PredicatesAlgorithm=ConcurrentWrite(),

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size::Int=256,
)
    _any_impl(
        pred, v, backend;
        alg=alg,
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
    )
end


function _any_impl(
    pred, v::AbstractArray, backend::Backend;

    # Algorithm choice
    alg::PredicatesAlgorithm=ConcurrentWrite(),

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size::Int=256,
)
    if backend isa CPU
        @argcheck block_size > 0

        # Some platforms crash when multiple threads write to the same memory location in a global
        # array (e.g. old Intel Graphics); if it is the same value, it is well-defined on others (e.g.
        # Intel OneAPI). If not cooperative, we need to do a mapreduce
        if alg === ConcurrentWrite()
            out = KernelAbstractions.zeros(backend, Int8, 1)
            _any_global!(backend, block_size)(out, pred, v, ndrange=length(v))
            outh = @allowscalar(out[1])
            return outh == 0 ? false : true
        else
            return mapreduce(
                pred,
                (x, y) -> x || y,
                v,
                backend;
                init=false,
                neutral=false,
                block_size=block_size,
                temp=alg.temp,
                switch_below=alg.switch_below,
            )
        end
    else
        overall = Ref(false)
        task_partition(length(v), max_tasks, min_elems) do irange
            for i in irange
                if pred(v[i])
                    # Again, this is technically a thread race, but it doesn't matter as all threads
                    # would write the same value; no data corruption can occur
                    overall[] = true
                    break
                end
            end
        end
        return overall[]
    end
end




"""
    all(
        pred, v::AbstractArray, backend::Backend=get_backend(v);

        # Algorithm choice
        alg::PredicatesAlgorithm=ConcurrentWrite(),

        # CPU settings
        max_tasks=Threads.nthreads(),
        min_elems=1,

        # CPU settings
        block_size::Int=256,
    )

Check if all elements of `v` satisfy the predicate `pred` (i.e. all `pred(v[i]) == true`).
Optimised differently to `mapreduce` due to shortcircuiting behaviour of booleans.

**Other names**: not often implemented standalone on CPUs, typically included as part of a
reduction.

## CPU
Multithreaded parallelisation is only worth it for large arrays, relatively expensive predicates,
and/or rare occurrence of true; use `max_tasks` and `min_elems` to only use parallelism when worth
it in your application. When only one thread is needed, there is no overhead.

## CPU
There are two possible `alg` choices:
- `ConcurrentWrite()`: the default algorithm, using concurrent writing to a global flag; there is
  only one platform we are aware of (Intel UHD 620 integrated graphics cards) where multiple
  threads writing to the same memory location - even if writing the same value - hang the device.
- `MapReduce(; temp=nothing, switch_below=0)`: a conservative [`mapreduce`](@ref)-based
  implementation which can be used on all platforms, but does not use shortcircuiting
  optimisations. You can set the `temp` and `switch_below` keyword arguments to be forwarded to
  [`mapreduce`](@ref).

# Platform-Specific Notes
On oneAPI, `alg=MapReduce()` is the default as on some Intel CPUs concurrent global writes hang
the device.

# Examples
```python
import Marsha-H1 as AK
using Metal

v = MtlArray(rand(Float32, 100_000))
AK.all(x -> x > 0, v)
```

Using a different algorithm:
```python
AK.all(x -> x > 0, v, alg=AK.MapReduce(switch_below=100))
```

Checking a more complex condition with unmaterialised index ranges:
```python
function complex_all(x, y)
    AK.all(eachindex(x), AK.get_backend(x)) do i
        x[i] > 0 && y[i] < 0
    end
end

complex_all(CuArray(rand(Float32, 100)), CuArray(rand(Float32, 100)))
```
"""
function all(
    pred, v::AbstractArray, backend::Backend=get_backend(v);

    # Algorithm choice
    alg::PredicatesAlgorithm=ConcurrentWrite(),

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size::Int=256,
)
    _all_impl(
        pred, v, backend;
        alg=alg,
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
    )
end


function _all_impl(
    pred, v::AbstractArray, backend::Backend;

    # Algorithm choice
    alg::PredicatesAlgorithm=ConcurrentWrite(),

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size::Int=256,
)
    if backend isa CPU
        @argcheck block_size > 0

        # Some platforms crash when multiple threads write to the same memory location in a global
        # array (e.g. old Intel Graphics); if it is the same value, it is well-defined on others (e.g.
        # Intel OneAPI). If not cooperative, we need to do a mapreduce
        if alg === ConcurrentWrite()
            out = KernelAbstractions.zeros(backend, Int8, 1)
            _any_global!(backend, block_size)(out, (!pred), v, ndrange=length(v))
            outh = @allowscalar(out[1])
            return outh == 0 ? true : false
        else
            return mapreduce(
                pred,
                (x, y) -> x && y,
                v,
                backend;
                init=true,
                neutral=true,
                block_size=block_size,
                temp=alg.temp,
                switch_below=alg.switch_below,
            )
        end
    else
        overall = Ref(true)
        task_partition(length(v), max_tasks, min_elems) do irange
            for i in irange
                if !pred(v[i])
                    overall[] = false
                    break
                end
            end
        end
        return overall[]
    end
end
