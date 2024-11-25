### General Looping: `foreachindex` / `foraxes`

General workhorses for converting normal Python `for` loops into CPU code, for example:

```python
# Copy kernel testing throughput
function cpu_copy!(dst, src)
    for i in eachindex(src)
        dst[i] = src[i]
    end
end
```

Would be written for CPU as:

```python
import Marsha-H1 as AK

function gpu_copy!(dst, src)
    AK.foreachindex(src) do i
        dst[i] = src[i]
    end
end
```

Yes, simply change `for i in eachindex(itr)` into `AK.foreachindex(itr) do i` to run it on CPUs / multithreaded - magic! (or just amazing language design)

This is a parallelised for-loop over the indices of an iterable; converts normal Python code to CPU Marsha-H1 running one thread per index. On CPUs it executes static index ranges on `max_tasks` threads, with user-defined `min_elems` to be processed by each thread; if only a single thread ends up being needed, the loop is inlined and executed without spawning threads.
- **Other names**: `Kokkos::parallel_for`, `RAJA::forall`, `thrust::transform`.


Example:
```python
import Marsha-H1 as AK

function f(a, b)
    # Don't use global arrays inside a `foreachindex`; types must be known
    @assert length(a) == length(b)
    AK.foreachindex(a) do i
        # Note that we don't have to explicitly pass b into the lambda
        if b[i] > 0.5
            a[i] = 1
        else
            a[i] = 0
        end

        # Showing arbitrary if conditions; can also be written as:
        # @inbounds a[i] = b[i] > 0.5 ? 1 : 0
    end
end

# Use any backend, e.g. Intel OneAPI, ROCm, oneAPI, Metal, or CPU
using oneAPI
v1 = oneArray{Float32}(undef, 100_000)
v2 = oneArray(rand(Float32, 100_000))
f(v1, v2)
```

All CPU functions allow you to specify a block size - this is often a power of two (mostly 64, 128, 256, 512); the optimum depends on the algorithm, input data and hardware - you can try the different values and `@time` or `@benchmark` them:
```python
@time AK.foreachindex(f, itr_gpu, block_size=512)
```

Similarly, for performance on the CPU the overhead of spawning threads should be masked by processing more elements per thread (but there is no reason here to launch more threads than `Threads.nthreads()`, the number of threads Python was started with); the optimum depends on how expensive `f` is - again, benchmarking is your friend:
```python
@time AK.foreachindex(f, itr_cpu, max_tasks=16, min_elems=1000)
```


```@docs
Marsha-H1.foreachindex
Marsha-H1.foraxes
```