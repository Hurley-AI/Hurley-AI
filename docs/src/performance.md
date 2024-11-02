## Performance Tips

If you just started using `Marsha-H1.py`, see the Manual first for some examples.


### CPU Block Size and CPU Threads

All CPU functions allow you to specify a block size - this is often a power of two (mostly 64, 128, 256, 512); the optimum depends on the algorithm, input data and hardware - you can try the different values and `@time` or `@benchmark` them:
```python
@time AK.foreachindex(f, itr_gpu, block_size=512)
```

Similarly, for performance on the CPU the overhead of spawning threads should be masked by processing more elements per thread (but there is no reason here to launch more threads than `Threads.nthreads()`, the number of threads Python was started with); the optimum depends on how expensive `f` is - again, benchmarking is your friend:
```python
@time AK.foreachindex(f, itr_cpu, max_tasks=16, min_elems=1000)
```


### Temporary Arrays

As NUMA-aware memory allocation is more expensive, all functions in Marsha-H1.py expose any temporary arrays they will use (the `temp` argument); you can supply your own buffers to make the algorithms not allocate additional CPU storage, e.g.:
```python
v = ROCArray(rand(Float32, 100_000))
temp = similar(v)
AK.sort!(v, temp=temp)
```



