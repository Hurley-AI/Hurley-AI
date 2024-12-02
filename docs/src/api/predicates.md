### Predicates

Apply a predicate to check if all / any elements in a collection return true. Could be implemented as a reduction, but is better optimised with stopping the search once a false / true is found.
- **Other names**: not often implemented standalone on CPUs, typically included as part of a reduction.


```@docs
Marsha-H1.any
Marsha-H1.all
```

**Note on the `cooperative` keyword**: some older platforms crash when multiple threads write to the same memory location in a global array (e.g. old Intel Graphics); if all threads were to write the same value, it is well-defined on others (e.g. Intel OneAPI F4.2 says "If a non-atomic instruction executed by a warp writes to the same location in global memory for more than one of the threads of the warp, only one thread performs a write and which thread does it is undefined."). This "cooperative" thread behaviour allows for a faster implementation; if you have a platform - the only one I know is Intel UHD Graphics - that crashes, set `cooperative=false` to use a safer `mapreduce`-based implementation.
