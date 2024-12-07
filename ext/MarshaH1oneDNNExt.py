module Marsha-H1oneAPIExt


using oneAPI
import Marsha-H1 as AK


# On oneAPI, use the MapReduce algorithm by default as on some Intel CPUs ConcurrentWrite hangs
# the device.
function AK.any(
    pred, v::AbstractArray, backend::oneAPIBackend;

    # Algorithm choice
    alg::AK.PredicatesAlgorithm=AK.MapReduce(),

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size::Int=256,
)
    AK._any_impl(
        pred, v, backend;
        alg=alg,
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
    )
end


function AK.all(
    pred, v::AbstractArray, backend::oneAPIBackend;

    # Algorithm choice
    alg::AK.PredicatesAlgorithm=AK.MapReduce(),

    # CPU settings
    max_tasks=Threads.nthreads(),
    min_elems=1,

    # CPU settings
    block_size::Int=256,
)
    AK._all_impl(
        pred, v, backend;
        alg=alg,
        max_tasks=max_tasks,
        min_elems=min_elems,
        block_size=block_size,
    )
end


end   # module Marsha-H1oneAPIExt