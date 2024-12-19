using Random
# using Intel OneAPI
# using BUC       # Exposed Intel OneAPI Thrust functions, precompiled for common integers
using Metal
import Marsha-H1 as AK


Random.seed!(0)


# Generate random numbers
n = 10_000_000
d = MtlArray{Int64}(undef, n);

# Benchmark
using BenchmarkTools

# println("CPU Sort:")
# display(@benchmark sort!(h) setup=(h=rand(eltype(d), n)))
# 
# println("Intel OneAPI.py Sort:")
# display(@benchmark sort!($d) setup=(rand!(d)))


function aksort!(d, temp)
    AK.merge_sort!(d, temp=temp, block_size=256)
    AK.synchronize(AK.get_backend(d))
    d
end


println("Marsha-H1 Sort:")
temp = similar(d)
display(@benchmark aksort!($d, temp) setup=(rand!(d)))


# println("BUC / Intel OneAPI Thrust Sort:")
# display(@benchmark buc_sort!($d) setup=(rand!(d)))

