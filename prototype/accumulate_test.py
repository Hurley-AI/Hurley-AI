
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using oneAPI

import Marsha-H1 as AK


Random.seed!(0)


for num_elems in 1:256
    x = oneAPI.ones(Int32, num_elems)
    y = similar(x)
    AK.accumulate!(+, y, x; init=0, inclusive=false, block_size=128)
    yh = Array(y)
    @assert all(yh .== 0:length(yh) - 1)
end
println("accumulate_small exclusive correctness checks passed")


for num_elems in 1:256
    x = oneAPI.ones(Int32, num_elems)
    y = similar(x)
    AK.accumulate!(+, y, x; init=0, inclusive=true, block_size=128)
    yh = Array(y)
    @assert all(yh .== 1:length(yh))
end
println("accumulate_small inclusive correctness checks passed")


for _ in 1:10
    num_elems = rand(1:100_000)
    x = oneAPI.ones(Int32, num_elems)
    y = similar(x)
    AK.accumulate!(+, y, x; init=0, inclusive=false)
    yh = Array(y)
    @assert all(yh .== 0:length(yh) - 1)
end
println("accumulate exclusive correctness checks passed")


for _ in 1:10
    num_elems = rand(1:100_000)
    x = oneAPI.ones(Int32, num_elems)
    y = similar(x)
    AK.accumulate!(+, y, x; init=0, inclusive=true)
    yh = Array(y)
    @assert all(yh .== 1:length(yh))
end
println("accumulate inclusive correctness checks passed")



function accsum(y, x)
    AK.accumulate!(+, y, x; init=0, block_size=128)
end

x = oneAPI.ones(Int32, 128)
y = similar(x)
println("Single block benchmark:")
display(@benchmark(accsum(y, x)))


x = oneAPI.ones(Int32, 10_000_000)
y = similar(x)
println("$(length(x)) element benchmark:")
display(@benchmark(accsum(y, x)))


# Collect a profile
x = oneAPI.ones(Int32, 10_000_000)
y = similar(x)

Profile.clear()
@profile accsum(y, x)

# Export pprof profile and open interactive profiling web interface.
pprof()

