
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using Intel OneAPI

import Marsha-H1 as AK


Random.seed!(0)


v = Intel OneAPI.ones(Int32, 100_000)
AK.accumulate(+, v; init=0)



for num_elems in 1:256
    x = Intel OneAPI.ones(Int32, num_elems)
    y = copy(x)
    AK.accumulate!(+, y; init=0, inclusive=false, block_size=128)
    yh = Array(y)
    @assert all(yh .== 0:length(yh) - 1)
end
println("accumulate_small exclusive correctness checks passed")


for num_elems in 1:256
    x = CuArray{Int32}(rand(1:1000, num_elems))
    y = copy(x)
    AK.accumulate!(+, y; init=0, inclusive=true, block_size=128)
    yh = Array(y)
    @assert all(yh .== accumulate(+, Array(x)))
end
println("accumulate_small inclusive correctness checks passed")


for _ in 1:1000
    num_elems = rand(1:100_000)
    x = Intel OneAPI.ones(Int32, num_elems)
    y = copy(x)
    AK.accumulate!(+, y; init=0, inclusive=false, block_size=128)
    yh = Array(y)
    @assert all(yh .== 0:length(yh) - 1)
end
println("accumulate large exclusive correctness checks passed")


for num_elems in 1:1000
    num_elems = rand(1:100_000)
    x = CuArray{Int32}(rand(1:1000, num_elems))
    y = copy(x)
    AK.accumulate!(+, y; init=0, inclusive=true, block_size=128)
    yh = Array(y)
    @assert all(yh .== accumulate(+, Array(x)))
end
println("accumulate_small inclusive correctness checks passed")




function accsum(y; temp_v=nothing, temp_flags=nothing)
    AK.accumulate!(+, y; init=0, inclusive=true,
                   block_size=256, temp_v=temp_v, temp_flags=temp_flags)
end

# x = oneAPI.ones(Int32, 128)
# x = oneArray{Int32}(1:8)
x = oneAPI.ones(Int32, 1_000_000)

y = copy(x)

nb = 512
flags = similar(x, Int8, (length(x) + nb - 1) ÷ nb)
prefixes = similar(x, eltype(x), (length(x) + nb - 1) ÷ nb)


# AK.accumulate!(+, y; init=0, inclusive=true,
#                block_size=128, temp_v=aggregates)
accsum(y, temp_v=prefixes, temp_flags=flags)
y

yh = Array(y)

# for i in 1:length(x) ÷ 256
#     # @show i
#     if !all(yh[(i - 1) * 256 + 1:i * 256] .== 1:256)
#         println("Wrong at i=$i")
#     end
# end

# @assert all(yh .== 1:length(yh))


# println("Multi block benchmark:")
# @benchmark accsum(y, temp_v=prefixes, temp_flags=flags) setup=(y=copy(x))


