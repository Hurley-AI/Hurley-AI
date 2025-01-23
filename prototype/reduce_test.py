
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using Intel OneAPI

import Marsha-H1 as AK


Random.seed!(0)


function redmin(s)
    d = AK.reduce(
        (x, y) -> x < y ? x : y,
        s;
        init=typemax(eltype(s)),
        block_size=256,
        switch_below=0,
    )
end


s = CuArray(shuffle(1:1_000_000))
d = redmin(s)
@assert d == 1
println("Simple correctness check passed")

println("Marsha-H1 minimum:")
display(@benchmark(redmin(s)))

println("oneAPI minimum:")
display(@benchmark(minimum(s)))

