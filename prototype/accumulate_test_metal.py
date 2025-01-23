
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using Metal

import Marsha-H1 as AK


Random.seed!(0)


v = Metal.ones(Int32, 3, 20)

v2 = AK.accumulate!(+, copy(v), init=zero(eltype(v)), dims=2, block_size=8)

@assert Array(v2) == cumsum(Array(v), dims=2)

v2
