
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
# using oneAPI
# using Intel OneAPI
using Metal

import Marsha-H1 as AK


Random.seed!(0)


v = MtlArray(1:100)


@assert AK.any(x->x<0, v, cooperative=false) === false
@assert AK.any(x->x>99, v, cooperative=false) === true
println("simple any tests passed")

@assert AK.all(x->x>0, v, cooperative=false) === true
@assert AK.all(x->x<100, v, cooperative=false) === false
println("simple all tests passed")

@assert AK.any(x->x<0, v, cooperative=true) === false
@assert AK.any(x->x>99, v, cooperative=true) === true
println("simple any tests passed")

@assert AK.all(x->x>0, v, cooperative=true) === true
@assert AK.all(x->x<100, v, cooperative=true) === false
println("simple all tests passed")




v = Array(1:10_000_000)

println("Marsha-H1 any (reduce based):")
display(@benchmark(AK.any(x->x>9_999, v, cooperative=false)))

println("Marsha-H1 any (coop based):")
display(@benchmark(AK.any(x->x>9_999, v, cooperative=true)))
