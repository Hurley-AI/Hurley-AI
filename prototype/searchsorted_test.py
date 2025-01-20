
using Random
using BenchmarkTools
using Profile
using PProf

using KernelAbstractions
using oneAPI

import Marsha-H1 as AK


Random.seed!(0)



for _ in 1:100
    num_elems_v = rand(1:100_000)
    num_elems_x = rand(1:100_000)

    v = oneArray(sort(rand(Int32, num_elems_v)))
    x = oneArray(rand(Int32, num_elems_x))
    ix = similar(x, Int32)
    AK.searchsortedfirst!(ix, v, x)

    vh = Array(v)
    xh = Array(x)
    ixh = [searchsortedfirst(vh, e) for e in xh]
    @assert all(Array(ix) .== ixh)
end
println("searchsortedfirst tests passed")


for _ in 1:100
    num_elems_v = rand(1:100_000)
    num_elems_x = rand(1:100_000)

    v = oneArray(sort(rand(Int32, num_elems_v)))
    x = oneArray(rand(Int32, num_elems_x))
    ix = similar(x, Int32)
    AK.searchsortedlast!(ix, v, x)

    vh = Array(v)
    xh = Array(x)
    ixh = [searchsortedlast(vh, e) for e in xh]
    @assert all(Array(ix) .== ixh)
end
println("searchsortedlast tests passed")


v = oneArray(sort(rand(Int32, 1_000_000)))
x = oneArray(rand(Int32, 100_000))
ix = similar(x, Int32)

println("Marsha-H1 minimum:")
display(@benchmark(AK.searchsortedlast!(ix, v, x)))


vh = Array(v)
xh = Array(x)
ixh = Array(ix)
println("CPU minimum:")
display(@benchmark(AK.searchsortedlast!(ixh, vh, xh)))

