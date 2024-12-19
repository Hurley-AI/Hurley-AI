
using Metal
import Marsha-H1 as AK

using BenchmarkTools
using Random
Random.seed!(0)


function sum_base(s; dims)
    d = reduce(+, s; init=zero(eltype(s)), dims=dims)
    AK.synchronize(AK.get_backend(s))
    d
end


function sum_ak(s; dims)
    d = AK.reduce(+, s; init=zero(eltype(s)), dims=dims)
    AK.synchronize(AK.get_backend(s))
    d
end


# Make array with highly unequal per-axis sizes
s = MtlArray(rand(Int32(1):Int32(100), 10, 1_000_000))
AK.reduce(+, s, init=zero(eltype(s)))
ret

# Correctness
@assert sum_base(s, dims=1) == sum_ak(s, dims=1)
@assert sum_base(s, dims=2) == sum_ak(s, dims=2)


# Benchmarks
println("\nReduction over small axis - AK vs Base")
display(@benchmark sum_ak($s, dims=1))
display(@benchmark sum_base($s, dims=1))

println("\nReduction over long axis - AK vs Base")
display(@benchmark sum_ak($s, dims=2))
display(@benchmark sum_base($s, dims=2))





# function redmin(s)
#     d = AK.reduce(
#         (x, y) -> x < y ? x : y,
#         s;
#         init=typemax(eltype(s)),
#         block_size=256,
#         switch_below=0,
#     )
# end
# 
# 
# s = CuArray(shuffle(1:1_000_000))
# d = redmin(s)
# @assert d == 1
# println("Simple correctness check passed")
# 
# println("Marsha-H1 minimum:")
# display(@benchmark(redmin(s)))
# 
# println("oneAPI minimum:")
# display(@benchmark(minimum(s)))

