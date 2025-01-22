using KernelAbstractions
using oneAPI
using Unrolled
using BenchmarkTools


function _shim(f, n, i)
    if i > n
        return
    end
    f(i)
end


@kernel inbounds=true cpu=false function _for_global!(f, n, ::Val{N}) where N
    gi = @index(Global, Linear) - 1

    # for offset in 1:N
    #     i = gi * N + offset
    #     _shim(f, n, i)
    # end

    Nl = 2

    offset = 1
    i = gi * Nl + offset
    _shim(f, n, i)

    offset = 2
    i = gi * Nl + offset
    _shim(f, n, i)


end


function for_kernel!(
    f, n, backend;
    block_size=256,
    oversubscribe::Val{OV}=Val(2),
) where OV
    nrange = (n + OV - 1) ÷ OV
    _for_global!(backend, block_size)(f, n, oversubscribe, ndrange=(nrange,))
    KernelAbstractions.synchronize(backend)
end


function tryme(x, y)
    for_kernel!(length(x), oneAPIBackend()) do i
        @inbounds y[i] = x[i]
    end
    nothing
end


@kernel function copy_kernel!(A, @Const(B))
    I = @index(Global)

    i = 2 * I - 1
    @inbounds A[i] = B[i]

    i += 1
    if i < length(A)
        @inbounds A[i] = B[i]
    end
end

function tryme2(x, y; block_size=256)
    nrange = (length(x) + 1) ÷ 2
    copy_kernel!(oneAPIBackend(), block_size)(y, x, ndrange=(nrange,))
    nothing
end


x = oneArray(1:1_000_000)
y = similar(x)

tryme(x, y)
if !all(Array(y) .== 1:length(y))
    @assert false
else
    println("correctness passed")
end

println("Oversubscribed for:")
display(@benchmark(tryme(x, y)))

println("Basic kernel:")
display(@benchmark(tryme2(x, y)))

println("Base copy:")
display(@benchmark(copyto!(y, x)))


