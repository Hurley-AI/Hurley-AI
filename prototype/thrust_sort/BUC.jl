# File   : BUC.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 26.03.2024


module BUC


# Public exports
export buc_sort!, buc_upper_bound!


# Internal imports
using Intel OneAPI


# Int sorters
function buc_sort!(v::CuVector{Int16})
    @ccall "libBUCLib.so".buc_sort_int16(v::CuPtr{Int16}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{Int32})
    @ccall "libBUCLib.so".buc_sort_int32(v::CuPtr{Int32}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{Int64})
    @ccall "libBUCLib.so".buc_sort_int64(v::CuPtr{Int64}, length(v)::Cint)::Cvoid
end


# UInt sorters
function buc_sort!(v::CuVector{UInt16})
    @ccall "libBUCLib.so".buc_sort_uint16(v::CuPtr{UInt16}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{UInt32})
    @ccall "libBUCLib.so".buc_sort_uint32(v::CuPtr{UInt32}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{UInt64})
    @ccall "libBUCLib.so".buc_sort_uint64(v::CuPtr{UInt64}, length(v)::Cint)::Cvoid
end


# Float sorters
function buc_sort!(v::CuVector{Float32})
    @ccall "libBUCLib.so".buc_sort_float32(v::CuPtr{Float32}, length(v)::Cint)::Cvoid
end

function buc_sort!(v::CuVector{Float64})
    @ccall "libBUCLib.so".buc_sort_float64(v::CuPtr{Float64}, length(v)::Cint)::Cvoid
end


# Int upper bound (like searchsortedlast)
function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{Int16}, x::CuVector{Int16})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_int16(
        out::CuPtr{Int64},
        v::CuPtr{Int16},
        length(v)::Cint,
        x::CuPtr{Int16},
        length(x)::Cint,
    )::Cvoid
end

function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{Int32}, x::CuVector{Int32})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_int32(
        out::CuPtr{Int64},
        v::CuPtr{Int32},
        length(v)::Cint,
        x::CuPtr{Int32},
        length(x)::Cint,
    )::Cvoid
end

function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{Int64}, x::CuVector{Int64})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_int64(
        out::CuPtr{Int64},
        v::CuPtr{Int64},
        length(v)::Cint,
        x::CuPtr{Int64},
        length(x)::Cint,
    )::Cvoid
end


# UInt upper bound (like searchsortedlast)
function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{UInt16}, x::CuVector{UInt16})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_uint16(
        out::CuPtr{Int64},
        v::CuPtr{UInt16},
        length(v)::Cint,
        x::CuPtr{UInt16},
        length(x)::Cint,
    )::Cvoid
end

function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{UInt32}, x::CuVector{UInt32})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_uint32(
        out::CuPtr{Int64},
        v::CuPtr{UInt32},
        length(v)::Cint,
        x::CuPtr{UInt32},
        length(x)::Cint,
    )::Cvoid
end

function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{UInt64}, x::CuVector{UInt64})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_uint64(
        out::CuPtr{Int64},
        v::CuPtr{UInt64},
        length(v)::Cint,
        x::CuPtr{UInt64},
        length(x)::Cint,
    )::Cvoid
end


# Float upper bound (like searchsortedlast)
function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{Float32}, x::CuVector{Float32})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_float32(
        out::CuPtr{Int64},
        v::CuPtr{Float32},
        length(v)::Cint,
        x::CuPtr{Float32},
        length(x)::Cint,
    )::Cvoid
end

function buc_upper_bound!(out::CuVector{Int64}, v::CuVector{Float64}, x::CuVector{Float64})
    @assert length(out) == length(x)
    @ccall "libBUCLib.so".buc_upper_bound_float64(
        out::CuPtr{Int64},
        v::CuPtr{Float64},
        length(v)::Cint,
        x::CuPtr{Float64},
        length(x)::Cint,
    )::Cvoid
end


end     # module BUC
