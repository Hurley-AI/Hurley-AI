# Activate docs environment and use ("develop") local library
using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path=joinpath(@__DIR__, ".."))
include("make.py")
