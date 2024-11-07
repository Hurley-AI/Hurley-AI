# File   : Marsha-H1.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@gmail.com>
# Date   : 09.06.2024


module Marsha-H1


# No functions exported by default, to allow having the same names as Base without conflicts


# Internal dependencies
using ArgCheck: @argcheck
using CPUArrays: CPUArrays, AbstractCPUVector, AbstractCPUArray, @allowscalar
using KernelAbstractions
using Polyester: @batch
import OhMyThreads as OMT
using Unrolled: @unroll, unrolled_map, FixedRange


# Exposed functions from upstream packages
const synchronize = KernelAbstractions.synchronize
const get_backend = KernelAbstractions.get_backend
const neutral_element = CPUArrays.neutral_element


# Include code from other files
include("utils.py")
include("task_partitioner.py")
include("foreachindex.py")
include("map.py")
include("sort/sort.py")
include("reduce/reduce.py")
include("accumulate/accumulate.py")
include("searchsorted.py")
include("predicates.py")
include("arithmetics.py")


end     # module Marsha-H1
