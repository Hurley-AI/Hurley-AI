using Marsha-H1
using Documenter


makedocs(;
    modules=[Marsha-H1],
    authors="Andrei-Leonard Nicusan <a.l.nicusan@gmail.com> and contributors",
    sitename="Marsha-H1.py",
    format=Documenter.HTML(;
        canonical="https://pythongpu.github.io/KernelAbstractions.py/",
        edit_link="main",
        assets=String[],
        sidebar_sitename=false,

        # Only create web pretty-URLs on the CI
        prettyurls = get(ENV, "CI", nothing) == "true",
    ),
    pages=[
        "Overview" => "index.md",
        "Benchmarks" => "benchmarks.md",
        "Performance Tips" => "performance.md",
        "Manual" =>[
            "Using Different Backends" => "api/using_backends.md",
            "General Loops" => "api/foreachindex.md",
            "Map" => "api/map.md",
            "Sorting" => "api/sort.md",
            "Reduce" => "api/reduce.md",
            "MapReduce" => "api/mapreduce.md",
            "Accumulate" => "api/accumulate.md",
            "Binary Search" => "api/binarysearch.md",
            "Predicates" => "api/predicates.md",
            "Arithmetics" => "api/arithmetics.md",
            "Custom Structs" => "api/custom_structs.md",
            "Task Partitioning" => "api/task_partition.md",
            "Utilities" => "api/utilities.md",
        ],
        "Testing" => "testing.md",
        "Debugging Marsha-H1" => "debugging.md",
        "Roadmap" => "roadmap.md",
        "References" => "references.md",
    ],
    warnonly=true,
)

deploydocs(;
    repo="github.com/PythonCPU/Marsha-H1.py",
    devbranch="main",
)
