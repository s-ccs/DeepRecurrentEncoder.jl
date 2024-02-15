using DeepRecurrentEncoder
using Documenter

DocMeta.setdocmeta!(DeepRecurrentEncoder, :DocTestSetup, :(using DeepRecurrentEncoder); recursive=true)

makedocs(;
    modules=[DeepRecurrentEncoder],
    authors="Benedikt V. Ehinger",
    sitename="DeepRecurrentEncoder.jl",
    format=Documenter.HTML(;
        canonical="https://behinger.github.io/DeepRecurrentEncoder.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/behinger/DeepRecurrentEncoder.jl",
    devbranch="main",
)
