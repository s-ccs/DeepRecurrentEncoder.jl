using DeepRecurrentEncoder
using Documenter
using Literate, Glob
DocMeta.setdocmeta!(DeepRecurrentEncoder, :DocTestSetup, :(using DeepRecurrentEncoder); recursive=true)


GENERATED = joinpath(@__DIR__, "src", "generated")
SOURCE = joinpath(@__DIR__, "literate")

for subfolder ∈ ["explanations", "howto", "tutorials", "reference"]
    local SOURCE_FILES = Glob.glob(subfolder * "/*.jl", SOURCE)
    #config=Dict(:repo_root_path=>"https://github.com/unfoldtoolbox/UnfoldSim")
    foreach(fn -> Literate.markdown(fn, GENERATED * "/" * subfolder), SOURCE_FILES)

end



# makedocs(;
#     modules=[DeepRecurrentEncoder],
#     authors="Benedikt V. Ehinger",
#     sitename="DeepRecurrentEncoder.jl",
#     format=Documenter.HTML(;
#         canonical="https://s-ccs.github.io/DeepRecurrentEncoder.jl",
#         edit_link="main",
#         assets=String[],
#     ),
#     pages=[
#         "Home" => "index.md",
#         "Getting Started" => "generated/tutorials/gettingstarted.md",
#         "Team_report" => "generated/tutorials/Autoencoder_EEG_Meeting_Minutes.md"
#     ],
# )

makedocs(
     modules=[DeepRecurrentEncoder],
     authors="Benedikt V. Ehinger",
     sitename="DeepRecurrentEncoder.jl",
     format=Documenter.HTML(;
         canonical="https://s-ccs.github.io/DeepRecurrentEncoder.jl",
         edit_link="main",
         assets=String[],
    ),
    pages = Any[
        "Home" => "index.md",
        "Introduction"=>[
            "Basics" => "generated/explanations/introduction.md",
            "Installation" => "generated/explanations/installation.md",
        ],
        "How To"=>[
            "Basic Usage" => "generated/howto/basicUsage.md",
        ],
        "Tutorials" => [
            "Effect of hidden channels" => "generated/tutorials/hiddenChannelsEffect.md",
            "Effect of stimuli input" => "generated/tutorials/stimuliEffect.md",
            # "Getting Started" => "generated/tutorials/gettingstarted.md",
            # "Autoencoder EEG Meeting Minutes" => "generated/tutorials/Autoencoder_EEG_Meeting_Minutes.md",
        ]
    ]
)


deploydocs(;
    repo="github.com/s-ccs/DeepRecurrentEncoder.jl",
    devbranch="main",
    push_preview = true,
)