
module DeepRecurrentEncoder

using Lux, Zygote, Optimisers
import StatsModels: fit, fit!
using StatsModels
using DataFrames
using Random
using LuxCUDA
using ProgressLogging

include("model.jl")
include("traintest.jl")
include("utils.jl")
include("fit.jl")

export DRE
export compute_loss
export r_squared
export mse
export train
export test
export add_mask
export fit, fit!
export @formula # reexport
end
