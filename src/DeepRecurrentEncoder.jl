module DeepRecurrentEncoder

using Lux, Zygote, Optimisers

include("model.jl")
include("dre_simple.jl")
include("traintest.jl")
include("utils.jl")


export DRE
export compute_loss
export train
export test
export add_mask

end
