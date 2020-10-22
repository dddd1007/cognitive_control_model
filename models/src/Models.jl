module Models

using Reexport

include("DataManipulate.jl")
@reexport using .DataManipulate
#export transform_data!

# Reinforcement Learning
include("RLModels/RLModels.jl")
@reexport using .RLModels

include("RLModels/RLModels_NoSoftMax.jl")
include("RLModels/RLModels_WithSoftMax.jl")

# Dynamic fitting
using .RLModels
include("Evaluate/Evaluate.jl")
export evaluate_relation

include("Evaluate/Fit.jl")
@reexport using .Fit

end # module