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
include("Optim.jl")
@reexport using .Optim

end # module