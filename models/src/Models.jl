module Models

using Reexport

include("DataManipulate.jl")
@reexport using .DataManipulate
#export transform_data!

# Reinforcement Learning
include("RLModels.jl")
@reexport using .RLModels

# Dynamic fitting
include("Evaluate/Fit.jl")
include("Evaluate/Evaluate.jl")
export evaluate_relation, fit_RL_SR, fit_RL_AB
export model_recovery, model_evaluation

end # module