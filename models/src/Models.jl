module Models

using DataFrames, DataFramesMeta, GLM, StatsBase, RecursiveArrayTools

export transform_data!
export ExpEnv, RealSub
export evaluate_relation, init_env_sub
export update_options_weight_matrix, init_param
export calc_CCC
export RLModels
export Optim

include("DataManipulate.jl")

# Reinforcement Learning
include("RLModels/RLModels.jl")

# Dynamic fitting
include("Optim.jl")
end # module