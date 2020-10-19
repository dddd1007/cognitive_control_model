module Models

export transform_data!,
       RL_NoSoftMax_basic, RL_NoSoftMax_witherror, RL_NoSoftMax_withCCC,
       ExpEnv, RealSub

include("DataManipulate.jl")

# Reinforcement Learning
include("RLModels/RLModels.jl")

# Dynamic fitting
include("Optim.jl")

end # module