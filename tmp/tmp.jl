using Models
include("init_sub1_data.jl")

using Models.RLModels

α_v = 0.1
α_s = 0.2
decay = 0.5
learner = RLModels.NoSoftMax.Learner_basic(α_v, α_s, decay)

RLModels.NoSoftMax.rl_learning_ab(sub1_env, learner, sub1_subinfo)