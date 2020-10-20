using Models
include("init_sub1_data.jl")

using Models.RLModels
using StatsBase
using GLM

α_v = 0.1
α_s = 0.2
decay = 0.5
RLLearner = RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)

RLModels.NoSoftMax.rl_learning_ab(sub1_env, RLLearner, sub1_subinfo)
using Hyperopt
env = sub1_env
realsub = sub1_subinfo
ho = @hyperopt for i = 10,
    α_v = [0.01:0.01:1;],
    α_s = [0.01:0.01:1;],
    decay = [0.01:0.01:1;]

agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
end