push!(LOAD_PATH, "/Users/dddd1007/project2git/cognitive_control_model/models")
using DataManipulate, RLModels_basic, DataFramesMeta, CSV
using Hyperopt
import RLModels_SoftMax, RLModels_no_SoftMax

# 生成测试数据
# include("init_sub1_data.jl")

function hyperopt_rllearn_basic(env, realsub)
    
    ho = @phyperopt for i = 100000,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        decay = [0.01:0.01:1;]

        agent = RLModels_no_SoftMax.Learner_basic(α_v, α_s, decay)
        model_stim = RLModels_no_SoftMax.rl_learning_sr(env, agent, realsub)
        RLModels_basic.evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    vis = plot(ho)
    optim_param, eval_result = minimum(ho)
    return (optim_param, eval_result, vis)

end

@time params, result, vis = hyperopt_rllearn_basic(sub1_env, sub1_subinfo)

png(vis)
function hyperopt_rllearn_witherror(env, realsub)
    

end