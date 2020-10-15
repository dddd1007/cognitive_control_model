push!(LOAD_PATH, "/Users/dddd1007/project2git/cognitive_control_model/models")
using DataManipulate, RLModels_basic, DataFramesMeta, CSV
using Hyperopt, Plots
import RLModels_SoftMax, RLModels_no_SoftMax

# 生成测试数据
# include("init_sub1_data.jl")

function hyperopt_rllearn_basic(env, realsub, looptime)
    
    ho = @phyperopt for i = looptime,
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

function hyperopt_rllearn_witherror(env, realsub, looptime)
    ho = @phyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = RLModels_no_SoftMax.Learner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
        model_stim = RLModels_no_SoftMax.rl_learning_sr(env, agent, realsub)
        RLModels_basic.evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    vis = plot(ho)
    optim_param, eval_result = minimum(ho)
    return (optim_param, eval_result, vis)
end

function hyperopt_rllearn_withCCC(env, realsub, looptime)
    ho = @phyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;],
                        α_s_CCC = [0.01:0.01:1;], 
                        α_v_CCC = [0.01:0.01:1;],
                        CCC = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = RLModels_no_SoftMax.Learner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
        model_stim = RLModels_no_SoftMax.rl_learning_sr(env, agent, realsub)
        RLModels_basic.evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    vis = plot(ho)
    optim_param, eval_result = minimum(ho)
    return (optim_param, eval_result, vis)
end