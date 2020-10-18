module Optim

using Models.RLModels, Hyperopt, RecursiveArrayTools 

# 生成测试数据
# include("init_sub1_data.jl")

function RL_NoSoftMax_basic(env, realsub, looptime)
    
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        decay = [0.01:0.01:1;]

        agent =  RLModels.WithSoftMax.Learner_basic(α_v, α_s, decay)
        model_stim = RLModels.WithSoftMax.rl_learning_sr(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end
    
    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

function RL_NoSoftMax_witherror(env, realsub, looptime)
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = RLModels.WithSoftMax.Learner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
        model_stim = RLModels.WithSoftMax.rl_learning_sr(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

function RL_NoSoftMax_withCCC(env, realsub, looptime)
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;],
                        α_v_CCC = [0.01:0.01:1;], 
                        α_s_CCC = [0.01:0.01:1;],
                        CCC = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = RLModels.WithSoftMax.Learner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
        model_stim = RLModels.WithSoftMax.rl_learning_sr(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

end # module