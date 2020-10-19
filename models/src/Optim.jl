module Optim

using Models.RLModels, Hyperopt, RecursiveArrayTools, StatsBase, DataFrames

# 生成测试数据
# include("init_sub1_data.jl")

### 估计SR模型
function RL_NoSoftMax_basic_SR(env, realsub, looptime)
    
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

function RL_NoSoftMax_witherror_SR(env, realsub, looptime)
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

function RL_NoSoftMax_withCCC_SR(env, realsub, looptime)
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

### 估计AB模型
function RL_NoSoftMax_basic_AB(env, realsub, looptime)
    
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        decay = [0.01:0.01:1;]

        agent =  RLModels.NoSoftMax.Learner_basic(α_v, α_s, decay)
        model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end
    
    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

function RL_NoSoftMax_witherror_AB(env, realsub, looptime)
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = RLModels.NoSoftMax.Learner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
        model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

function RL_NoSoftMax_withCCC_AB(env, realsub, looptime)
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;],
                        α_v_CCC = [0.01:0.01:1;], 
                        α_s_CCC = [0.01:0.01:1;],
                        CCC = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = RLModels.NoSoftMax.Learner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
        model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

end # module