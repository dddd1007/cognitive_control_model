
using .RLModels
using Hyperopt, RecursiveArrayTools, StatsBase, DataFrames, GLM 

#####
##### 强化学习模型的模型拟合
#####

### 估计SR模型
function fit_RL_SR(env, realsub, looptime; model_type)
    
    ## Fit the hyperparameters
    if model_type == :_1a
    ho = @hyperopt for i = looptime,
                        α = [0.001:0.001:0.999;]

        agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, 0)
        model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub, dodecay = false)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end
    elseif model_type == :_1a1d
        ho = @hyperopt for i = looptime,
                            α = [0.001:0.001:0.999;],
                            decay = [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1e
        ho = @hyperopt for i = looptime,
                           α = [0.001:0.001:0.999;],
                           α_error = [0.001:0.001:1;],
                           decay = [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α, α, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1e1CCC
        ho = @hyperopt for i = looptime,
                        α = [0.001:0.001:1;],
                        α_error = [0.001:0.001:1;],
                        α_CCC = [0.001:0.001:1;],
                        CCC = [0.001:0.001:1;], 
                        decay = [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α, α, α_error, α_error, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1CCC
        ho = @hyperopt for i = looptime,
                         α = [0.001:0.001:1;],
                       α_CCC = [0.001:0.001:1;],
                         CCC = [0.001:0.001:1;], 
                       decay = [0.001:0.001:1;]
    
            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α, α, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a
        ho = @hyperopt for i = looptime,
                            α_v = [0.001:0.001:1;],
                            α_s = [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub, dodecay = false)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d
        ho = @hyperopt for i = looptime,
                            α_v = [0.001:0.001:1;],
                            α_s = [0.001:0.001:1;],
                            decay = [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1e
        ho = @hyperopt for i = looptime,
                        α_v = [0.001:0.001:1;],
                        α_s = [0.001:0.001:1;],
                        α_error = [0.001:0.001:1;], 
                        decay = [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1e1CCC
        ho = @hyperopt for i = looptime,
                        α_v = [0.001:0.001:1;],
                        α_s = [0.001:0.001:1;],
                        α_error = [0.001:0.001:1;],
                        α_CCC = [0.001:0.001:1;],
                        CCC = [0.001:0.001:1;], 
                        decay = [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_error, α_error, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1CCC
        ho = @hyperopt for i = looptime,
                         α_v = [0.001:0.001:1;],
                         α_s = [0.001:0.001:1;],
                       α_CCC = [0.001:0.001:1;],
                         CCC = [0.001:0.001:1;], 
                       decay = [0.001:0.001:1;]
    
            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α_v, α_s, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    end

    optim_params_value, eval_result = minimum(ho)
    optim_params = Dict(zip(ho.params, optim_params_value))
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[!, :MSE] = ho.results

    return (optim_params, eval_result, verbose_table)
end

### 估计AB模型
function fit_RL_AB(env, realsub, looptime; model_type)

    # Fit the hyperparameters
    if model == :basic
        ho = @hyperopt for i = looptime,
                        α_v = [0.001:0.001:1;],
                        α_s = [0.001:0.001:1;],
                        decay = [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
            evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
        end
    elseif model == :error
        ho = @hyperopt for i = looptime,
                        α_v = [0.001:0.001:1;],
                        α_s = [0.001:0.001:1;],
                        α_v_error = [0.001:0.001:1;],
                        α_s_error = [0.001:0.001:1;], 
                        decay = [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
            evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
        end
    elseif model == :CCC
        ho = @hyperopt for i = looptime,
                            α_v = [0.001:0.001:1;],
                            α_s = [0.001:0.001:1;],
                            α_v_error = [0.001:0.001:1;],
                            α_s_error = [0.001:0.001:1;],
                            α_v_CCC = [0.001:0.001:1;], 
                            α_s_CCC = [0.001:0.001:1;],
                            CCC = [0.001:0.001:1;], 
                            decay = [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
            evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
        end
    end
    
    optim_params_value, eval_result = minimum(ho)
    optim_params = Dict(zip(ho.params, optim_params_value))
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_params, eval_result, verbose_table)
end