
using .RLModels, Hyperopt, RecursiveArrayTools, StatsBase, DataFrames, GLM 

#####
##### 强化学习模型的模型拟合
#####

### 估计SR模型
function fit_RL_SR(env, realsub, looptime; model_type)
    
    ## Fit the hyperparameters
    if model_type == :single_alpha
        ho = @hyperopt for i = looptime,
                            α = [0.01:0.01:1;],
                            decay = [0.01:0.01:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end 
    elseif model_type == :single_alpha_no_decay
        ho = @hyperopt for i = looptime,
                            α = [0.01:0.01:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, 0, dodecay=false)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :no_decay
        ho = @hyperopt for i = looptime,
                            α_v = [0.01:0.01:1;],
                            α_s = [0.01:0.01:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0, dodecay=false)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :total_decay
        ho = @hyperopt for i = looptime,
                           α_v = [0.01:0.01:1;],
                           α_s = [0.01:0.01:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :single_alpha_total_decay
        ho = @hyperopt for i = looptime,
                           α = [0.01:0.01:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, 1)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :basic
        ho = @hyperopt for i = looptime,
                            α_v = [0.01:0.01:1;],
                            α_s = [0.01:0.01:1;],
                            decay = [0.01:0.01:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :error
        ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_error = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :CCC_same_alpha
        ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_error = [0.01:0.01:1;],
                        α_CCC = [0.01:0.01:1;],
                        CCC = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_error, α_error, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :CCC_different_alpha
        ho = @hyperopt for i = looptime,
                            α_v = [0.01:0.01:1;],
                            α_s = [0.01:0.01:1;],
                            α_error = [0.01:0.01:1;],
                            α_v_CCC = [0.01:0.01:1;],
                            α_s_CCC = [0.01:0.01:1;],
                            CCC = [0.01:0.01:1;], 
                            decay = [0.01:0.01:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_error, α_error, α_v_CCC, α_s_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    end

    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

### 估计AB模型
function fit_RL_AB(env, realsub, looptime; model_type)

    # Fit the hyperparameters
    if model == :basic
        ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        decay = [0.01:0.01:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
            evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
        end
    elseif model == :error
        ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
            evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
        end
    elseif model == :CCC
        ho = @hyperopt for i = looptime,
                            α_v = [0.01:0.01:1;],
                            α_s = [0.01:0.01:1;],
                            α_v_error = [0.01:0.01:1;],
                            α_s_error = [0.01:0.01:1;],
                            α_v_CCC = [0.01:0.01:1;], 
                            α_s_CCC = [0.01:0.01:1;],
                            CCC = [0.01:0.01:1;], 
                            decay = [0.01:0.01:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
            evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
        end
    end
    
    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end