using .RLModels
using Hyperopt, RecursiveArrayTools, StatsBase, DataFrames, GLM
using CategoricalArrays

#####
##### 强化学习模型的模型拟合
#####

# Type1 不进行 detrend
function fit_RL_base(env, realsub, looptime; model_type)

    ## Fit the hyperparameters
    if model_type == :_1a
        ho = @hyperopt for i in looptime,
                        α in [0.001:0.001:0.999;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub, dodecay=false)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d
        ho = @hyperopt for i in looptime,
                            α in [0.001:0.001:0.999;],
                            decay in [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1e
        ho = @hyperopt for i in looptime,
                           α in [0.001:0.001:0.999;],
                           α_error in [0.001:0.001:1;],
                           decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α, α, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1e1CCC
        ho = @hyperopt for i in looptime,
                        α in [0.001:0.001:1;],
                        α_error in [0.001:0.001:1;],
                        α_CCC in [0.001:0.001:1;],
                        CCC in [-0.001:-0.001:-1;],
                        decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α, α, α_error, α_error,
                                                         α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_1a1d1CCC
        ho = @hyperopt for i in looptime,
                         α in [0.001:0.001:1;],
                       α_CCC in [0.001:0.001:1;],
                         CCC in [-0.001:-0.001:-1;],
                       decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α, α,
                                                                  α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a
        ho = @hyperopt for i in looptime,
                            α_v in [0.001:0.001:1;],
                            α_s in [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub,
                                                           dodecay=false)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d
        ho = @hyperopt for i in looptime,
                         α_v in [0.001:0.001:1;],
                         α_s in [0.001:0.001:1;],
                       decay in [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1e
        ho = @hyperopt for i in looptime,
                        α_v in [0.001:0.001:1;],
                        α_s in [0.001:0.001:1;],
                    α_error in [0.001:0.001:1;],
                      decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s,
                                                           α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1e1CCC
        ho = @hyperopt for i in looptime,
                        α_v in [0.001:0.001:1;],
                        α_s in [0.001:0.001:1;],
                    α_error in [0.001:0.001:1;],
                      α_CCC in [0.001:0.001:1;],
                        CCC in [-0.001:-0.001:-1;],
                      decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_error, α_error,
                                                         α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :_2a1d1CCC
        ho = @hyperopt for i in looptime,
                         α_v in [0.001:0.001:1;],
                         α_s in [0.001:0.001:1;],
                       α_CCC in [0.001:0.001:1;],
                         CCC in [-0.001:-0.001:-1;],
                       decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α_v, α_s,
                                                                  α_CCC, α_CCC, CCC, decay)
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

# Type2 按照每次改变比例算一个 miniblock 进行 detrend
function fit_RL_detrend_miniblock(env, realsub, looptime; model_type)
    ## Add miniblock
    cache_prop = realsub.prop_seq[1]
    cache_index = 1
    prop_seq_changed = Array{Int64,1}(undef, length(realsub.prop_seq))
    prop_seq_changed[1] = cache_index
    for i in 2:length(realsub.prop_seq)
        println(i)
        if realsub.prop_seq[i] != cache_prop
            cache_index += 1
            cache_prop = realsub.prop_seq[i]
        end
        prop_seq_changed[i] = cache_index
    end
    ## Fit the hyperparameters
    if model_type == :_1a
        ho = @hyperopt for i in looptime,
                    α in [0.001:0.001:0.999;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub, dodecay=false)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d
        ho = @hyperopt for i in looptime,
                        α in [0.001:0.001:0.999;],
                        decay in [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α, α, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                            RT=realsub.RT,
                                            miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d1e
        ho = @hyperopt for i in looptime,
                           α in [0.001:0.001:0.999;],
                           α_error in [0.001:0.001:1;],
                           decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α, α, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d1e1CCC
        ho = @hyperopt for i in looptime,
                        α in [0.001:0.001:1;],
                        α_error in [0.001:0.001:1;],
                        α_CCC in [0.001:0.001:1;],
                        CCC in [-0.001:-0.001:-1;],
                        decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α, α, α_error, α_error, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_1a1d1CCC
        ho = @hyperopt for i in looptime,
                         α in [0.001:0.001:1;],
                       α_CCC in [0.001:0.001:1;],
                         CCC in [-0.001:-0.001:-1;],
                       decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α, α, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a
        ho = @hyperopt for i in looptime,
                            α_v in [0.001:0.001:1;],
                            α_s in [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, 0)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub, dodecay=false)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d
        ho = @hyperopt for i in looptime,
                         α_v in [0.001:0.001:1;],
                         α_s in [0.001:0.001:1;],
                       decay in [0.001:0.001:1;]

            agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d1e
        ho = @hyperopt for i in looptime,
                        α_v in [0.001:0.001:1;],
                        α_s in [0.001:0.001:1;],
                    α_error in [0.001:0.001:1;],
                      decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_error, α_error, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d1e1CCC
        ho = @hyperopt for i in looptime,
                        α_v in [0.001:0.001:1;],
                        α_s in [0.001:0.001:1;],
                    α_error in [0.001:0.001:1;],
                      α_CCC in [0.001:0.001:1;],
                        CCC in [-0.001:-0.001:-1;],
                      decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_error, α_error, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    elseif model_type == :_2a1d1CCC
        ho = @hyperopt for i in looptime,
                         α_v in [0.001:0.001:1;],
                         α_s in [0.001:0.001:1;],
                       α_CCC in [0.001:0.001:1;],
                         CCC in [-0.001:-0.001:-1;],
                       decay in [0.001:0.001:1;]

            agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(α_v, α_s, α_CCC, α_CCC, CCC, decay)
            model_stim = RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
            validation_dataframe = DataFrame(predicted_var=model_stim[:p_selection_history],
                                                RT=realsub.RT,
                                                miniblock=CategoricalArray(prop_seq_changed, ordered=false))
            validation_formula = @formula(RT ~ predicted_var + miniblock)
            evaluate_relation(validation_dataframe, validation_formula)[:MSE]
        end
    end

    optim_params_value, eval_result = minimum(ho)
    optim_params = Dict(zip(ho.params, optim_params_value))
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[!, :MSE] = ho.results

    return (optim_params, eval_result, verbose_table)
end
