module Fit 

using ..RLModels, Hyperopt, RecursiveArrayTools, StatsBase, DataFrames

#####
##### 强化学习模型的模型拟合
#####

# 定义评估变量关系的函数 
function evaluate_relation(x, y, method=:regression)
    if method == :mse
        return (sum(abs2.(x .- y)))/length(x)
    elseif method == :cor
        return cor(x, y)
    elseif method == :regression
        data = DataFrame(x=x, y=y)
        reg_result = lm(@formula(y ~ x), data)
        β_value = coef(reg_result)[2]
        aic_value = aic(reg_result)
        bic_value = bic(reg_result)
        r2_value = r2(reg_result)
        mse_value = deviance(reg_result)
        result = Dict(:β => β_value, :AIC => aic_value, :BIC => bic_value, :R2 => r2_value,
                      :MSE => mse_value)
        return result
    end
end
### 估计SR模型
function fit_RL_SR_Model(env, realsub, looptime; model_type)
    
    ## Fit the Hyperparameters
    if model_type == :basic
        ho = @hyperopt for i = looptime,
                            α_v = [0.01:0.01:1;],
                            α_s = [0.01:0.01:1;],
                            decay = [0.01:0.01:1;]

            agent =  RLModels.WithSoftMax.RLLearner_basic(α_v, α_s, decay)
            model_stim = RLModels.WithSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :error
        ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = α_s_error = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

            agent = RLModels.WithSoftMax.RLLearner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
            model_stim = RLModels.WithSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
    elseif model_type == :CCC 
        ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = α_s_error = [0.01:0.01:1;],
                        α_v_CCC = α_s_CCC = [0.01:0.01:1;],
                        CCC = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

            agent = RLModels.WithSoftMax.RLLearner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
            model_stim = RLModels.WithSoftMax.rl_learning_sr(env, agent, realsub)
            evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
        end
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

        agent =  RLModels.NoSoftMax.RLLearner_basic(α_v, α_s, decay)
        model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
        evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
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

        agent = RLModels.NoSoftMax.RLLearner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
        model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
        evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
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

        agent = RLModels.NoSoftMax.RLLearner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
        model_stim = RLModels.NoSoftMax.rl_learning_ab(env, agent, realsub)
        evaluate_relation(model_stim[:PE], realsub.RT)[:MSE]
    end

    optim_param, eval_result = minimum(ho)
    verbose_table = DataFrame(VectorOfArray(ho.history)', collect(ho.params))
    verbose_table[:MSE] = ho.results

    return (optim_param, eval_result, verbose_table)
end

end # module