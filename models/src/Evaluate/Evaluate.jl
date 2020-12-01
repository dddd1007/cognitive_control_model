# Evaluate the models' Performences and give the criterion 
# of goodness of Fit

#####
##### Define the Functions
#####

using .Models, DataFrames

# 定义评估变量关系的函数 
function evaluate_relation(x, y, method=:regression)
    if method == :mse
        return (sum(abs2.(x .- y))) / length(x)
    elseif method == :cor
        return cor(x, y)
    elseif method == :regression
        data = DataFrame(x=x, y=y)
        reg_result = lm(@formula(y ~ x), data)
        β_value = coef(reg_result)[2]
        aic_value = aic(reg_result)
        bic_value = bic(reg_result)
        r2_value = r2(reg_result)
        mse_value = deviance(reg_result)/dof_residual(reg_result)
        loglikelihood_value = loglikelihood(reg_result)
        result = Dict(:β => β_value, :AIC => aic_value, :BIC => bic_value, :R2 => r2_value,
                      :MSE => mse_value, :Loglikelihood => loglikelihood_value)
        return result
    end
end

# 根据最优参数重新拟合模型
function model_recovery(env::ExpEnv, realsub::RealSub, opt_params::Array{Float64,1};
                        model_type)
    if model_type == :single_alpha
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[1],
                                                   opt_params[2])
    elseif model_type == :single_alpha_no_decay
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[1], 0)
    elseif model_type == :no_decay
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[2], 0)
    elseif model_type == :single_alpha_total_decay
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[1], 1)
    elseif model_type == :total_decay
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[2], 1)
    elseif model_type == :basic
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[2], opt_params[3])
    elseif model_type == :error
        agent = RLModels.NoSoftMax.RLLearner_witherror(opt_params[1], opt_params[2],
                                                       opt_params[3], opt_params[3],
                                                       opt_params[4])
    elseif model_type == :CCC_same_alpha
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[1], opt_params[2],
                                                     opt_params[3], opt_params[3],
                                                     opt_params[4], opt_params[4],
                                                     opt_params[5], opt_params[6])
    elseif model_type == :CCC_different_alpha
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[1], opt_params[2],
                                                     opt_params[3], opt_params[3],
                                                     opt_params[4], opt_params[5],
                                                     opt_params[6], opt_params[7])
    elseif model_type == :CCC_same_alpha_no_error
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[1], opt_params[2],
                                                     opt_params[3], opt_params[3],
                                                     opt_params[4], opt_params[5])
    elseif model_type == :CCC_different_alpha_no_error
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[1], opt_params[2],
                                                     opt_params[3], opt_params[4],
                                                     opt_params[5], opt_params[6])
    end

    if model_type == :single_alpha_no_decay || model_type == :no_decay
        return RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub, dodecay=false)
    else
        return RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
    end
end

# 快速拟合模型并评估拟合度
function fit_and_evaluate(env, realsub; model_type, number_iterations)
    optim_param, _, _ = fit_RL_SR(env, realsub, number_iterations, model_type=model_type)
    p_history = model_recovery(env, realsub, optim_param, model_type=model_type)[:p_selection_history]
    eval_result = evaluate_relation(p_history, realsub.RT)

    return Dict(:optim_param => optim_param, :p_history => p_history, 
                :eval_result => eval_result, :model_type => model_type)
end