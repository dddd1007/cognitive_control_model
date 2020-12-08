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
function model_recovery(env::ExpEnv, realsub::RealSub, opt_params;
                        model_type)
                    
    if model_type == :_1a
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α], opt_params[:α], 0)
    elseif model_type == :_1a1d
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α], opt_params[:α],
                                                   opt_params[:decay])
    elseif model_type == :_1a1d1e
        agent = RLModels.NoSoftMax.RLLearner_witherror(opt_params[:α], opt_params[:α],
                                                       opt_params[:α_error], opt_params[:α_error],
                                                       opt_params[:decay])
    elseif model_type == :_1a1d1e1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[:α], opt_params[:α],
                                                     opt_params[:α_error], opt_params[:α_error],
                                                     opt_params[:α_CCC], opt_params[:α_CCC],
                                                     opt_params[:CCC], opt_params[:decay])
    elseif model_type == :_1a1d1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(opt_params[:α], opt_params[:α],
                                                              opt_params[:α_CCC], opt_params[:α_CCC],
                                                              opt_params[:CCC], opt_params[:decay])
    elseif model_type == :_2a
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α_v], opt_params[:α_s], 0)
    elseif model_type == :_2a1d
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[:α_v], opt_params[:α_s], opt_params[:decay])
    elseif model_type == :_2a1d1e
        agent = RLModels.NoSoftMax.RLLearner_witherror(opt_params[:α_v], opt_params[:α_s],
                                                       opt_params[:α_error], opt_params[:α_error],
                                                       opt_params[:decay])
    elseif model_type == :_2a1d1e1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params[:α_v], opt_params[:α_s],
                                                     opt_params[:α_error], opt_params[:α_error],
                                                     opt_params[:α_CCC], opt_params[:α_CCC],
                                                     opt_params[:CCC], opt_params[:decay])
    elseif model_type == :_2a1d1CCC
        agent = RLModels.NoSoftMax.RLLearner_withCCC_no_error(opt_params[:α_v], opt_params[:α_s],
                                                              opt_params[:α_CCC], opt_params[:α_CCC],
                                                              opt_params[:CCC], opt_params[:decay])
    end

    if model_type == :_1a || model_type == :_2a
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