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
        mse_value = deviance(reg_result)
        result = Dict(:β => β_value, :AIC => aic_value, :BIC => bic_value, :R2 => r2_value,
                      :MSE => mse_value)
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
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[1], 0, dodecay=false)
    elseif model_type == :no_decay
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[2], 0, dodecay=false)
    elseif model_type == :single_alpha_total_decay
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[1], 1)
    elseif model_type == :total_decay
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[2], 1)
    elseif model_type == :basic
        agent = RLModels.NoSoftMax.RLLearner_basic(opt_params[1], opt_params[2],
                                                   opt_params[3])
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
    end

    return RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
end

# 快速拟合模型并评估拟合度
function fit_and_evaluate(env, realsub; criteria=:AIC, model_type, number_iterations)
    optim_param_basic, _, _ = fit_RL_SR(env, realsub, number_iterations,
                                        model_type=model_type)
    p_history_basic = model_recovery(env, realsub, optim_param_basic,
                                     model_type=model_type)[:p_selection_history]
    return result_basic = evaluate_relation(p_history_basic, realsub.RT)[criteria]
end

# 模型选择
function model_evaluation(env, realsub; criteria=:AIC)
    result_list = zeros(9)

    subname = realsub.sub_tag[1]
    
    println("+++ " * subname * " basic model +++")
    result_list[1] = fit_and_evaluate(env, realsub, criteria=criteria,
                                      model_type=:single_alpha, number_iterations=3000)
    result_list[2] = fit_and_evaluate(env, realsub, criteria=criteria,
                                      model_type=:single_alpha_no_decay,
                                      number_iterations=5000)
    result_list[3] = fit_and_evaluate(env, realsub, criteria=criteria, model_type=:no_decay,
                                      number_iterations=5000)
    result_list[4] = fit_and_evaluate(env, realsub, criteria=criteria, model_type=:single_alpha_total_decay,
                                      number_iterations=5000)
    result_list[5] = fit_and_evaluate(env, realsub, criteria=criteria, model_type=:total_decay,
                                      number_iterations=5000)

    println("+++ " * subname * " complex model +++")
    result_list[6] = fit_and_evaluate(env, realsub, criteria=criteria, model_type=:basic,
                                      number_iterations=10000)
    result_list[7] = fit_and_evaluate(env, realsub, criteria=criteria, model_type=:error,
                                      number_iterations=50000)
    result_list[8] = fit_and_evaluate(env, realsub, criteria=criteria,
                                      model_type=:CCC_same_alpha, number_iterations=100000)
    result_list[9] = fit_and_evaluate(env, realsub, criteria=criteria,
                                      model_type=:CCC_different_alpha, number_iterations=300000)

    return result_list
end