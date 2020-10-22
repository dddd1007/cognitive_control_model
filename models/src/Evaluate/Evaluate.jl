# Evaluate the models' Performences and give the criterion 
# of goodness of Fit

#####
##### Define the Functions
#####

using .Models

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

# 根据最优参数重新拟合模型
function model_recovery(env::ExpEnv, realsub::RealSub, opt_params::Array{Float64,1}; type = :SR)
    if type == :SR
        nargs = length(opt_params)

        if nargs == 3
            agent = RLModels.NoSoftMax.RLLearner_basic(opt_params)
        elseif nargs == 5
            agent = RLModels.NoSoftMax.RLLearner_witherror(opt_params)
        elseif nargs == 8
            agent = RLModels.NoSoftMax.RLLearner_withCCC(opt_params)
        end
        
        return RLModels.NoSoftMax.rl_learning_sr(env, agent, realsub)
    end
end

# 模型选择
function model_evaluation(env, realsub; criteria = :AIC)
    result_list = zeros(4)
    
    optim_param_basic, _, _ = fit_RL_SR(env, realsub, 10000, model_type = :basic)
    p_history_basic = model_recovery(env, realsub, optim_param_basic)[:p_selection_history]
    result_basic = evaluate_relation(p_history_basic, realsub.RT)[criteria]
    result_table[1] = result_basic

    optim_param_error, _, _ = fit_RL_SR(env, realsub, 100000, model_type = :error)
    p_history_error = model_recovery(env, realsub, optim_param_error)[:p_selection_history]
    result_error = evaluate_relation(p_history_error, realsub.RT)[criteria]
    result_table[2] = result_error

    optim_param_ccc_same, _, _ = fit_RL_SR(env, realsub, 1000000, model_type = :CCC_same_alpha)
    p_history_ccc_same = model_recovery(env, realsub, optim_param_ccc_same)[:p_selection_history]
    result_ccc_same = evaluate_relation(p_history_ccc_same, realsub.RT)[criteria]
    result_table[3] = result_ccc_same 

    optim_param_ccc_diff, _, _ = fit_RL_SR(env, realsub, 5000000, model_type = :CCC_different_alpha)
    p_history_ccc_diff = model_recovery(env, realsub, optim_param_ccc_diff)[:p_selection_history]
    result_ccc_diff = evaluate_relation(p_history_ccc_diff, realsub.RT)[criteria]
    result_table[4] = result_ccc_diff 

    return result_list
end