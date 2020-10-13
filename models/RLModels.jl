#=
Reinforcement Learning Models for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

This model try to using reinforcement learning and Softmax to simulate the learning process in
Simon Tasks.

There are two models with four different methods:

## Two models

1. Reinforcement learning are used to learning the value of each rules
2. Softmax are used to select action by the value which RL learnt

## Four methods

- What to learn
    - Abstract concepts (Con, Inc)
    - S-R association
- The information attenuation
    - Have a Decay
    - Without Decay
- How to treat the error trials
    - Have a CCC to change action
    - Without CCC
- How to react to error trials
    - Change RL model's learning rate
    - Change Softmax model's Q-value
    - Both
=#

#=
Import nessary library
=#
using GLM, StatsBase, DataFrames

#= 
Init Class system
=#

"""
    Learner_basic

A learner which learnt parameters from the experiment environment.
"""
# 环境中的学习者, 基本条件下
struct Learner_basic
    α_v::Float64
    β_v::Float64
    α_s::Float64
    β_s::Float64
    decay::Float64
end

# 环境中的学习者, 在错误试次下学习率不同
struct Learner_witherror
    α_v::Float64
    β_v::Float64
    α_s::Float64
    β_s::Float64
    α_v_error::Float64
    β_v_error::Float64
    α_s_error::Float64
    β_s_error::Float64
    decay::Float64
end

struct Learner_withCCC
    α_v::Float64
    β_v::Float64
    α_s::Float64
    β_s::Float64

    α_v_error::Float64
    β_v_error::Float64
    α_s_error::Float64
    β_s_error::Float64

    α_v_CCC::Float64
    β_v_CCC::Float64
    α_s_CCC::Float64
    β_s_CCC::Float64

    CCC::Float64
    decay::Float64
end

#=
Define the functions
=#

# 定义SR学习中的决策过程
function sr_softmax(
    options_vector::Array{Float64,1},
    β::Float64,
    true_selection::Tuple,
    debug = false,
)
    options_matrix = reshape(options_vector, 2, 2)'

    op_selection_idx =
        CartesianIndex(true_selection[1], abs(true_selection[2] - 1)) + CartesianIndex(1, 1)
    true_selection_idx = CartesianIndex(true_selection) + CartesianIndex(1, 1)

    if debug
        println(options_matrix)
        println("True selection is " * repr(options_matrix[true_selection_idx]))
        println("Op selection is " * repr(options_matrix[op_selection_idx]))
    end

    exp(β * options_matrix[true_selection_idx]) / (
        exp(β * options_matrix[true_selection_idx]) +
        exp(β * options_matrix[op_selection_idx])
    )
end

# 定义更新价值矩阵的函数
function update_options_weight_matrix(
    weight_vector::Array{Float64,1},
    α::Float64,
    decay::Float64,
    correct_selection::Tuple;
    dodecay = true,
    debug = false,
)
    weight_matrix = reshape(weight_vector, 2, 2)'
    correct_selection_idx = CartesianIndex(correct_selection) + CartesianIndex(1, 1)
    op_selection_idx = abs(correct_selection[1] - 1) + 1

    if debug
        println("True selection is " * repr(correct_selection_idx))
        println("The value is " * repr(weight_matrix[correct_selection_idx]))
    end

    weight_matrix[correct_selection_idx] =
        weight_matrix[correct_selection_idx] +
        α * (1 - weight_matrix[correct_selection_idx])

    if dodecay
        weight_matrix[op_selection_idx, :] =
            weight_matrix[op_selection_idx, :] .+
            decay .* (0.5 .- weight_matrix[op_selection_idx, :])
    end

    return reshape(weight_matrix', 1, 4)
end

# 定义计算冲突程度的函数
function calc_CCC(weight_vector::Array{Float64,1}, correct_selection::Tuple)
    weight_matrix = reshape(weight_vector, 2, 2)'

    correct_selection_idx = CartesianIndex(correct_selection) + CartesianIndex(1, 1)
    op_selection_idx =
        CartesianIndex(correct_selection_idx[1], (abs(correct_selection[2] - 1) + 1))

    CCC = weight_matrix[correct_selection_idx] - weight_matrix[op_selection_idx]
end

# 定义评估变量相关性的函数
function evaluate_relation(x, y, method = :regression)
    if method == :mse
        return sum(abs2.(x .- y))
    elseif method == :cor
        return cor(x, y)
    elseif method == :regression
        data = DataFrame(x = x, y = y);
        reg_result = lm(@formula(y~x), data)
        β = coef(reg_result)[2]
        AIC = aic(reg_result)
        BIC = bic(reg_result)
        R2 = (reg_result)
        result = Dict(:β => β, :AIC => AIC, :BIC => BIC, :R2 => r2)
        return result
    end
end

# 初始化更新价值矩阵和基本参数
function init_param(env, agent, learn_type = :sr)

    total_trials_num = length(env.stim_task_unrelated)

    if learn_type == :sr
        options_weight_matrix = zeros(Float64, (total_trials_num + 1, 4))
        options_weight_matrix[1, :] = [0.5, 0.5, 0.5, 0.5]
    elseif learn_type == :ab
        options_weight_matrix = zeros(Float64, (total_trials_num + 1, 2))
        options_weight_matrix[1, :] = [0.5, 0.5]
    end

    p_softmax_history = zeros(Float64, total_trials_num)
    α = 0.0
    β = 0.0
    decay = agent.decay

    return (total_trials_num, options_weight_matrix, p_softmax_history, α, β, decay)
end

# 定义强化学习相关函数

# 学习抽象概念的强化学习过程
function rl_learning_sr(
    env::ExpEnv,
    agent::Learner_basic,
    realsub::RealSub;
    eval_method = "mse",
    verbose = false,
)

    # Check the subtag
    if env.sub_tag != realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_softmax_history, α, β, decay =
        init_param(env, agent)

    # Start learning
    for idx = 1:total_trials_num
        if env.env_type[idx] == "v"
            β = agent.β_v
            α = agent.α_v
        elseif env.env_type[idx] == "s"
            β = agent.β_s
            α = agent.α_s
        end

        ## Update 
        options_weight_matrix[idx+1, :] =
            update_options_weight_matrix(
                options_weight_matrix[idx, :],
                α,
                decay,
                (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
            )

        ## Decision
        p_softmax_history[idx] = sr_softmax(
            options_weight_matrix[idx+1, :],
            β,
            (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
        )
    end

    # Evaluate result
    eval_result = evaluate_relation(realsub.RT, p_softmax_history, eval_method)

    return Dict(
        "options_weight_matrix" => options_weight_matrix,
        "p_softmax_history" => p_softmax_history,
    )
end

function rl_learning_sr(
    env::ExpEnv,
    agent::Learner_witherror,
    realsub::RealSub;
    eval_method = "mse",
    verbose = false,
)

    # Check the subtag
    if env.sub_tag != realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_softmax_history, α, β, decay =
        init_param(env, agent)

    # Start learning
    for idx = 1:total_trials_num
        if env.env_type[idx] == "v"
            if realsub.corrections[idx] == 1
                β = agent.β_v
                α = agent.α_v
            elseif realsub.corrections[idx] == 0
                β = agent.β_v_error
                α = agent.α_v_error
            end
        elseif env.env_type[idx] == "s"
            if realsub.corrections[idx] == 1
                β = agent.β_s
                α = agent.α_s
            elseif realsub.corrections[idx] == 0
                β = agent.β_s_error
                α = agent.α_s_error
            end
        end

        ## Update 
        options_weight_matrix[idx+1, :] =
            update_options_weight_matrix(
                options_weight_matrix[idx, :],
                α,
                decay,
                (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
            )

        ## Decision
        p_softmax_history[idx] = sr_softmax(
            options_weight_matrix[idx+1, :],
            β,
            (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
        )
    end

    # Evaluate result
    eval_result = evaluate_relation(realsub.RT, p_softmax_history, eval_method)

    return Dict(
        "options_weight_matrix" => options_weight_matrix,
        "p_softmax_history" => p_softmax_history,
    )
end

# 学习过程中存在认知控制影响学习的过程
function rl_learning_sr(env::ExpEnv, agent::Learner_withCCC, realsub::RealSub; eval_method = "mse", verbose = false)

    # Check the subtag
    if env.sub_tag != realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_softmax_history, α, β, decay =
    init_param(env, agent)
    conflict = 0.0

    # Start learning
    for idx = 1:total_trials_num
        conflict = calc(options_weight_matrix[idx,:], (env.stim_task_unrelated[idx], env.stim_correct_action[idx]))

        if env.env_type[idx] == "v" 
            if realsub.corrections[idx] == 1 & conflict < agent.CCC
                β = agent.β_v
                α = agent.α_v
            elseif realsub.corrections[idx] == 1 & conflict > agent.CCC
                β = agent.β_v_CCC
                α = agent.α_v_CCC
            elseif realsub.corrections[idx] == 0
                β = agent.β_v_error
                α = agent.α_v_error
            end
        elseif env.env_type[idx] == "s"
            if realsub.corrections[idx] == 1 & conflict < agent.CCC
                β = agent.β_s
                α = agent.α_s
            elseif realsub.corrections[idx] == 1 & conflict > agent.CCC
                β = agent.β_s_CCC
                α = agent.α_s_CCC
            elseif realsub.corrections[idx] == 0
                β = agent.β_s_error
                α = agent.α_s_error
            end
        end

        ## Update 
        options_weight_matrix[idx+1, :] =
            update_options_weight_matrix(
                options_weight_matrix[idx, :],
                α,
                decay,
                (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
            )

        ## Decision
        p_softmax_history[idx] = sr_softmax(
            options_weight_matrix[idx+1, :],
            β,
            (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
        )
    end

    return Dict(
        "options_weight_matrix" => options_weight_matrix,
        "p_softmax_history" => p_softmax_history,
    )
end