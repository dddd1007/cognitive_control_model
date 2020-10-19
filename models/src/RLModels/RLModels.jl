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

#============================================================================
# Module0: Basic calculate elements of RLModels                             #
============================================================================#
module RLModels

using GLM, DataFrames, StatsBase

export ExpEnv, RealSub
export init_env_sub, evaluate_relation 
export update_options_weight_matrix, init_param
export calc_CCC

#### 定义类型系统

# Init Class system

"""
    ExpEnv

The **experiment environment** which the learner will to learn.
"""
struct ExpEnv
    stim_task_related::Array{Int64,1}
    stim_task_unrelated::Array{Int64,1}
    stim_correct_action::Array{Int64,1}
    stim_action_congruency::Array{Int64,1}
    env_type::Array{String,1}
    sub_tag::Array{String,1}
end

"""
    RealSub

All of the actions the **real subject** have done.
"""
struct RealSub
    response::Array{Int64,1}
    RT::Array{Float64,1}
    corrections::Array{Int64,1}
    sub_tag::Array{String,1}
end

#### Define the functions

"""
init_env_sub(
    transformed_data::DataFrame,
    env_idx_dict::Dict,
    sub_idx_dict::Dict
)

Init the env and subject objects for simulation.

# Examples
```julia
# Define the trasnform rule
begin
    env_idx_dict = Dict("stim_task_related" => "color", "stim_task_unrelated" => "location", 
		                "stim_action_congruency" => "contigency", 
		                "env_type" => "condition", "sub_tag" => "Subject")
	sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
		                "corrections" => "Type", "sub_tag" => "Subject")
end
# Excute the transform
env, sub = init_env_realsub(transformed_data, env_idx_dict, sub_idx_dict, task_rule)
```
"""
function init_env_sub(
    transformed_data::DataFrame,
    env_idx_dict::Dict,
    sub_idx_dict::Dict
)

    exp_env = ExpEnv(
        transformed_data[!, env_idx_dict["stim_task_related"]],
        transformed_data[!, env_idx_dict["stim_task_unrelated"]],
        transformed_data[!, env_idx_dict["correct_action"]],
        transformed_data[!, env_idx_dict["stim_action_congruency"]],
        transformed_data[!, env_idx_dict["env_type"]],
        transformed_data[!, env_idx_dict["sub_tag"]],
    )
    real_sub = RealSub(
        # Because of the miss action, we need the tryparse() 
        # to parse "miss" to "nothing"
        tryparse.(Float64, transformed_data[!, sub_idx_dict["response"]]),
        tryparse.(Float64, transformed_data[!, sub_idx_dict["RT"]]),
        transformed_data[!, sub_idx_dict["corrections"]],
        transformed_data[!, sub_idx_dict["sub_tag"]],
    )
    println(
        "The env and sub info of " *
        transformed_data[!, env_idx_dict["sub_tag"]][1] *
        " is generated!",
    )

    return (exp_env, real_sub)
end

# 初始化更新价值矩阵和基本参数
function init_param(env, learn_type)

    total_trials_num = length(env.stim_task_unrelated)

    if learn_type == :sr
        options_weight_matrix = zeros(Float64, (total_trials_num + 1, 4))
        options_weight_matrix[1, :] = [0.5, 0.5, 0.5, 0.5]
    elseif learn_type == :ab
        options_weight_matrix = zeros(Float64, (total_trials_num + 1, 2))
        options_weight_matrix[1, :] = [0.5, 0.5]
    end

    p_softmax_history = zeros(Float64, total_trials_num)

    return (total_trials_num, options_weight_matrix, p_softmax_history)
end

#### 定义工具性的计算函数

# 定义评估变量相关性的函数
function evaluate_relation(x, y, method = :regression)
    if method == :mse
        return sum(abs2.(x .- y))
    elseif method == :cor
        return cor(x, y)
    elseif method == :regression
        data = DataFrame(x = x, y = y);
        reg_result = lm(@formula(y~x), data)
        β_value = coef(reg_result)[2]
        aic_value = aic(reg_result)
        bic_value = bic(reg_result)
        r2_value = r2(reg_result)
        mse_value = deviance(reg_result)
        result = Dict(:β => β_value, :AIC => aic_value, :BIC => bic_value, :R2 => r2_value, :MSE => mse_value)
        return result
    end
end

# 定义更新价值矩阵的函数

# 具体SR联结学习的价值更新函数
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

# 抽象概念的价值更新函数
function update_options_weight_matrix(
    weight_vector::Array{Float64,1},
    α::Float64,
    correct_selection::Int;
    doreduce = true,
    debug = false,
)
    correct_selection_idx = correct_selection + 1
    op_selection_idx = 2 - correct_selection

    if debug
        println("True selection is " * repr(correct_selection_idx))
        println("The value is " * repr(weight_vector[correct_selection_idx]))
    end

    weight_vector[correct_selection_idx] =
        weight_vector[correct_selection_idx] +
        α * (1 - weight_vector[correct_selection_idx])
    
    if doreduce
        weight_vector[op_selection_idx] = 1 - weight_vector[correct_selection_idx]
    end

    return weight_vector
end

# 定义计算冲突程度的函数
function calc_CCC(weight_vector::Array{Float64,1}, correct_selection::Tuple)
    weight_matrix = reshape(weight_vector, 2, 2)'

    correct_selection_idx = CartesianIndex(correct_selection) + CartesianIndex(1, 1)
    op_selection_idx =
        CartesianIndex(correct_selection_idx[1], (abs(correct_selection[2] - 1) + 1))

    CCC = abs(weight_matrix[correct_selection_idx] - weight_matrix[op_selection_idx])
end

function calc_CCC(weight_vector::Array{Float64,1}, correct_selection::Int)
    correct_selection_idx = correct_selection + 1
    op_selection_idx = 2 - correct_selection

    CCC = abs(weight_vector[correct_selection_idx] - weight_vector[op_selection_idx])
end

include("RLModels_NoSoftMax.jl")
include("RLModels_WithSoftMax.jl")
end # module