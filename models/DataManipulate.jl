#=
Data importer for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

导入研究数据并初始化学习环境
=#

#============================================================================ 
# Global: Define RLModels with Softmax                                      #
============================================================================#
module DataManipulate

using DataFrames, DataFramesMeta, GLM
import CSV
export ExpEnv, RealSub
export evaluate_relation, init_env_sub, update_options_weight_matrix
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

#### 定义数据导入的函数
"""
    transform_data!(raw_data, transform_rule)

Convert the string tag into binary format.

# Examples
```julia
# Define the trasnform rule
begin
	code_rule = Dict("red" => "0" , "green" => "1")
	contigency = Dict("con" => "1", "inc" => "0")
	Type_rule = Dict("hit" => "1", "incorrect" => "0")
	location = Dict("left" => "0", "right" => "1")
	transform_rule = Dict("color" => code_rule, "Type" => Type_rule, 
		            "location" => location, "contigency" => contigency)
end
# Excute the transform
transform_data!(experiment_data, transform_rule)
```
"""
function transform_data!(raw_data::DataFrame, transform_rule::Dict)
    for rules in transform_rule
        colname = rules.first
        replace_rules = rules.second
        for replace_pair in replace_rules
            replace!(raw_data[!, colname], replace_pair)
        end

        if !isa(raw_data[!, colname], Array{Int64})
            raw_data[!, colname] = parse.(Int, raw_data[!, colname])
        end
    end
end

"""
    transform_sub!(transformed_data::DataFrame, env_idx_dict::Dict, 
		                  sub_idx_dict::Dict)

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
        β = coef(reg_result)[2]
        AIC = aic(reg_result)
        BIC = bic(reg_result)
        R2 = (reg_result)
        result = Dict(:β => β, :AIC => AIC, :BIC => BIC, :R2 => r2)
        return result
    end
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

end