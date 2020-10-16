using DataFramesMeta, DataFrames, GLM, StatsBase
using Hyperopt, Plots
import CSV

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

#### 定义初始化计算的函数

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

#### Define the Class System
"""
    Learner_basic

A learner which learnt parameters from the experiment environment.
"""
# 环境中的学习者
abstract type Learner end

# 环境中的学习者, 在基本条件下
struct Learner_basic <: Learner
    α_v::Float64
    α_s::Float64
    decay::Float64
end

# 环境中的学习者, 在错误试次下学习率不同
struct Learner_witherror <: Learner
    α_v::Float64
    α_s::Float64

    α_v_error::Float64
    α_s_error::Float64

    decay
end

# 存在冲突控制的学习者
struct Learner_withCCC <: Learner
    α_v::Float64
    α_s::Float64

    α_v_error::Float64
    α_s_error::Float64

    α_v_CCC::Float64
    α_s_CCC::Float64

    CCC::Float64
    decay::Float64
end

#### Define the Functions

# 定义SR学习中的决策过程
function selection_value(
    options_vector::Array{Float64,1},
    true_selection::Tuple,
    debug = false,
)
    options_matrix = reshape(options_vector, 2, 2)'
    true_selection_idx = CartesianIndex(true_selection) + CartesianIndex(1, 1)
    
    if debug
        println(true_selection_idx)
    end
    
    return options_matrix[true_selection_idx]
end

# 定义抽象概念学习的决策过程
function selection_value(
    options_vector::Array{Float64,1},
    true_selection::Int,
    debug = false,)
    true_selection_idx = true_selection + 1
    
    if debug
        println(true_selection_idx)
    end 

    return options_vector[true_selection_idx] 
end

function get_action_para(env::ExpEnv, agent::Learner_basic, realsub::RealSub, idx::Int)
    if env.env_type[idx] == "v"
        α = agent.α_v
    elseif env.env_type[idx] == "s"
        α = agent.α_s
    end

    return(α)
end

function get_action_para(env::ExpEnv, agent::Learner_witherror, realsub::RealSub, idx::Int)
    if env.env_type[idx] == "v"
        if realsub.corrections[idx] == 1
            α = agent.α_v
        elseif realsub.corrections[idx] == 0
            α = agent.α_v_error
        end
    elseif env.env_type[idx] == "s"
        if realsub.corrections[idx] == 1
            α = agent.α_s
        elseif realsub.corrections[idx] == 0
            α = agent.α_s_error
        end
    end

    return(α)
end

function get_action_para(env::ExpEnv, agent::Learner_withCCC, realsub::RealSub, idx::Int, conflict)

    if env.env_type[idx] == "v" 
        if realsub.corrections[idx] == 1 && conflict < agent.CCC
            α = agent.α_v
        elseif realsub.corrections[idx] == 1 && conflict > agent.CCC
            α = agent.α_v_CCC
        elseif realsub.corrections[idx] == 0
            α = agent.α_v_error
        end
    elseif env.env_type[idx] == "s"
        if realsub.corrections[idx] == 1 && conflict < agent.CCC
            α = agent.α_s
        elseif realsub.corrections[idx] == 1 && conflict > agent.CCC
            α = agent.α_s_CCC
        elseif realsub.corrections[idx] == 0
            α = agent.α_s_error
        end
    end
    
    return(α)

end

##### 定义强化学习相关函数

# 学习具体SR联结的强化学习过程
function rl_learning_sr(
    env::ExpEnv,
    agent::Learner,
    realsub::RealSub
)

    # Check the subtag
    if env.sub_tag ≠ realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_selection_history =
        init_param(env, :sr)

    # Start learning
    for idx = 1:total_trials_num
        
        if isa(agent, Learner_withCCC)
            conflict = calc_CCC(options_weight_matrix[idx,:], (env.stim_task_unrelated[idx], env.stim_correct_action[idx]))
            α = get_action_para(env, agent, realsub, idx, conflict)
        else
            α = get_action_para(env, agent, realsub, idx)
        end
        
        ## Update
        # Please note the first row of the value matrix 
        # represent the preparedness of the subject!
        options_weight_matrix[idx+1, :] =
            update_options_weight_matrix(
                options_weight_matrix[idx, :],
                α,
                agent.decay,
                (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
            )

        ## Decision
        p_selection_history[idx] = selection_value(
            options_weight_matrix[idx+1, :],
            (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
        )
    end

    options_weight_result = options_weight_matrix[2:end, :]
    return Dict(
        :options_weight_history => options_weight_result,
        :p_selection_history => p_selection_history,
    )
end

# 学习抽象的 con/inc 概念的强化学习过程
function rl_learning_ab(env::ExpEnv, agent::Learner, realsub::RealSub)

    # Check the subtag
    if env.sub_tag ≠ realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_selection_history = 
        init_param(env, :ab)

    # Start learning
    for idx = 1:total_trials_num

        if isa(agent, Learner_withCCC)
            conflict = calc_CCC(options_weight_matrix[idx,:], env.stim_action_congruency[idx])
            α = get_action_para(env, agent, realsub, idx, conflict)
        else
            α = get_action_para(env, agent, realsub, idx)
        end

        ## Update
        options_weight_matrix[idx+1, :] = update_options_weight_matrix(options_weight_matrix[idx, :], α, env.stim_action_congruency[idx])

        ## Decision
        p_selection_history[idx] = selection_value(options_weight_matrix[idx+1, :], env.stim_action_congruency[idx])
    
    end

    options_weight_result = options_weight_matrix[2:end, :]
    return Dict(
        :options_weight_history => options_weight_result,
        :p_selection_history => p_selection_history,
    )
end

function hyperopt_rllearn_basic(env, realsub, looptime)
    
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        decay = [0.01:0.01:1;]

        agent = Learner_basic(α_v, α_s, decay)
        model_stim = rl_learning_sr(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end
    
    vis = plot(ho)
    optim_param, eval_result = minimum(ho)
    return (optim_param, eval_result, vis)
end

function hyperopt_rllearn_witherror(env, realsub, looptime)
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = Learner_witherror(α_v, α_s, α_v_error, α_s_error, decay)
        model_stim = rl_learning_sr(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    vis = plot(ho)
    optim_param, eval_result = minimum(ho)
    return (optim_param, eval_result, vis)
end

function hyperopt_rllearn_withCCC(env, realsub, looptime)
    ho = @hyperopt for i = looptime,
                        α_v = [0.01:0.01:1;],
                        α_s = [0.01:0.01:1;],
                        α_v_error = [0.01:0.01:1;],
                        α_s_error = [0.01:0.01:1;],
                        α_s_CCC = [0.01:0.01:1;], 
                        α_v_CCC = [0.01:0.01:1;],
                        CCC = [0.01:0.01:1;], 
                        decay = [0.01:0.01:1;]

        agent = Learner_withCCC(α_v, α_s, α_v_error, α_s_error, α_v_CCC, α_s_CCC, CCC, decay)
        model_stim = rl_learning_sr(env, agent, realsub)
        evaluate_relation(model_stim[:p_selection_history], realsub.RT)[:MSE]
    end

    vis = plot(ho)
    optim_param, eval_result = minimum(ho)
    return (optim_param, eval_result, vis)
end

# set the generated file locations
imgpath = "/Data3/Xiaxk/research_data/cognitive_control_model/data/output/RLModels/img_rl_model_hyperopt/" 
csvpath = "/Data3/Xiaxk/research_data/cognitive_control_model/data/output/RLModels/"

import CSV
using DataFrames, DataFramesMeta, Plots

# import all data
all_data = CSV.read("/Data3/Xiaxk/research_data/cognitive_control_model/data/input/pure_all_data.csv");
all_data = @where(all_data, :Response .!= "NA")

# Remove the subject who always move the head
sub27 = @where(all_data, :Subject_num .== 27)

begin
    color_rule = Dict("red" => "0" , "green" => "1")
    congruency_rule = Dict("con" => "1", "inc" => "0")
    Type_rule = Dict("hit" => "1", "incorrect" => "0", "miss" => "0")
    loc_rule = Dict("left" => "0", "right" => "1")
    transform_rule = Dict("stim_color" => color_rule, "Type" => Type_rule, 
        "stim_loc" => loc_rule, "congruency" => congruency_rule)
end
transform_data!(all_data, transform_rule)
begin
    env_idx_dict = Dict("stim_task_related" => "stim_color", 
                        "stim_task_unrelated" => "stim_loc", 
                        "stim_action_congruency" => "congruency", 
                        "correct_action" => "correct_action",
                        "env_type" => "condition", "sub_tag" => "Subject")
    sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
                        "corrections" => "Type", "sub_tag" => "Subject")
end

params_basic = zeros(36, 4)
params_error = zeros(36, 6)
params_CCC = zeros(36, 9)

# For analysis each subject
Threads.@threads for sub_num in 1:36
    println(sub_num)

    if sub_num == 27
        continue
    end

    each_sub_data = @where(all_data, :Subject_num .== sub_num);
    each_env, each_subinfo = init_env_sub(each_sub_data, env_idx_dict, sub_idx_dict);
    
    # basic model
    optim_param, eval_result, vis = hyperopt_rllearn_basic(each_env, each_subinfo, 100000)

    params_basic[sub_num, 1:3] .= optim_param
    params_basic[sub_num, 4]   = eval_result
    
    savefig(vis, imgpath * "basic/" * each_env.sub_tag[1] * ".png")

    # error model
    optim_param, eval_result, vis = hyperopt_rllearn_witherror(each_env, each_subinfo, 1000000)

    params_error[sub_num, 1:5] .= optim_param
    params_error[sub_num, 6]   = eval_result
    
    savefig(vis,imgpath * "error/" * each_env.sub_tag[1] * ".png")

    # CCC model
    optim_param, eval_result, vis = hyperopt_rllearn_withCCC(each_env, each_subinfo, 10000000)

    params_CCC[sub_num, 1:8] .= optim_param
    params_CCC[sub_num, 9]   = eval_result
    
    savefig(vis,imgpath * "CCC/" * each_env.sub_tag[1] * ".png")
end

params_basic_table = DataFrame(params_basic, [:α_v, :α_s, :decay, :MSE])
params_error_table = DataFrame(params_error, [:α_v, :α_s, :α_v_error, :α_s_error, :decay, :MSE])
params_CCC_table   = DataFrame(params_CCC,   [:α_v, :α_s, :α_v_error, :α_s_error, :α_v_CCC, :α_s_CCC, :CCC, :decay, :MSE])

CSV.write(csvpath * "params_basic.csv", params_basic_table)
CSV.write(csvpath * "params_error.csv", params_error_table)
CSV.write(csvpath * "params_CCC.csv",   params_CCC_table)
