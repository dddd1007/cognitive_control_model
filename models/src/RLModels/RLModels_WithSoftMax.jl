#============================================================================ 
# Module1: RLModels with Softmax                                            #
============================================================================#
module WithSoftMax

using ..RLModels

#### Define the Class System
"""
    RLLearner_basic

A RLLearner which learnt parameters from the experiment environment.
"""
#####
##### 定义类型系统
#####

# 环境中的学习者, 在基本条件下
struct RLLearner_basic <: RLLearner
    α_v::Float64
    β_v::Float64
    α_s::Float64
    β_s::Float64
    decay::Any
end

# 环境中的学习者, 在错误试次下学习率不同
struct RLLearner_witherror <: RLLearner
    α_v::Float64
    β_v::Float64
    α_s::Float64
    β_s::Float64

    α_v_error::Float64
    β_v_error::Float64
    α_s_error::Float64
    β_s_error::Float64

    decay::Any
end

# 存在冲突控制的学习者
struct RLLearner_withCCC <: RLLearner
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
    decay::Any
end

#### Define the data update functions

# 定义SR学习中的决策过程
function sr_softmax(options_vector::Array{Float64,1}, β, true_selection::Tuple, debug=false)
    options_matrix = reshape(options_vector, 2, 2)'

    op_selection_idx = CartesianIndex(true_selection[1], abs(true_selection[2] - 1)) +
                       CartesianIndex(1, 1)
    true_selection_idx = CartesianIndex(true_selection) + CartesianIndex(1, 1)

    if debug
        println(options_matrix)
        println("True selection is " * repr(options_matrix[true_selection_idx]))
        println("Op selection is " * repr(options_matrix[op_selection_idx]))
    end

    return exp(β * options_matrix[true_selection_idx]) /
           (exp(β * options_matrix[true_selection_idx]) +
            exp(β * options_matrix[op_selection_idx]))
end

# 定义参数选择过程的函数
function get_action_para(env::ExpEnv, agent::RLLearner_basic, realsub::RealSub, idx::Int)
    if env.env_type[idx] == "v"
        β = agent.β_v
        α = agent.α_v
    elseif env.env_type[idx] == "s"
        β = agent.β_s
        α = agent.α_s
    end

    return (α, β)
end

function get_action_para(env::ExpEnv, agent::RLLearner_witherror, realsub::RealSub, idx::Int)
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

    return (α, β)
end

function get_action_para(env::ExpEnv, agent::RLLearner_withCCC, realsub::RealSub, idx::Int,
                         conflict)
    if env.env_type[idx] == "v"
        if realsub.corrections[idx] == 1 && conflict ≤ agent.CCC
            β = agent.β_v
            α = agent.α_v
        elseif realsub.corrections[idx] == 1 && conflict > agent.CCC
            β = agent.β_v_CCC
            α = agent.α_v_CCC
        elseif realsub.corrections[idx] == 0
            β = agent.β_v_error
            α = agent.α_v_error
        end
    elseif env.env_type[idx] == "s"
        if realsub.corrections[idx] == 1 && conflict ≤ agent.CCC
            β = agent.β_s
            α = agent.α_s
        elseif realsub.corrections[idx] == 1 && conflict > agent.CCC
            β = agent.β_s_CCC
            α = agent.α_s_CCC
        elseif realsub.corrections[idx] == 0
            β = agent.β_s_error
            α = agent.α_s_error
        end
    end

    return (α, β)
end

##### 定义强化学习相关函数

# 学习具体SR联结的强化学习过程
function rl_learning_sr(env::ExpEnv, agent::RLLearner, realsub::RealSub)

    # Check the subtag
    if env.sub_tag != realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_softmax_history = init_param(env, :sr)

    # Start learning
    for idx in 1:total_trials_num
        if isa(agent, RLLearner_withCCC)
            conflict = calc_CCC(options_weight_matrix[idx, :],
                                (env.stim_task_unrelated[idx],
                                 env.stim_correct_action[idx]))
            α, β = get_action_para(env, agent, realsub, idx, conflict)
        else
            α, β = get_action_para(env, agent, realsub, idx)
        end

        ## Update 
        options_weight_matrix[idx + 1, :] = update_options_weight_matrix(options_weight_matrix[idx,
                                                                                               :],
                                                                         α, agent.decay,
                                                                         (env.stim_task_unrelated[idx],
                                                                          env.stim_correct_action[idx]))

        ## Decision
        p_softmax_history[idx] = sr_softmax(options_weight_matrix[idx + 1, :], β,
                                            (env.stim_task_unrelated[idx],
                                             env.stim_correct_action[idx]))
    end

    options_weight_result = options_weight_matrix[2:end, :]
    return Dict(:options_weight_history => options_weight_result,
                :p_softmax_history => p_softmax_history)
end

end # RLModels_SoftMax