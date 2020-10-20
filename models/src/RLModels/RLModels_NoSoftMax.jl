
#= =========================================================================== 
# Module2: RLModels without selection                                        #
=========================================================================== =#
module NoSoftMax

using ..RLModels

#####
##### Define the Class System
#####

"""
    RLLearner_basic

A RLLearner which learnt parameters from the experiment environment.
"""

# 环境中的学习者, 在基本条件下
struct RLLearner_basic <: RLLearner
    α_v::Float64
    α_s::Float64
    decay::Float64
end

# 环境中的学习者, 在错误试次下学习率不同
struct RLLearner_witherror <: RLLearner
    α_v::Float64
    α_s::Float64

    α_v_error::Float64
    α_s_error::Float64

    decay::Float64
end

# 存在冲突控制的学习者
struct RLLearner_withCCC <: RLLearner
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
    debug=false,
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
    debug=false,)
    true_selection_idx = true_selection + 1
    
    if debug
        println(true_selection_idx)
    end 

    return options_vector[true_selection_idx] 
end

function get_action_para(env::ExpEnv, agent::RLLearner_basic, realsub::RealSub, idx::Int)
    if env.env_type[idx] == "v"
        α = agent.α_v
    elseif env.env_type[idx] == "s"
        α = agent.α_s
    end

    return(α)
end

function get_action_para(env::ExpEnv, agent::RLLearner_witherror, realsub::RealSub, idx::Int)
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

function get_action_para(env::ExpEnv, agent::RLLearner_withCCC, realsub::RealSub, idx::Int, conflict)

    if env.env_type[idx] == "v" 
        if realsub.corrections[idx] == 1 && conflict ≤ agent.CCC
            α = agent.α_v
        elseif realsub.corrections[idx] == 1 && conflict > agent.CCC
            α = agent.α_v_CCC
        elseif realsub.corrections[idx] == 0
            α = agent.α_v_error
        end
    elseif env.env_type[idx] == "s"
        if realsub.corrections[idx] == 1 && conflict ≤ agent.CCC
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
    agent::RLLearner,
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
        
        if isa(agent, RLLearner_withCCC)
            conflict = calc_CCC(options_weight_matrix[idx,:], (env.stim_task_unrelated[idx], env.stim_correct_action[idx]))
            α = get_action_para(env, agent, realsub, idx, conflict)
        else
            α = get_action_para(env, agent, realsub, idx)
        end
        
        ## Update
        # Please note the first row of the value matrix 
        # represent the preparedness of the subject!
        options_weight_matrix[idx + 1, :] =
            update_options_weight_matrix(
                options_weight_matrix[idx, :],
                α,
                agent.decay,
                (env.stim_task_unrelated[idx], env.stim_correct_action[idx]),
            )

        ## Decision
        p_selection_history[idx] = selection_value(
            options_weight_matrix[idx + 1, :],
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
function rl_learning_ab(env::ExpEnv, agent::RLLearner, realsub::RealSub)

    # Check the subtag
    if env.sub_tag ≠ realsub.sub_tag
        return println("The env and sub_real_data not come from the same one!")
    end

    # init learning parameters list
    total_trials_num, options_weight_matrix, p_selection_history, PE_history = 
        init_param(env, :ab)

    # Start learning
    for idx = 1:total_trials_num

        if isa(agent, RLLearner_withCCC)
            conflict = calc_CCC(options_weight_matrix[idx,:], env.stim_action_congruency[idx])
            α = get_action_para(env, agent, realsub, idx, conflict)
        else
            α = get_action_para(env, agent, realsub, idx)
        end

        ## Update
        options_weight_matrix[idx + 1, :] = update_options_weight_matrix(options_weight_matrix[idx, :], α, env.stim_action_congruency[idx])

        ## Decision
        p_selection_history[idx] = selection_value(options_weight_matrix[idx + 1, :], env.stim_action_congruency[idx])
        
        ## Calc PE
        if env.stim_action_congruency[idx] == 1
            PE_history[idx] = 1 - p_selection_history[idx]
        elseif env.stim_action_congruency[idx] == 0
            PE_history[idx] = p_selection_history[idx]
        end
    
    end

    options_weight_result = options_weight_matrix[2:end, :]
    return Dict(
        :options_weight_history => options_weight_result,
        :p_selection_history => p_selection_history,
        :PE => PE_history
    )
end

end # RLModels_no_SoftMax