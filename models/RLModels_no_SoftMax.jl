
#============================================================================ 
# Module2: RLModels without Softmax                                         #
============================================================================#
module RLModels_no_Softmax

using DataFrames, RLModels

#### Define the Class system

# 环境中的学习者, 基本条件下
struct Learner_basic
    α_v::Float64
    α_s::Float64
    decay::Float64
end

# 环境中的学习者, 在错误试次下学习率不同
struct Learner_witherror
    α_v::Float64
    α_s::Float64

    α_v_error::Float64
    α_s_error::Float64

    decay::Float64
end

# 存在冲突控制的学习者
struct Learner_withCCC
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




end