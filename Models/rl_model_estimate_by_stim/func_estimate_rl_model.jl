## 拟合估计模型参数
using DataFrames, GLM, StatsBase
include("/Users/dddd1007/project2git/cognitive_control_model/Models/rl_model_estimate_by_stim/rl_models.jl")

# 构建输入数据结构体以实现多重派发

struct rl_ab_data
    α::Float64
    stim_consistency_seq::Vector{Int}
end

struct rl_ab_volatility_data
    α_s::Float64
    α_v::Float64
    stim_consistency_seq::Vector{Int}
    exp_volatility_seq::Vector{Int}
end

struct rl_sr_data
    α::Float64
    stim_loc_seq::Vector{Int}
    reaction_loc_seq::Vector{Int}
end

struct rl_sr_volatility_data
    α_s::Float64
    α_v::Float64
    stim_loc_seq::Vector{Int}
    reaction_loc_seq::Vector{Int}
    exp_volatility_seq::Vector{Int}
end

# 构建计算 loglikelihood 的函数

## helper functions
function calc_fit_idx(model_fitting, fit_idx)
    if fit_idx == "loglikelihood"
        return loglikelihood(model_fitting)
    elseif fit_idx == "aic"
        return aic(model_fitting)
    elseif fit_idx == "bic"
        return bic(model_fitting)
    end
end

## Calc goodness of fit
function calc_rl_fit_goodness(rl_ab_data::rl_ab_data; fit_idx="loglikelihood")
    rl_model_result = ab_model(rl_ab_data.α, rl_ab_data.stim_consistency_seq)
    predicted_probability = rl_model_result["Predicted sequence"]
    lm_data = DataFrame(; stim_feature_seq=rl_ab_data.stim_consistency_seq,
                        predicted_probability=predicted_probability)
    fit = glm(@formula(stim_feature_seq ~ predicted_probability), lm_data, Binomial(),
              ProbitLink())

    fit_goodness = calc_fit_idx(fit, fit_idx)

    return (fit_goodness)
end

function calc_rl_fit_goodness(rl_ab_volatility_data::rl_ab_volatility_data; fit_idx="loglikelihood")
    rl_model_result = ab_model(rl_ab_volatility_data.α_s, rl_ab_volatility_data.α_v,
                               rl_ab_volatility_data.stim_consistency_seq,
                               rl_ab_volatility_data.exp_volatility_seq)
    predicted_probability = rl_model_result["Predicted sequence"]
    lm_data = DataFrame(; stim_feature_seq=rl_ab_volatility_data.stim_consistency_seq,
                        predicted_probability=predicted_probability)
    fit = glm(@formula(stim_feature_seq ~ predicted_probability), lm_data, Binomial(),
              ProbitLink())

    fit_goodness = calc_fit_idx(fit, fit_idx)

    return (fit_goodness)
end

function calc_rl_fit_goodness(rl_sr_data::rl_sr_data; fit_idx="loglikelihood")
    rl_model_result = sr_model(rl_sr_data.α, rl_sr_data.stim_loc_seq,
                               rl_sr_data.reaction_loc_seq)
    predicted_probability = rl_model_result["Predicted sequence"]
    lm_data = DataFrame(; stim_feature_seq=rl_sr_data.stim_loc_seq,
                        predicted_probability=predicted_probability)
    fit = glm(@formula(stim_feature_seq ~ predicted_probability), lm_data, Binomial(),
              ProbitLink())

    fit_goodness = calc_fit_idx(fit, fit_idx)

    return (fit_goodness)
end

function calc_rl_fit_goodness(rl_sr_volatility_data::rl_sr_volatility_data; fit_idx="loglikelihood")
    rl_model_result = sr_model(rl_sr_volatility_data.α_s, rl_sr_volatility_data.α_v,
                               rl_sr_volatility_data.stim_loc_seq,
                               rl_sr_volatility_data.reaction_loc_seq,
                               rl_sr_volatility_data.exp_volatility_seq)
    predicted_probability = rl_model_result["Predicted sequence"]
    lm_data = DataFrame(; stim_feature_seq=rl_sr_volatility_data.stim_loc_seq,
                        predicted_probability=predicted_probability)
    fit = glm(@formula(stim_feature_seq ~ predicted_probability), lm_data, Binomial(),
              ProbitLink())

    fit_goodness = calc_fit_idx(fit, fit_idx)

    return (fit_goodness)
end
