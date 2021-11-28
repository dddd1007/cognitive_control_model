using DataFrames, DataFramesMeta, GLM, StatsBase
using CSV

bl_data = DataFrame(CSV.File("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_sr_ab_bayesian_learner.csv"))

bl_sr_to_softmax = @select(bl_data, :Subject_num, :stim_loc, :Response, :RT, :sr_r_selected, :sr_ll, :sr_lr, :sr_rl, :sr_rr)

function selected_softmax_value(beta, r::Float64)
    exp(beta * r) / (exp(beta * r) + exp(beta * (1-r)))
end

function obj_softmax_beta_aic(beta, r::Vector{Float64}, RT::Vector{Float64}; verbose = false)
    softmaxed_value = zeros(Float64, length(r))
    count_num = 1
    for i in r
        softmaxed_value[count_num] = selected_softmax_value(beta, i)
        count_num = count_num + 1
    end
    ols = lm(reshape(softmaxed_value, length(softmaxed_value), 1), RT)
    aic_result = aic(ols)

    if verbose == false
        return(aic_result)
    else
        return(aic_result, softmaxed_value, ols)
    end
end

# sub_num_list = unique(bl_sr_to_softmax[:Subject_num])

# for sub_num in sub_num_list
    sub_num = 1
    tmp_data = @subset(bl_sr_to_softmax, :Subject_num .== sub_num)

    
# end

beta_band = [-100:0.001:100;]
aic_result = zeros(Float64, length(beta_band))
for i in 1:length(beta_band)
    aic_result[i] = obj_softmax_beta_aic(beta_band[i], tmp_data[!,:sr_r_selected], tmp_data[!,:RT], verbose = true)
end