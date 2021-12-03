# 理解 sr-learner 的实现
using DataFrames, GLM, StatsBase
using DataFramesMeta
all_sub_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv",
                        DataFrame)
sub_num_list = unique(all_sub_data[!, "Subject_num"]);
single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[1])

α_band = collect(0.01:0.01:1)
for i in collect(1:length(α_band))
    α = α_band[i]
    α_vector = Vector{Float64}(undef, length(α_band))
    AIC_vector = Vector{Float64}(undef, length(α_band))
    AIC_part_vector = Vector{Float64}(undef, length(α_band))
    AIC_left_vector = Vector{Float64}(undef, length(α_band))
    AIC_right_vector = Vector{Float64}(undef, length(α_band))
    # α = 0.4
    # stim_loc_seq     = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
    # reaction_loc_seq = [0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1]

    α_vector[i] = α

    stim_loc_seq     = [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1]
    reaction_loc_seq = [0,0,1,0,1,0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1]

    ## all data into loglikelihood
    lm_data1 = DataFrame(y = reaction_loc_seq, x = estimate_seq)
    fit = glm(@formula(y ~ x), lm_data1, Binomial(), ProbitLink())
    AIC_vector[i] = loglikelihood(fit)

    ## part different stim loc
    lm_data2_l = DataFrame(y = reaction_loc_seq[1:10], x = estimate_seq[1:10])
    lm_data2_r = DataFrame(y = reaction_loc_seq[11:20], x = estimate_seq[11:20])
    fit_l = glm(@formula(y ~ x), lm_data2_l, Binomial(), ProbitLink())
    fit_r = glm(@formula(y ~ x), lm_data2_r, Binomial(), ProbitLink())
    AIC_left_vector[i]  = loglikelihood(fit_l)
    AIC_right_vector[i] = loglikelihood(fit_r)
    AIC_part_vector[i]  = AIC_left_vector[i] + AIC_right_vector[i]
end

## TODO 查看为什么单侧估计会出现 0.01 的情况
# 输入参数
stim_loc_seq = single_sub_data[!, :stim_loc_num]
reaction_loc_seq = single_sub_data[!, :correct_action]
α_l = 0.01
α_r = 0.25
# Init predict code sequence
predict_seq = Vector{Float64}(undef, (length(stim_loc_seq) + 1))
prediction_error_seq = Vector{Float64}(undef, length(stim_loc_seq))
predict_seq_l = [0.5]
predict_seq_r = [0.5]
predict_l_l = 0.5
predict_r_l = 0.5
predict_seq[1] = 0.5

for i in 1:length(stim_loc_seq)
    if stim_loc_seq[i] == 0 # Stim come from left
        PE = reaction_loc_seq[i] - predict_l_l
        predict_l_l = predict_l_l + α_l * PE
        predict_seq[i + 1] = predict_l_l
        push!(predict_seq_l, predict_l_l + α_l * PE)
        prediction_error_seq[i] = PE
    else
        PE = reaction_loc_seq[i] - predict_r_l
        predict_r_l = predict_r_l + α_r * PE
        predict_seq[i + 1] = predict_r_l
        push!(predict_seq_r, predict_r_l + α_r * PE)
        prediction_error_seq[i] = PE
    end
end

## 计算 AIC
stim_loc_left = convert(Vector{Bool}, abs.(stim_loc_seq .- 1))
stim_loc_right = convert(Vector{Bool}, stim_loc_seq)
lm_data_l = DataFrame(reaction_loc_seq[stim_loc_left],
                      predicted_probability=predict_seq_l)
    lm_data_r = DataFrame(reaction_loc_seq[stim_loc_right],
                          predicted_probability=predict_seq_l)

##### tmp
rl_model_result = sr_sep_alpha_model(data.α_l, 
                                         data.α_r,
                                         data.stim_loc_seq,
                                         data.reaction_loc_seq)
    predicted_probability = rl_model_result["Predicted sequence"]

    # calc stim left part
    stim_loc_left = convert(Vector{Bool}, abs.(data.stim_loc_seq .- 1))
    stim_loc_right = convert(Vector{Bool}, data.stim_loc_seq)

    lm_data_l = DataFrame(;
                          stim_feature_seq=data.reaction_loc_seq[stim_loc_left],
                          predicted_probability=rl_model_result["Predicied Left sequence"])
    lm_data_r = DataFrame(;
                          stim_feature_seq=data.reaction_loc_seq[stim_loc_right],
                          predicted_probability=rl_model_result["Predicied Right sequence"])
    fit_l = glm(@formula(stim_feature_seq ~ predicted_probability), lm_data_l, Binomial(),
                ProbitLink())
    fit_r = glm(@formula(stim_feature_seq ~ predicted_probability), lm_data_r, Binomial(),
                ProbitLink())