include("/Users/dddd1007/project2git/cognitive_control_model/Models/rl_model_estimate_by_stim/func_estimate_rl_model.jl")
using CSV, DataFramesMeta

# 导入数据

all_sub_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv",
                        DataFrame)
sub_num_list = unique(all_sub_data[!, "Subject_num"]);
ab_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_param_set.csv")
ab_v_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_volatility_param_set.csv")
sr_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_param_set.csv")
sr_v_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_volatility_param_set.csv")

# model recovery
ab_p_value = []
ab_v_p_value = []
sr_p_value = []
sr_v_p_value = []

ab_pe_value = []
ab_v_pe_value = []
sr_pe_value = []
sr_v_pe_value = []

for i in sub_num_list
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])

    stim_feature_seq   = single_sub_data[!, :congruency_num]
    exp_volatility_seq = single_sub_data[!, :volatility_num]
    stim_loc_seq       = single_sub_data[!, :stim_loc_num]
    reaction_loc_seq   = single_sub_data[!, :correct_action]

    # ab model
    ab_params = @subset(ab_optim_params, :sub == sub_num_list[i])
    model_result = ab_model(ab_params.α, stim_feature_seq)
    push!(ab_p_value, model_result["Predicted sequence"])
    push!(ab_pe_value, model_result["Prediciton error"])

    # ab_v model
    ab_v_params = @subset(ab_v_optim_params, :sub == sub_num_list[i])
    model_result = ab_volatility_model(ab_v_params.α_s, ab_v_params.α_v, stim_feature_seq, exp_volatility_seq)
    push!(ab_v_p_value, model_result["Predicted sequence"])
    push!(ab_v_pe_value, model_result["Prediciton error"])
    
    # sr model
    sr_model = @subset
