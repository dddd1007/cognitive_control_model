include("/Users/dddd1007/project2git/cognitive_control_model/Models/rl_model_estimate_by_stim/func_estimate_rl_model.jl")
using CSV, DataFramesMeta

# 导入数据

all_sub_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv",
                        DataFrame)
sub_num_list = unique(all_sub_data[!, "Subject_num"]);
ab_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_param_set.csv", DataFrame)
ab_v_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_volatility_param_set.csv", DataFrame)
sr_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_param_set.csv", DataFrame)
sr_v_optim_params = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_volatility_param_set.csv", DataFrame)

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
    single_sub_data = @subset(all_sub_data, :Subject_num .== i)

    stim_feature_seq   = single_sub_data[!, :congruency_num]
    exp_volatility_seq = single_sub_data[!, :volatility_num]
    stim_loc_seq       = single_sub_data[!, :stim_loc_num]
    reaction_loc_seq   = single_sub_data[!, :correct_action]

    # ab model
    ab_params = @subset(ab_optim_params, :sub .== i)
    model_result = ab_model(ab_params.α[1], stim_feature_seq)
    ab_p_value = [ab_p_value; model_result["Predicted sequence"]]
    ab_pe_value = [ab_pe_value; model_result["Prediciton error"]]

    # ab_v model
    ab_v_params = @subset(ab_v_optim_params, :sub .== i)
    model_result = ab_volatility_model(ab_v_params.α_s[1], ab_v_params.α_v[1], stim_feature_seq, exp_volatility_seq)
    ab_v_p_value = [ab_v_p_value; model_result["Predicted sequence"]]
    ab_v_pe_value = [ab_v_pe_value; model_result["Prediciton error"]]
    
    # sr model
    sr_params = @subset(sr_optim_params, :sub .== i)
    model_result = sr_sep_alpha_model(sr_params.α_l[1], sr_params.α_r[1], stim_loc_seq, reaction_loc_seq)
    sr_p_value = [sr_p_value; model_result["Predicted sequence"]]
    sr_pe_value = [sr_pe_value; model_result["Prediciton error"]]

    # sr_v model
    sr_v_params = @subset(sr_v_optim_params, :sub .== i)
    model_result = sr_sep_alpha_volatility_model(sr_v_params.α_s_l[1], sr_v_params.α_s_r[1], sr_v_params.α_v_l[1], sr_v_params.α_v_r[1], stim_loc_seq, reaction_loc_seq, exp_volatility_seq)
    sr_v_p_value = [sr_v_p_value; model_result["Predicted sequence"]]
    sr_v_pe_value = [sr_v_pe_value; model_result["Prediciton error"]]
end

insertcols!(all_sub_data, :rl_ab_p => ab_p_value, :rl_ab_pe => ab_pe_value, :rl_ab_v_p => ab_v_p_value, :rl_ab_v_pe => ab_v_pe_value, :rl_sr_p => sr_p_value, :rl_sr_pe => sr_pe_value, :rl_sr_v_p => sr_v_p_value, :rl_sr_v_pe => sr_v_pe_value)
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_rl_model_estimate_by_stim.csv", all_sub_data)