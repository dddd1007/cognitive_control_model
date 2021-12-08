using DataFrames, CSV, DataFramesMeta, GLM, StatsBase
include("/Users/dddd1007/project2git/cognitive_control_model/Models/rl_model_estimate_by_stim/func_estimate_rl_model.jl")
all_sub_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv",
                        DataFrame)
sub_num_list = unique(all_sub_data[!, "Subject_num"]);
single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[1])

stim_loc_seq = single_sub_data[!, :stim_loc_num]
reaction_loc_seq = single_sub_data[!, :correct_action]
exp_volatility_seq = single_sub_data[!, :volatility_num]
rl_sr_sep_alpha_volatility_data(0.16,0.14,0.01,0.22,stim_loc_seq, reaction_loc_seq, exp_volatility_seq)
model_data = rl_sr_sep_alpha_volatility_data(0.16,0.14,0.2,0.22,stim_loc_seq, reaction_loc_seq, exp_volatility_seq)
calc_rl_fit_goodness(model_data)