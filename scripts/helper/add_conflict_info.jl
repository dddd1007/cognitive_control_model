using Models, DataFrames, DataFramesMeta, CSV

# prepare data
raw_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv", DataFrame)
raw_data = @where(raw_data, :Response .!= "NA")

estimate_parameter = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/optim_model_wang/2a1d1ccc_wang.csv", DataFrame)
include("/Users/dddd1007/project2git/cognitive_control_model/scripts/helper/import_all_data.jl")
Models.DataManipulate.transform_data!(raw_data, transform_rule)

# for sub_num in 1:16
sub_num = 1
raw_sub_data = @where(raw_data, :Subject_num .== sub_num)
sub_optim_params = @where(estimate_parameter, :sub_num .== sub_num)
sub_env, sub_info = Models.RLModels.init_env_sub(raw_sub_data, env_idx_dict, sub_idx_dict);
optim_params_dict = Dict(:α_v => sub_optim_params.:alpha_v...,
                        :α_s => sub_optim_params.:alpha_s...,
                        :α_CCC => sub_optim_params.:alpha_ccc...,
                        :CCC => sub_optim_params.:ccc...,
                        :decay => sub_optim_params.:decay...)
temp_result = model_recovery(sub_env, sub_info, optim_params_dict, model_type = :_2a1d1CCC)
raw_sub_data.pe = temp_result[:prediction_error]
raw_sub_data.conflict_value = temp_result[:conflict_list]
raw_sub_data[!, :CCC] .= optim_params_dict[:CCC]
sub1_result_table = @transform(raw_sub_data, if_below_CCC = (-(sign.(:conflict_value - :CCC)) .- 1) ./ 2 .+ 1)

for sub_num in 2:16
    if sub_num == 27 || sub_num == 6
        continue
    end
    raw_sub_data = @where(raw_data, :Subject_num .== sub_num)
    sub_optim_params = @where(estimate_parameter, :sub_num .== sub_num)
    println(sub_num)
    sub_env, sub_info = Models.RLModels.init_env_sub(raw_sub_data, env_idx_dict, sub_idx_dict);
    optim_params_dict = Dict(:α_v => sub_optim_params.:alpha_v...,
                            :α_s => sub_optim_params.:alpha_s...,
                            :α_CCC => sub_optim_params.:alpha_ccc...,
                            :CCC => sub_optim_params.:ccc...,
                            :decay => sub_optim_params.:decay...)
    temp_result = model_recovery(sub_env, sub_info, optim_params_dict, model_type = :_2a1d1CCC)
    raw_sub_data.pe = temp_result[:prediction_error]
    raw_sub_data.conflict_value = temp_result[:conflict_list]
    raw_sub_data[!, :CCC] .= optim_params_dict[:CCC]
    result_table = @transform(raw_sub_data, if_below_CCC = (-(sign.(:conflict_value - :CCC)) .- 1) ./ 2 .+ 1)
    append!(sub1_result_table, result_table)
end

CSV.write("/Users/dddd1007/project2git/fmri_analysis2_nipype_type/data/behavioral_data/subdata_with_conflict.csv", sub1_result_table)
