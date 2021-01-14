# 导入数据
using JLD2, DataFrames, DataFramesMeta, FileIO, CSV, Models, StatsBase, HypothesisTests

@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2021-01-07-145437final_selection_correct_CCC.jld2"

include("/Users/dddd1007/project2git/cognitive_control_model/scripts/import_all_data.jl")

# 定义工具函数

## 提取特定被试的估计后参数
function extract_CCC_optim_params(eval_results, subname)
    return eval_results[subname][10][:optim_param]
end

## 添加被试单次trial的冲突程度信息
function add_conflict_list(sub_dataframe, env, subinfo, opt_params)
    model_result = model_recovery(env, subinfo, opt_params, model_type = :_2a1d1CCC)
    conflict_list = model_result[:conflict_list]
    weight_history = model_result[:options_weight_history]
    sub_dataframe.if_below_CCC = conflict_list .< opt_params[:CCC]
    sub_dataframe.conflict = conflict_list

    sub_dataframe[!, :alpha_v] .= opt_params[:α_v]
    sub_dataframe[!, :alpha_s] .= opt_params[:α_s]
    sub_dataframe[!, :alpha_CCC] .= opt_params[:α_CCC]
    sub_dataframe[!, :CCC] .= opt_params[:CCC]
    sub_dataframe[!, :decay] .= opt_params[:decay]

    weight_history_table = DataFrame(weight_history, [:ll,:lr,:rl,:rr])
    sub_dataframe = hcat(sub_dataframe, weight_history_table)
    return sub_dataframe
end

# 给所有被试添加 conflict 程度信息
temp_dataframe_stake = []

for subname in sort(collect(keys(eval_results)))
    each_sub_data = @where(all_data, :Subject .== subname)
    env, subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict, sub_idx_dict)
    opt_params = extract_CCC_optim_params(eval_results, subname)
    each_sub_data = add_conflict_list(each_sub_data, env, subinfo, opt_params) 
    push!(temp_dataframe_stake, each_sub_data)
end

dataframe_with_CCC_tag = temp_dataframe_stake[1]

for i in 2:length(temp_dataframe_stake)
    dataframe_with_CCC_tag = vcat(dataframe_with_CCC_tag, temp_dataframe_stake[i])
end

# 导出数据到R中分析
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/subdata_with_CCC.csv", dataframe_with_CCC_tag)

