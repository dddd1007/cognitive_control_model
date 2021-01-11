using JLD2, DataFrames, DataFramesMeta, FileIO, CSV, Models, StatsBase, HypothesisTests

@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2021-01-07-145437final_selection_correct_CCC.jld2"

include("/Users/dddd1007/project2git/cognitive_control_model/scripts/import_all_data.jl")

temp_dataframe_stake = []

function extract_CCC_optim_params(eval_results, subname)
    return eval_results[subname][5][:optim_param]
end

function add_conflict_list(sub_dataframe, env, subinfo, opt_params)
    model_result = model_recovery(env, subinfo, opt_params, model_type = :_1a1d1CCC)
    conflict_list = model_result[:conflict_list]
    weight_history = model_result[:options_weight_history]
    sub_dataframe.if_over_CCC = conflict_list .> opt_params[:CCC]
    weight_history_table = DataFrame(weight_history, [:ll,:lr,:rl,:rr])
    sub_dataframe = hcat(sub_dataframe, weight_history_table)
    return sub_dataframe
end

for subname in collect(keys(eval_results))
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

dataframe_with_CCC_tag

CSV.write("/Users/dddd1007/Downloads/test.csv", dataframe_with_CCC_tag)

# 验证假设
function test_RT_difference(dataframe_with_CCC_tag)
    same_stim_index = []
    for i in 1:(nrow(dataframe_with_CCC_tag)-1)
        cache_row = dataframe_with_CCC_tag[i, :]
        cache_second_row = dataframe_with_CCC_tag[i+1, :]
        if cache_row.if_over_CCC && cache_row.stim_color == cache_second_row.stim_color && cache_row.stim_loc == cache_second_row.stim_loc
            push!(same_stim_index, i)
        end
    end
    same_stim_table_part1 = dataframe_with_CCC_tag[same_stim_index, :]
    same_stim_table_part2 = dataframe_with_CCC_tag[same_stim_index .+ 1, :]
    samestim_RT_list = parse.(Float64, same_stim_table_part1.RT) .- parse.(Float64, same_stim_table_part2.RT)

    same_stim_noCCC_index = []
    for i in 1:(nrow(dataframe_with_CCC_tag)-1)
        cache_row = dataframe_with_CCC_tag[i, :]
        cache_second_row = dataframe_with_CCC_tag[i+1, :]
        if !cache_row.if_over_CCC && cache_row.stim_color == cache_second_row.stim_color && cache_row.stim_loc == cache_second_row.stim_loc
            push!(same_stim_noCCC_index, i)
        end
    end
    same_stim_noCCC_table_part1 = dataframe_with_CCC_tag[same_stim_noCCC_index, :]
    same_stim_noCCC_table_part2 = dataframe_with_CCC_tag[same_stim_noCCC_index .+ 1, :]
    samestim_noCCC_RT_list = parse.(Float64, same_stim_noCCC_table_part1.RT) .- parse.(Float64, same_stim_noCCC_table_part2.RT)

    return UnequalVarianceTTest(samestim_RT_list, samestim_noCCC_RT_list)
end

test_RT_difference(dataframe_with_CCC_tag)