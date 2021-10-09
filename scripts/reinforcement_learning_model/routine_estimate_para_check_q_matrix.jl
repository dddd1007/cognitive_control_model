using Models, DataFrames, Dates, JLD2
import CSV

savepath = joinpath(dirname(pathof(Models)), "..", "..", "data", "output", "RLModels", "model_selection")
include("import_all_data.jl")

# 定义模型选择函数
function model_evaluation(env, realsub, number_iterations)
    
    eval_result = []
    subname = realsub.sub_tag[1]

    push!(eval_result, subname)

    println("+++ " * subname * " 1a model +++")
    
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_1a, number_iterations=number_iterations))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_1a1d, number_iterations=number_iterations * 10))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_1a1d1e, number_iterations=number_iterations * 30))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_1a1d1CCC, number_iterations=number_iterations * 50))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_1a1d1e1CCC, number_iterations=number_iterations * 100))

    println("+++ " * subname * " 2a model +++")
    
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_2a, number_iterations=number_iterations))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_2a1d, number_iterations=number_iterations * 10))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_2a1d1e, number_iterations=number_iterations * 30))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_2a1d1CCC, number_iterations=number_iterations * 50))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:_2a1d1e1CCC, number_iterations=number_iterations * 100))

    return eval_result
end

#####
##### Begin evaluation
#####

eval_results = Dict()

for sub_num in 1:2
        
    if sub_num == 27 || sub_num == 6
        continue
    end
    println("========= Begin Sub " * repr(sub_num) * " ==========")
    
    each_sub_data = @where(all_data, :Subject_num .== sub_num)
    each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict,
                                                          sub_idx_dict)
    eval_results[each_subinfo.sub_tag[1]] = model_evaluation(each_env, each_subinfo, 1)
end

temp_dataframe_stake = []

function extract_CCC_optim_params(eval_results, subname)
    return eval_results[subname][5][:optim_param]
end

function add_conflict_list(sub_dataframe, env, subinfo, opt_params)
    model_result = model_recovery(env, subinfo, opt_params, model_type = :_1a1d1CCC)
    conflict_list = model_result[:conflict_list]
    weight_history = model_result[:options_weight_history]
    sub_dataframe.if_below_CCC = conflict_list .< opt_params[:CCC]
    println(opt_params[:CCC])
    sub_dataframe.conflict = conflict_list

    weight_history_table = DataFrame(weight_history, [:ll,:lr,:rl,:rr])
    sub_dataframe = hcat(sub_dataframe, weight_history_table)
    return sub_dataframe
end

for subname in collect(keys(eval_results))
    subname = collect(keys(eval_results))[1]
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