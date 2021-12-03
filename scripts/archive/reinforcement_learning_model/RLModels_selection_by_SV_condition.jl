using Models, DataFrames, Dates, JLD2, FileIO
import CSV

savepath = joinpath(dirname(pathof(Models)), "..", "..", "data", "output", "RLModels", "model_selection")
include("import_all_data.jl")

struct Model_eval_result
    subname
    single_alpha
    single_alpha_no_decay
    no_decay
    single_alpha_total_decay
    total_decay
    basic
    error
    CCC_same_alpha
    CCC_different_alpha
end

#sub1_data = @where(all_data, :Subject_num .== 1)
#env, realsub = init_env_sub(sub1_data, env_idx_dict, sub_idx_dict)

# 定义模型选择函数
function model_evaluation(env, realsub, number_iterations)
    
    eval_result = []
    subname = realsub.sub_tag[1]

    push!(eval_result, subname)

    println("+++ " * subname * " basic model +++")
    
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:single_alpha, number_iterations=number_iterations))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:single_alpha_no_decay, number_iterations=number_iterations))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:no_decay, number_iterations=number_iterations))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:single_alpha_total_decay, number_iterations=number_iterations))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:total_decay, number_iterations=number_iterations))

    println("+++ " * subname * " complex model +++")
    
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:basic, number_iterations=number_iterations * 10))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:error, number_iterations=number_iterations * 30))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:CCC_same_alpha, number_iterations=number_iterations * 50))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:CCC_different_alpha, number_iterations=number_iterations * 100))

    return Model_eval_result(eval_result...)
end

#####
##### Begin evaluation
#####

# Evaluate V condition

all_data_v = @where(all_data, :condition .== "v")

eval_results_v = Dict()

Threads.@threads for sub_num in 1:36
        
    if sub_num == 27 || sub_num == 6
        continue
    end
    println("========= Begin Sub " * repr(sub_num) * " ==========")
    
    each_sub_data = @where(all_data_v, :Subject_num .== sub_num)
    each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict,
                                                          sub_idx_dict)
    eval_results_v[each_subinfo.sub_tag[1]] = model_evaluation(each_env, each_subinfo, 10000)
end

current_time = Dates.format(now(), "yyyy-mm-dd-HHMMSS")
filename = savepath * "/" * current_time  * "_v.jld2"

@save filename eval_results_v

# Evaluate S condition

all_data_s = @where(all_data, :condition .== "s")

eval_results_s = Dict()

Threads.@threads for sub_num in 1:36
        
    if sub_num == 27 || sub_num == 6
        continue
    end
    println("========= Begin Sub " * repr(sub_num) * " ==========")
    
    each_sub_data = @where(all_data_s, :Subject_num .== sub_num)
    each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict,
                                                          sub_idx_dict)
    eval_results_s[each_subinfo.sub_tag[1]] = model_evaluation(each_env, each_subinfo, 10000)
end

filename = savepath * "/" * current_time  * "_s.jld2"

@save filename eval_results_s