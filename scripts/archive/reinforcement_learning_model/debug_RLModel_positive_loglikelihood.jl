using Models, DataFrames, Dates
import CSV

savepath = joinpath(dirname(pathof(Models)), "..", "..", "data", "output", "RLModels", "model_selection")
include("import_all_data.jl")

#===============================
!!!modifiy the RT into logRT

Please note the parameters fitted in the scripts come from the log(RT)
===============================#
all_data.logRT = string.(log.(parse.(Float64,all_data.RT)))
select!(all_data, Not(:RT))
rename!(all_data, Dict(:logRT => :RT))

struct Model_eval_result
    subname
    single_alpha
    single_alpha_no_decay
    no_decay
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

    return Model_eval_result(eval_result...)
end

#####
##### Begin evaluation
#####

eval_results = Dict()

Threads.@threads for sub_num in 1:36
        
    if sub_num == 27 || sub_num == 6
        continue
    end
    println("========= Begin Sub " * repr(sub_num) * " ==========")
    
    each_sub_data = @where(all_data, :Subject_num .== sub_num)
    each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict,
                                                          sub_idx_dict)
    eval_results[each_subinfo.sub_tag[1]] = model_evaluation(each_env, each_subinfo, 10000)
end

current_time = Dates.format(now(), "yyyy-mm-dd-HHMMSS")
filename = savepath * "/" * current_time  * "_debug_positive_loglikelihood.jld2"

using JLD2, FileIO

@save filename Model_eval_result eval_results