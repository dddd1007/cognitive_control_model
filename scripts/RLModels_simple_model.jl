using Models, DataFrames, Dates
import CSV

savepath = joinpath(dirname(pathof(Models)), "..", "..", "data", "output", "RLModels", "model_selection")
include("import_all_data.jl")

struct Model_eval_result
    CCC_same_alpha_no_error
    CCC_different_alpha_no_error
end
#sub1_data = @where(all_data, :Subject_num .== 1)
#env, realsub = init_env_sub(sub1_data, env_idx_dict, sub_idx_dict)

# 定义模型选择函数
function model_evaluation(env, realsub, number_iterations)
    
    eval_result = []
    subname = realsub.sub_tag[1]

    push!(eval_result, subname)

    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:CCC_same_alpha_no_error, number_iterations=number_iterations * 30))
    push!(eval_result, fit_and_evaluate(env, realsub, model_type=:CCC_different_alpha_no_error, number_iterations=number_iterations * 50))

    return eval_result
end

#####
##### Begin evaluation
#####

eval_results = Dict()

sub_num = 1
println("========= Begin Sub " * repr(sub_num) * " ==========")

each_sub_data = @where(all_data, :Subject_num .== sub_num)
each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict, sub_idx_dict)
eval_results[each_subinfo.sub_tag[1]] = model_evaluation(each_env, each_subinfo, 1)


current_time = Dates.format(now(), "yyyy-mm-dd-HHMMSS")
filename = savepath * "/" * current_time  * "_sub1_simple_models.jld2"

using JLD2, FileIO

@save filename eval_results