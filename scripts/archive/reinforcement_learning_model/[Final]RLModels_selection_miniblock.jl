using Models, DataFrames, Dates, JLD2
using CSV:CSV

savepath = joinpath(dirname(pathof(Models)), "..", "..", "data", "output", "RLModels",
                    "model_selection")
include("import_all_data.jl")

# 定义模型选择函数
function model_evaluation_miniblock(env, realsub, number_iterations)
    eval_result = []
    subname = realsub.sub_tag[1]

    push!(eval_result, subname)

    println("+++ " * subname * " 1a model +++")

    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_1a,
                                     number_iterations=number_iterations))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_1a1d,
                                     number_iterations=number_iterations * 10))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_1a1d1e,
                                     number_iterations=number_iterations * 30))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_1a1d1CCC,
                                     number_iterations=number_iterations * 50))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_1a1d1e1CCC,
                                     number_iterations=number_iterations * 100))

    println("+++ " * subname * " 2a model +++")

    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_2a,
                                     number_iterations=number_iterations))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_2a1d,
                                     number_iterations=number_iterations * 10))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_2a1d1e,
                                     number_iterations=number_iterations * 30))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_2a1d1CCC,
                                     number_iterations=number_iterations * 50))
    push!(eval_result,
          fit_and_evaluate_miniblock(env, realsub; model_type=:_2a1d1e1CCC,
                                     number_iterations=number_iterations * 100))

    return eval_result
end

#####
##### Begin evaluation
#####

eval_results = Dict()

for sub_num in 1:36

    if sub_num == 27 || sub_num == 6
        continue
    end

    println("========= Begin Sub " * repr(sub_num) * " ==========")

    each_sub_data = @where(all_data, :Subject_num .== sub_num)
    each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict,
                                                sub_idx_dict)

    println("output result sub " * repr(sub_num))
    eval_results[each_subinfo.sub_tag[1]] = model_evaluation_miniblock(each_env, each_subinfo,
                                                            10000)
end

current_time = Dates.format(now(), "yyyy-mm-dd-HHMMSS")
filename = savepath * "/" * current_time * "final_selection_miniblock.jld2"

@save filename eval_results
