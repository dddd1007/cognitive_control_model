using Models, DataFrames
import CSV

csvpath = joinpath(dirname(pathof(Models)), "..", "..", "data", "output", "RLModels", "model_selection", "BMS")
include("import_all_data.jl")

Threads.@threads for sub_num in 1:36
        
    if sub_num == 27 || sub_num == 6
        continue
    end
    println("========= Begin Sub " * repr(sub_num) * " ==========")
    
    each_sub_data = @where(all_data, :Subject_num .== sub_num)
    each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict,
                                                          sub_idx_dict)
    BIC_results = model_evaluation(each_env, each_subinfo, criteria=:BIC)
    table = DataFrame(BIC_results', [:single_alpha, :single_alpha_no_decay, :no_decay, 
                                     :single_alpha_total_decay, :total_decay, :basic, 
                                     :error, :same_CCC, :diff_CCC])

    filename = convert(String, each_env.sub_tag[1]) * "_BIC.csv"
    filepath = joinpath(csvpath, filename)
    CSV.write(filepath, table)
end
