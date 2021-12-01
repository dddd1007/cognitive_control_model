using JLD2, DataFrames, DataFramesMeta, FileIO, CSV, Models

@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-12-09-155938final_selection.jld2"

include("/Users/dddd1007/project2git/cognitive_control_model/scripts/import_all_data.jl")

# Beginning of for loop
sub_names = collect(keys(eval_results))

single_sub = sub_names[1]
optim_param = eval_results[single_sub][10][:optim_param]

α_s = optim_param[:α_s]