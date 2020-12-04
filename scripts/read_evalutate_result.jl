using JLD2, DataFrames, FileIO, CSV

@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-11-12-065333.jld2"

function show_index(eval_results, index)
	[eval_results.single_alpha[:eval_result][index],
		eval_results.single_alpha_no_decay[:eval_result][index],
		eval_results.no_decay[:eval_result][index],
		eval_results.basic[:eval_result][index],
		eval_results.error[:eval_result][index],
		eval_results.CCC_same_alpha[:eval_result][index],
		eval_results.CCC_different_alpha[:eval_result][index]]
end

foo = DataFrame(subject = [], _1a1d = [], _1a = [], _2a = [], _2a1d = [], _2a1d1e = [], _2a1d1e1CCC = [], _2a1d1e2CCC = [])
subject_names = keys(eval_results)

for i in collect(subject_names)
	push!(foo, [i, show_index(eval_results[i], :AIC)...])
end

CSV.write("/Users/dddd1007/Dropbox/My Mac (eXrld-MBP)/Desktop/first_time.csv", foo)

eval_results = nothing
foo = nothing

@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-12-02-204435CCC_no_error.jld2"

subject_names = collect(keys(eval_results))

DataTable = DataFrame(subject = [], _2a1d1CCC = [], _2a1d12CCC = [])
for i in values(eval_results)
    result = []
    push!(result, i[1])
    push!(result, i[2][:eval_result][:AIC])
    push!(result, i[3][:eval_result][:AIC])
    push!(DataTable, result)
end

CSV.write("/Users/dddd1007/Dropbox/My Mac (eXrld-MBP)/Desktop/second_time.csv", DataTable)