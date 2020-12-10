using JLD2, DataFrames, FileIO, CSV

@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-12-09-155938final_selection.jld2"

DataTable = DataFrame(subject = [], _1a = [], _1a1d = [], _1a1d1e = [], _1a1d1CCC = [], _1a1d1e1CCC = [], _2a = [], _2a1d = [], _2a1d1e = [], _2a1d1CCC = [], _2a1d1e1CCC = [])
subject_names = collect(keys(eval_results))

for i in values(eval_results)
    result = []
	push!(result, i[1])
	
	for j in 2:11
        push!(result, i[j][:eval_result][:AIC])
	end
	
    push!(DataTable, result)
end

CSV.write("/Users/dddd1007/Dropbox/My Mac (eXrld-MBP)/Desktop/model_selection_final.csv", DataTable)