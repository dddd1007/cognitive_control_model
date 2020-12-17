using JLD2, DataFrames, DataFramesMeta, FileIO, CSV

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

Q_value_table = DataFrame(subject = [], _1a = [], _1a1d = [], _1a1d1e = [], _1a1d1CCC = [], _1a1d1e1CCC = [], _2a = [], _2a1d = [], _2a1d1e = [], _2a1d1CCC = [], _2a1d1e1CCC = [])
for i in values(eval_results)
    result = []
	push!(result, i[1])
	
	for j in 2:11
        push!(result, i[j][:p_history])
	end
	
    push!(Q_value_table, result)
end

_2a1d1CCC = sort!(select(Q_value_table, :subject, :_2a1d1CCC))

pure_all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv", DataFrame)
_2a1d1CCC_p = Array{Float64,1}(undef, nrow(pure_all_data))
pure_all_data[!, :_2a1d1CCC] = _2a1d1CCC_p

new_dataframe = @where(pure_all_data, :Subject .== 37)
for i in _2a1d1CCC[!, :subject]
    input_dataframe = @where(_2a1d1CCC, :subject .== i)
    foo = @where(pure_all_data, :Subject .== i, :Type .!= "miss")
    println(length(input_dataframe._2a1d1CCC[1]))
    foo._2a1d1CCC = input_dataframe._2a1d1CCC[1]
    new_dataframe = vcat(new_dataframe, foo)
end

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_2a1d1CCC.csv", new_dataframe)