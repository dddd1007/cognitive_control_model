using JLD2, DataFrames, DataFramesMeta, FileIO, CSV

@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2021-01-07-145437final_selection_correct_CCC.jld2"

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

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/model_selection_correct_CCC.csv", DataTable)

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
_2a1d1e = sort!(select(Q_value_table, :subject, :_2a1d1e))

pure_all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv", DataFrame)
_2a1d1CCC_p = Array{Float64,1}(undef, nrow(pure_all_data))
pure_all_data[!, :_2a1d1CCC] = _2a1d1CCC_p
_2a1d1e_p = Array{Float64,1}(undef, nrow(pure_all_data))
pure_all_data[!, :_2a1d1e] = _2a1d1e_p

new_dataframe = @where(pure_all_data, :Subject .== 37)
for i in _2a1d1CCC[!, :subject]
    input_dataframe1 = @where(_2a1d1CCC, :subject .== i)
    input_dataframe2 = @where(_2a1d1e,   :subject .== i)
    foo = @where(pure_all_data, :Subject .== i, :Type .!= "miss")
    foo._2a1d1CCC = input_dataframe1._2a1d1CCC[1]
    foo._2a1d1e = input_dataframe2._2a1d1e[1]
    new_dataframe = vcat(new_dataframe, foo)
end

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_2a1d1CCC_2a1d1e.csv", new_dataframe)

optim_para_table = DataFrame(subject = [], a_s = [], a_v = [], CCC = [], a_CCC = [], decay = [], AIC = [], MSE = [])
for i in sort(subject_names)
    optim_para = values(eval_results[i][10][:optim_param])
    mse = [eval_results[i][10][:eval_result][:AIC], eval_results[i][10][:eval_result][:MSE]]
    foo = [i, optim_para..., mse...]
    push!(optim_para_table, foo)
end

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/2a1d1CCC_optim_para.csv", optim_para_table)

@load "/Data3/Xiaxk/research_data/cognitive_control_model/data/output/RLModels/model_selection/2021-02-28-010313final_selection_miniblock.jld2"

DataTable = DataFrame(subject = [], _1a = [], _1a1d = [], _1a1d1e = [], _1a1d1CCC = [], _1a1d1e1CCC = [], _2a = [], _2a1d = [], _2a1d1e = [], _2a1d1CCC = [], _2a1d1e1CCC = [])
subject_names = collect(keys(eval_results))

for i in values(eval_results)
    result = []
	push!(result, i[1])

	for j in 2:11
        push!(result, i[j][:AIC])
	end

    push!(DataTable, result)
end

CSV.write("/Data3/Xiaxk/research_data/cognitive_control_model/data/output/summary/model_selection_miniblock.csv", DataTable)
