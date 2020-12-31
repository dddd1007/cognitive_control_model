### A Pluto.jl notebook ###
# v0.12.18

using Markdown
using InteractiveUtils

# ╔═╡ 252d0d72-437b-11eb-14f8-178f68c0822a
using JLD2, DataFrames, DataFramesMeta, FileIO, CSV, Models

# ╔═╡ cc772372-4a53-11eb-27ff-01b3b6a3bcd2
using StatsBase

# ╔═╡ 4f816fb6-4a54-11eb-16be-19db593b1461
using HypothesisTests

# ╔═╡ ab327f3e-4a55-11eb-0d6d-a7ab651a258c
using Plots

# ╔═╡ 3f0fb80c-437b-11eb-1aa4-13904f69a0c3
include("/Users/dddd1007/project2git/cognitive_control_model/tmp/init_sub1_data.jl")

# ╔═╡ d526d122-4691-11eb-2b31-9f9df9dbea69
@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-12-09-155938final_selection.jld2"

# ╔═╡ 5daf372a-4695-11eb-3bce-bdf9ec622426
md"选择匹配的被试列表和最优参数"

# ╔═╡ 2bf48d06-4696-11eb-08f8-ef20da7d369d
single_sub = "sub01_Yangmiao"

# ╔═╡ 42792686-4696-11eb-2bcc-0d86b26850b7
single_optim_params = eval_results[single_sub][5][:optim_param]

# ╔═╡ 878021dc-4699-11eb-0822-076c7c4eec79
all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv", DataFrame)

# ╔═╡ b0b55704-469a-11eb-3c57-c14c412009d6
single_sub_table = @where(all_data, :Subject .== single_sub)

# ╔═╡ d727bbe0-469a-11eb-1997-5d0ef8ad65c6
conflict_list = model_recovery(sub1_env, sub1_subinfo, single_optim_params, model_type = :_1a1d1CCC)[:conflict_list]

# ╔═╡ d672da24-469d-11eb-3517-e973151723c5
conflict_list .> single_optim_params[:CCC]

# ╔═╡ 27d229ec-469e-11eb-097e-b3039e676ae0
single_sub_table.if_over_CCC = conflict_list .> single_optim_params[:CCC]

# ╔═╡ a0364e6a-48e7-11eb-2390-af6902e3af03
second_conflict_list = push!([false], single_sub_table.if_over_CCC...)

# ╔═╡ ee37b32e-48e7-11eb-2c7d-09b0187e20ee
a = [1,2,3]

# ╔═╡ f1fd5f02-48e7-11eb-0461-c3185097dd4f
deleteat!(a, 3)

# ╔═╡ d483e090-48e7-11eb-13f5-919b86527ab9
single_sub_table.if_second_CCC = deleteat!(second_conflict_list, 961)

# ╔═╡ 3c129cca-469e-11eb-3863-572568442d76
single_sub_table

# ╔═╡ 7a947288-4a50-11eb-2667-55c9d4347f90
begin
	cache_row = single_sub_table[2, :]
	cache_second_row = single_sub_table[2+1, :]
end

# ╔═╡ 7eb5bffa-4a50-11eb-371a-330773a98c78
typeof(cache_row.stim_color == cache_second_row.stim_color)

# ╔═╡ 5bf92592-48ea-11eb-0297-3974e5194bdf
md"接下来把CCC当前trial和之后的trial一致的表格提取出来, 再把不一致的也提取出来看看"

# ╔═╡ e1a7aa90-4a40-11eb-36a5-79131f701cad
begin
	same_stim_index = []
	for i in 1:(nrow(single_sub_table)-1)
		cache_row = single_sub_table[i, :]
		cache_second_row = single_sub_table[i+1, :]
		if cache_row.if_over_CCC && cache_row.stim_color == cache_second_row.stim_color && cache_row.stim_loc == cache_second_row.stim_loc
			push!(same_stim_index, i)
		end
	end	
end

# ╔═╡ 51acdf30-4a51-11eb-2f44-f7f6f0260e69
same_stim_index

# ╔═╡ 5ffa7688-4a51-11eb-02d7-f9cf67be69b4
begin
	same_stim_table_part1 = single_sub_table[same_stim_index, :]
	same_stim_table_part2 = single_sub_table[same_stim_index .+ 1, :]
end

# ╔═╡ 7de685fe-4a53-11eb-0813-7d4ec74846e7
parse.(Float64,same_stim_table_part1.RT)

# ╔═╡ 25da5fac-4a53-11eb-0e8a-6f22ad8ad0bc
samestim_RT_list = parse.(Float64, same_stim_table_part1.RT) .- parse.(Float64, same_stim_table_part2.RT)

# ╔═╡ c0842b28-4a53-11eb-3707-85142981229f
mean(samestim_RT_list)

# ╔═╡ d2d97986-4a53-11eb-0e03-a73c15d1a016
begin
	same_stim_noCCC_index = []
	for i in 1:(nrow(single_sub_table)-1)
		cache_row = single_sub_table[i, :]
		cache_second_row = single_sub_table[i+1, :]
		if !cache_row.if_over_CCC && cache_row.stim_color == cache_second_row.stim_color && cache_row.stim_loc == cache_second_row.stim_loc
			push!(same_stim_noCCC_index, i)
		end
	end	
end

# ╔═╡ f390f672-4a53-11eb-218a-8d4ba1914f51
same_stim_noCCC_index

# ╔═╡ 0e50a16a-4a54-11eb-3de2-a9b6e2440113
begin
	same_stim_noCCC_table_part1 = single_sub_table[same_stim_noCCC_index, :]
	same_stim_noCCC_table_part2 = single_sub_table[same_stim_noCCC_index .+ 1, :]
end

# ╔═╡ 2751ce00-4a54-11eb-2cc6-27eca8b345bb
begin 
	samestim_noCCC_RT_list = parse.(Float64, same_stim_noCCC_table_part1.RT) .- parse.(Float64, same_stim_noCCC_table_part2.RT)
	mean(samestim_noCCC_RT_list)
end

# ╔═╡ 3092a03c-4a59-11eb-2686-7f8e15d17b84
samestim_RT_list

# ╔═╡ 68a53f2c-4a54-11eb-0b66-f54c37c8d4e9
UnequalVarianceTTest(samestim_RT_list, samestim_noCCC_RT_list)

# ╔═╡ 174c97f4-4a59-11eb-0cf4-83fcaa80ca23
MannWhitneyUTest(samestim_RT_list, samestim_noCCC_RT_list)

# ╔═╡ 58e7cc72-4a58-11eb-3fc3-4fef77726434
histogram(samestim_RT_list)

# ╔═╡ b9f22670-4a58-11eb-302f-2d6857efb9d3
histogram(samestim_noCCC_RT_list)

# ╔═╡ Cell order:
# ╠═252d0d72-437b-11eb-14f8-178f68c0822a
# ╠═3f0fb80c-437b-11eb-1aa4-13904f69a0c3
# ╠═d526d122-4691-11eb-2b31-9f9df9dbea69
# ╟─5daf372a-4695-11eb-3bce-bdf9ec622426
# ╠═2bf48d06-4696-11eb-08f8-ef20da7d369d
# ╠═42792686-4696-11eb-2bcc-0d86b26850b7
# ╠═878021dc-4699-11eb-0822-076c7c4eec79
# ╠═b0b55704-469a-11eb-3c57-c14c412009d6
# ╠═d727bbe0-469a-11eb-1997-5d0ef8ad65c6
# ╠═d672da24-469d-11eb-3517-e973151723c5
# ╠═27d229ec-469e-11eb-097e-b3039e676ae0
# ╠═a0364e6a-48e7-11eb-2390-af6902e3af03
# ╠═ee37b32e-48e7-11eb-2c7d-09b0187e20ee
# ╠═f1fd5f02-48e7-11eb-0461-c3185097dd4f
# ╠═d483e090-48e7-11eb-13f5-919b86527ab9
# ╠═3c129cca-469e-11eb-3863-572568442d76
# ╠═7a947288-4a50-11eb-2667-55c9d4347f90
# ╠═7eb5bffa-4a50-11eb-371a-330773a98c78
# ╟─5bf92592-48ea-11eb-0297-3974e5194bdf
# ╠═e1a7aa90-4a40-11eb-36a5-79131f701cad
# ╠═51acdf30-4a51-11eb-2f44-f7f6f0260e69
# ╠═5ffa7688-4a51-11eb-02d7-f9cf67be69b4
# ╠═7de685fe-4a53-11eb-0813-7d4ec74846e7
# ╠═25da5fac-4a53-11eb-0e8a-6f22ad8ad0bc
# ╠═cc772372-4a53-11eb-27ff-01b3b6a3bcd2
# ╠═c0842b28-4a53-11eb-3707-85142981229f
# ╠═d2d97986-4a53-11eb-0e03-a73c15d1a016
# ╠═f390f672-4a53-11eb-218a-8d4ba1914f51
# ╠═0e50a16a-4a54-11eb-3de2-a9b6e2440113
# ╠═2751ce00-4a54-11eb-2cc6-27eca8b345bb
# ╠═3092a03c-4a59-11eb-2686-7f8e15d17b84
# ╠═4f816fb6-4a54-11eb-16be-19db593b1461
# ╠═68a53f2c-4a54-11eb-0b66-f54c37c8d4e9
# ╠═174c97f4-4a59-11eb-0cf4-83fcaa80ca23
# ╠═ab327f3e-4a55-11eb-0d6d-a7ab651a258c
# ╠═58e7cc72-4a58-11eb-3fc3-4fef77726434
# ╠═b9f22670-4a58-11eb-302f-2d6857efb9d3
