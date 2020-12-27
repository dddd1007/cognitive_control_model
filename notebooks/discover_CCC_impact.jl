### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 252d0d72-437b-11eb-14f8-178f68c0822a
using JLD2, DataFrames, DataFramesMeta, FileIO, CSV, Models

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

# ╔═╡ 3c129cca-469e-11eb-3863-572568442d76
single_sub_table

# ╔═╡ Cell order:
# ╠═252d0d72-437b-11eb-14f8-178f68c0822a
# ╠═3f0fb80c-437b-11eb-1aa4-13904f69a0c3
# ╠═d526d122-4691-11eb-2b31-9f9df9dbea69
# ╠═5daf372a-4695-11eb-3bce-bdf9ec622426
# ╠═2bf48d06-4696-11eb-08f8-ef20da7d369d
# ╠═42792686-4696-11eb-2bcc-0d86b26850b7
# ╠═878021dc-4699-11eb-0822-076c7c4eec79
# ╠═b0b55704-469a-11eb-3c57-c14c412009d6
# ╠═d727bbe0-469a-11eb-1997-5d0ef8ad65c6
# ╠═d672da24-469d-11eb-3517-e973151723c5
# ╠═27d229ec-469e-11eb-097e-b3039e676ae0
# ╠═3c129cca-469e-11eb-3863-572568442d76
