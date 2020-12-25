### A Pluto.jl notebook ###
# v0.12.17

using Markdown
using InteractiveUtils

# ╔═╡ 252d0d72-437b-11eb-14f8-178f68c0822a
using JLD2, DataFrames, DataFramesMeta, FileIO, CSV

# ╔═╡ 596a7708-4378-11eb-08bb-cb858da86173
include("/Users/dddd1007/project2git/cognitive_control_model/scripts/import_all_data.jl")

# ╔═╡ 3f0fb80c-437b-11eb-1aa4-13904f69a0c3
@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-12-09-155938final_selection.jld2"

# ╔═╡ 68897362-437b-11eb-14d9-5721802d6780
subjects_name = collect(keys(eval_results))

# ╔═╡ e0c8e056-437b-11eb-17e0-07145276d761


# ╔═╡ Cell order:
# ╠═596a7708-4378-11eb-08bb-cb858da86173
# ╠═252d0d72-437b-11eb-14f8-178f68c0822a
# ╠═3f0fb80c-437b-11eb-1aa4-13904f69a0c3
# ╠═68897362-437b-11eb-14d9-5721802d6780
# ╠═e0c8e056-437b-11eb-17e0-07145276d761
