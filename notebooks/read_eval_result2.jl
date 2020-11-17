### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ b7211716-27d6-11eb-0baa-e5791b5954ee
using JLD2, Models

# ╔═╡ 2b61fe22-27d7-11eb-1a33-194ac35b816b
@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-11-12-065333.jld2" eval_results

# ╔═╡ 5a173d42-27d7-11eb-22df-a501241a8701
eval_results

# ╔═╡ c3fb7ef8-27d7-11eb-1411-b9d0ebf4b21e
md"## 获取关于结果的表格"

# ╔═╡ dd8b42d6-27d7-11eb-337d-2f5ab764aad8
subject_names = keys(eval_results)

# ╔═╡ dd6137d8-27d8-11eb-27b0-190c68ba8a88
query_result = eval_results['sub25_LuSihan']['single_alpha']

# ╔═╡ 601d0ccc-27d9-11eb-2551-0feb4f859f46


# ╔═╡ Cell order:
# ╠═b7211716-27d6-11eb-0baa-e5791b5954ee
# ╠═2b61fe22-27d7-11eb-1a33-194ac35b816b
# ╠═5a173d42-27d7-11eb-22df-a501241a8701
# ╠═c3fb7ef8-27d7-11eb-1411-b9d0ebf4b21e
# ╠═dd8b42d6-27d7-11eb-337d-2f5ab764aad8
# ╠═dd6137d8-27d8-11eb-27b0-190c68ba8a88
# ╠═601d0ccc-27d9-11eb-2551-0feb4f859f46
