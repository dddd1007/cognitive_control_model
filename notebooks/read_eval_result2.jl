### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ b7211716-27d6-11eb-0baa-e5791b5954ee
using JLD2, Models

# ╔═╡ c97206aa-27e0-11eb-08b0-71e374f93c88
using DataFrames

# ╔═╡ 8eea045a-27e1-11eb-21e4-8b71eaff7541
using CSV

# ╔═╡ 2b61fe22-27d7-11eb-1a33-194ac35b816b
@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-11-12-065333.jld2" eval_results

# ╔═╡ 5a173d42-27d7-11eb-22df-a501241a8701
eval_results

# ╔═╡ c3fb7ef8-27d7-11eb-1411-b9d0ebf4b21e
md"## 获取关于结果的表格"

# ╔═╡ dd8b42d6-27d7-11eb-337d-2f5ab764aad8
subject_names = keys(eval_results)

# ╔═╡ dd6137d8-27d8-11eb-27b0-190c68ba8a88
query_result = eval_results["sub25_LuSihan"]

# ╔═╡ 601d0ccc-27d9-11eb-2551-0feb4f859f46
function show_aic(eval_results, index)
	[eval_results.single_alpha[:eval_result][index],
		eval_results.single_alpha_no_decay[:eval_result][index],
		eval_results.no_decay[:eval_result][index],
		eval_results.basic[:eval_result][index],
		eval_results.error[:eval_result][index],
		eval_results.CCC_same_alpha[:eval_result][index],
		eval_results.CCC_different_alpha[:eval_result][index]]
end

# ╔═╡ 3b0f2ec4-27e0-11eb-244c-6709a228f931
foo = []

# ╔═╡ 0164e7ec-27e3-11eb-1406-fb26c1ef9ac6
show_aic(eval_results["sub31_TanPeixian"], :MSE)

# ╔═╡ f6530bbc-27de-11eb-10ac-a1e9de05c68f
for i in collect(subject_names)
	push!(foo, [i, show_aic(eval_results[i], :MSE)...])
end

# ╔═╡ 9becefc2-27e2-11eb-1f9c-1376fe2caf1f
#foo

# ╔═╡ f1ac02ae-27e0-11eb-03e5-9d2614ffd5b9
foo2 = DataFrame(foo)

# ╔═╡ bf923aa0-27e1-11eb-1937-69d27fd159da
CSV.write("/Users/dddd1007/Dropbox/My Mac (eXrld-MBP)/Desktop/test.csv", foo2)

# ╔═╡ Cell order:
# ╠═b7211716-27d6-11eb-0baa-e5791b5954ee
# ╠═2b61fe22-27d7-11eb-1a33-194ac35b816b
# ╠═5a173d42-27d7-11eb-22df-a501241a8701
# ╠═c3fb7ef8-27d7-11eb-1411-b9d0ebf4b21e
# ╠═dd8b42d6-27d7-11eb-337d-2f5ab764aad8
# ╠═dd6137d8-27d8-11eb-27b0-190c68ba8a88
# ╠═601d0ccc-27d9-11eb-2551-0feb4f859f46
# ╠═c97206aa-27e0-11eb-08b0-71e374f93c88
# ╠═3b0f2ec4-27e0-11eb-244c-6709a228f931
# ╠═0164e7ec-27e3-11eb-1406-fb26c1ef9ac6
# ╠═f6530bbc-27de-11eb-10ac-a1e9de05c68f
# ╠═9becefc2-27e2-11eb-1f9c-1376fe2caf1f
# ╠═f1ac02ae-27e0-11eb-03e5-9d2614ffd5b9
# ╠═8eea045a-27e1-11eb-21e4-8b71eaff7541
# ╠═bf923aa0-27e1-11eb-1937-69d27fd159da
