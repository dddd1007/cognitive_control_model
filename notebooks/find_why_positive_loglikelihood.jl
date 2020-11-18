### A Pluto.jl notebook ###
# v0.12.10

using Markdown
using InteractiveUtils

# ╔═╡ 86aff392-28a8-11eb-3dad-b12acfa2615c
using JLD2, DataFrames, Models, FileIO

# ╔═╡ adf0aa16-28a9-11eb-3dae-4f3e69092f05
using CSV

# ╔═╡ dc62c186-28a9-11eb-348f-f94ca6b22d98
using DataFramesMeta

# ╔═╡ 6ed25b26-28aa-11eb-109a-0167a4175f9b
using GLM

# ╔═╡ dbe590fc-28aa-11eb-30f0-ebe5c8e320c9
using Gadfly

# ╔═╡ a09f9e22-28b2-11eb-0392-a1bfef16d44a
using Distributions

# ╔═╡ 91aceed8-28a8-11eb-1ca7-4568aea9a09b
@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-11-17-153045_debug_positive_loglikelihood.jld2" eval_results

# ╔═╡ 03ea8104-28a9-11eb-3b57-67d984d8f8f9
eval_results

# ╔═╡ 70654c76-28a9-11eb-0840-130b9bf8c173
foo_x = eval_results["sub25_LuSihan"].single_alpha[:p_history]

# ╔═╡ b9b44bbe-28a9-11eb-1059-175bb79edcf2
all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv", DataFrame)

# ╔═╡ e39bc77c-28a9-11eb-2dac-3ff36d17d189
sub25 = @where(all_data, :Subject_num .== 25)

# ╔═╡ 5c70f654-28aa-11eb-37d2-b9fbfae57098
foo_y = parse.(Float64, sub25.RT)

# ╔═╡ 72491ed4-28aa-11eb-04eb-91d734868465
foo_table = DataFrame(x = foo_x, y = log.(foo_y))

# ╔═╡ 88c24794-28aa-11eb-15de-2519058c11e7
foo_results = lm(@formula(y ~ x), foo_table)

# ╔═╡ a0a98af2-28aa-11eb-040f-0d0960f363a5
loglikelihood(foo_results)

# ╔═╡ df7828b2-28aa-11eb-0036-c5aabbce90bb
plot(foo_table, x="y", Geom.histogram)

# ╔═╡ 7956162c-28ba-11eb-165f-afe013b719bd
plot(log(foo_x), Geom.histogram)

# ╔═╡ 73688504-28ad-11eb-1c72-b987ab0fde11
plot(foo_table, x="x", Geom.histogram)

# ╔═╡ ecda4f90-28b1-11eb-2dfe-499dbb2796a7
test_x = (foo_x .*-0.138257) .+ 6.16272

# ╔═╡ 14bafad2-28b2-11eb-2095-53216d2b3574
plot(x = test_x, Geom.histogram)

# ╔═╡ 52ed7c76-28b2-11eb-102c-2397bf2f7c62
diff_xy = log.(foo_y) .- test_x

# ╔═╡ 8ff7ce6e-28b2-11eb-1e0c-d3f075a3b091
plot(x = diff_xy, Geom.histogram)

# ╔═╡ ab8ef476-28b3-11eb-1b7b-bf4087717817
fit(Normal, diff_xy)

# ╔═╡ efb29344-28b3-11eb-0b09-51169ca02329
function calc_likelihood(x)
	likelihood = (1/(sqrt(2π)*0.155790))*exp((-(x - 0)^2) / (2 * (0.155790^2)))
end

# ╔═╡ eea703a4-28b4-11eb-2f7c-fb16500b0590
sum(calc_likelihood.(diff_xy))

# ╔═╡ 001eeec8-28b5-11eb-1551-05572eec8178
calc_likelihood.(diff_xy)

# ╔═╡ Cell order:
# ╠═86aff392-28a8-11eb-3dad-b12acfa2615c
# ╠═91aceed8-28a8-11eb-1ca7-4568aea9a09b
# ╠═03ea8104-28a9-11eb-3b57-67d984d8f8f9
# ╠═70654c76-28a9-11eb-0840-130b9bf8c173
# ╠═adf0aa16-28a9-11eb-3dae-4f3e69092f05
# ╠═b9b44bbe-28a9-11eb-1059-175bb79edcf2
# ╠═dc62c186-28a9-11eb-348f-f94ca6b22d98
# ╠═e39bc77c-28a9-11eb-2dac-3ff36d17d189
# ╠═5c70f654-28aa-11eb-37d2-b9fbfae57098
# ╠═6ed25b26-28aa-11eb-109a-0167a4175f9b
# ╠═72491ed4-28aa-11eb-04eb-91d734868465
# ╠═88c24794-28aa-11eb-15de-2519058c11e7
# ╠═a0a98af2-28aa-11eb-040f-0d0960f363a5
# ╠═dbe590fc-28aa-11eb-30f0-ebe5c8e320c9
# ╠═df7828b2-28aa-11eb-0036-c5aabbce90bb
# ╠═7956162c-28ba-11eb-165f-afe013b719bd
# ╠═73688504-28ad-11eb-1c72-b987ab0fde11
# ╠═ecda4f90-28b1-11eb-2dfe-499dbb2796a7
# ╠═14bafad2-28b2-11eb-2095-53216d2b3574
# ╠═52ed7c76-28b2-11eb-102c-2397bf2f7c62
# ╠═8ff7ce6e-28b2-11eb-1e0c-d3f075a3b091
# ╠═a09f9e22-28b2-11eb-0392-a1bfef16d44a
# ╠═ab8ef476-28b3-11eb-1b7b-bf4087717817
# ╠═efb29344-28b3-11eb-0b09-51169ca02329
# ╠═eea703a4-28b4-11eb-2f7c-fb16500b0590
# ╠═001eeec8-28b5-11eb-1551-05572eec8178
