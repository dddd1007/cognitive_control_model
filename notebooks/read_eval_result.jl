### A Pluto.jl notebook ###
# v0.12.6

using Markdown
using InteractiveUtils

# ╔═╡ c1b284d4-19ef-11eb-1bc4-4fcaa0c69d32
using JLD2, GLM, DataFrames

# ╔═╡ f958d8c8-1e6f-11eb-2a90-4b871154b6b4
using Models

# ╔═╡ 63f8e116-1e7b-11eb-05c7-11a295c5f92f
using CSV

# ╔═╡ ffbf48c8-1e6f-11eb-2457-6756a4adad7e
include("/Users/dddd1007/project2git/cognitive_control_model/tmp/init_sub1_data.jl")

# ╔═╡ 96be983c-1dd0-11eb-1357-fffbc6026e5b
@load "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/model_selection/2020-11-04-095421_sub1_simple_models.jld2" eval_results

# ╔═╡ ad5bda0a-1dd0-11eb-0f61-21147035bf5d
eval_results["sub01_Yangmiao"]

# ╔═╡ 7e889710-1e4b-11eb-26e8-49060ac794f0
md"## Model Recovery and Eval"

# ╔═╡ 33cd744a-1e71-11eb-1ba8-2fb53c591f47
# Single Alpha
begin
	single_alpha_wang = model_recovery(sub1_env, sub1_subinfo, [0.0001], 
		model_type = :single_alpha_no_decay)
	single_alpha_xia = model_recovery(sub1_env, sub1_subinfo, [1.0], 
		model_type = :single_alpha_no_decay)
end

# ╔═╡ f76c60a0-1e71-11eb-268c-8727014ac338
# Calc MSE
begin
	test_data = DataFrame(y = sub1_subinfo.RT, 
		x_wang = single_alpha_wang[:p_selection_history], 
		x_xia = single_alpha_xia[:p_selection_history])
    α_xia = lm(@formula(y ~ x_xia), test_data)
	α_wang = lm(@formula(y ~ x_wang), test_data)
end

# ╔═╡ f62ac668-1e72-11eb-39c2-9f7854a1ef59
# 老师的MSE
deviance(α_wang)/dof_residual(α_wang)

# ╔═╡ 66a53bf8-1e73-11eb-3e86-f1baf62d6fb5
# Calc PE MSE
begin
	test_data_pe = DataFrame(y = sub1_subinfo.RT, 
		x_wang_pe = single_alpha_wang[:prediction_error], 
		x_xia_pe = single_alpha_xia[:prediction_error])
    α_xia_pe = lm(@formula(y ~ x_xia_pe), test_data_pe)
	α_wang_pe = lm(@formula(y ~ x_wang_pe), test_data_pe)
end

# ╔═╡ a309e17a-1e73-11eb-1167-b54371104747
# 老师的MSE
deviance(α_wang_pe)/dof_residual(α_wang_pe)

# ╔═╡ 6ae7720c-1e7b-11eb-0039-3b5580bd3d9f
correData = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/ref_code/Lingwang_CCC/data/sub01.csv");

# ╔═╡ 88e15384-1e7b-11eb-3bc2-aff9bfe96868
begin
	test_data_wangtype = DataFrame(y = sub1_subinfo.RT, 
		p_diff = 2 * single_alpha_wang[:p_selection_history] .- 1,
		congruency = correData[:congruency], hand = correData[:hand], 					    postError = correData[:postError], block1 = correData[:block1], 			           block2 = correData[:block2], block3 = correData[:block3])
end

# ╔═╡ 08c8962a-1e7c-11eb-1098-2500daf6df22
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/ref_code/Lingwang_CCC/data/wang_rl_rest.csv", test_data_wangtype)

# ╔═╡ Cell order:
# ╠═c1b284d4-19ef-11eb-1bc4-4fcaa0c69d32
# ╠═96be983c-1dd0-11eb-1357-fffbc6026e5b
# ╠═ad5bda0a-1dd0-11eb-0f61-21147035bf5d
# ╠═7e889710-1e4b-11eb-26e8-49060ac794f0
# ╠═f958d8c8-1e6f-11eb-2a90-4b871154b6b4
# ╠═ffbf48c8-1e6f-11eb-2457-6756a4adad7e
# ╠═33cd744a-1e71-11eb-1ba8-2fb53c591f47
# ╠═f76c60a0-1e71-11eb-268c-8727014ac338
# ╠═f62ac668-1e72-11eb-39c2-9f7854a1ef59
# ╠═66a53bf8-1e73-11eb-3e86-f1baf62d6fb5
# ╠═a309e17a-1e73-11eb-1167-b54371104747
# ╠═63f8e116-1e7b-11eb-05c7-11a295c5f92f
# ╠═6ae7720c-1e7b-11eb-0039-3b5580bd3d9f
# ╠═88e15384-1e7b-11eb-3bc2-aff9bfe96868
# ╠═08c8962a-1e7c-11eb-1098-2500daf6df22
