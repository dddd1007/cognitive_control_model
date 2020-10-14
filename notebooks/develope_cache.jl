### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 49632e92-060e-11eb-05a8-9b0fccf669bd
begin
	push!(LOAD_PATH, "/Users/dddd1007/project2git/cognitive_control_model/models")
	using DataManipulate, RLModels_basic, DataFramesMeta, CSV, Statistics, StatsBase, GLM, Plots, DataFrames
    import RLModels_SoftMax, RLModels_no_SoftMax
end

# ╔═╡ 2a5d7276-0df4-11eb-2eed-39e04e00e1e5
using Flux

# ╔═╡ ba570318-0e0c-11eb-2d0b-67a9242e2204
using Hyperopt

# ╔═╡ a60d9102-060a-11eb-1c04-fdb0ce5006ac
md"# 编写模型各功能"

# ╔═╡ ec09183e-060a-11eb-2690-c9aa2c7e2a31
md"## 测试DataImporter模块； 导入数据"

# ╔═╡ 57f65b86-0d26-11eb-3de0-b5af36db7c1e


# ╔═╡ 30a9fb66-0644-11eb-08d7-310a38336154
#导入一个被试的数据开始分析
all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv");

# ╔═╡ 91240684-060e-11eb-3e4a-d1c1878d8c3a
begin
	color_rule = Dict("red" => "0" , "green" => "1")
	congruency_rule = Dict("con" => "1", "inc" => "0")
	Type_rule = Dict("hit" => "1", "incorrect" => "0", "miss" => "0")
	loc_rule = Dict("left" => "0", "right" => "1")
	transform_rule = Dict("stim_color" => color_rule, "Type" => Type_rule, 
		"stim_loc" => loc_rule, "congruency" => congruency_rule)
end

# ╔═╡ ba1d75b4-077d-11eb-22de-e35a8ca85485
transform_data!(all_data, transform_rule)

# ╔═╡ 5cb10ab4-07d8-11eb-323a-355d87f5075d
sub1_data = @where(all_data, :Subject .== "sub01_Yangmiao");

# ╔═╡ 8360a24a-062d-11eb-2ae2-891d1573864c
begin
    env_idx_dict = Dict("stim_task_related" => "stim_color", 
		                "stim_task_unrelated" => "stim_loc", 
		                "stim_action_congruency" => "congruency", 
		                "correct_action" => "correct_action",
		                "env_type" => "condition", "sub_tag" => "Subject")
	sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
		                "corrections" => "Type", "sub_tag" => "Subject")
end

# ╔═╡ a4f8b8c6-0715-11eb-3453-9d4442e54113
sub1_env, sub1_subinfo = init_env_sub(sub1_data, env_idx_dict, sub_idx_dict)

# ╔═╡ 622e58a4-087f-11eb-12aa-c9e85acc2d8e
md"## 强化学习模型"

# ╔═╡ 9a0ce0ec-09db-11eb-3142-2d5c455cf2f8
md"### 模型1 学习 S-R 联结的强化学习模型, 使用 SoftMax 决策, 带 Decay"

# ╔═╡ 1e690c28-0df4-11eb-254b-8b81eed2eff3
md"## 编写最优化部分"

# ╔═╡ 31efd0b2-0dfa-11eb-1e6b-c996cf6deecb
begin
    α_v = range(0,1,step = 0.01)
	α_s = range(0,1,step = 0.01)
	decay = range(0,1,step = 0.01)
end

# ╔═╡ 73efba36-0e0b-11eb-040c-e91eefbf0aff
θ = params([α_v, α_s, decay])

# ╔═╡ ab63b618-0e18-11eb-3711-7377713dd74f
abstract type Learner end

# ╔═╡ 65fa15da-0e19-11eb-3802-b3e48ba5bf33
begin
	ho = @phyperopt for i = 10000,
					   α_v = [0.1:0.1:1;],
					   α_s = [0.1:0.1:1;],
					   decay = [0.1:0.1:1;]
		
		agent = RLModels_no_SoftMax.Learner_basic(α_v, α_s, decay)
		model_stim = RLModels_no_SoftMax.rl_learning_sr(sub1_env, agent, sub1_subinfo)
		RLModels_basic.evaluate_relation(model_stim[:p_selection_history], sub1_subinfo.RT)[:MSE]
	end
end

# ╔═╡ 0f9b2e2a-0e1b-11eb-2ced-674a5af761bb
plot(ho)

# ╔═╡ 4352aaee-0e1c-11eb-0077-07b0aeeb97e2
ho

# ╔═╡ Cell order:
# ╠═a60d9102-060a-11eb-1c04-fdb0ce5006ac
# ╟─ec09183e-060a-11eb-2690-c9aa2c7e2a31
# ╠═49632e92-060e-11eb-05a8-9b0fccf669bd
# ╟─57f65b86-0d26-11eb-3de0-b5af36db7c1e
# ╠═30a9fb66-0644-11eb-08d7-310a38336154
# ╠═91240684-060e-11eb-3e4a-d1c1878d8c3a
# ╠═ba1d75b4-077d-11eb-22de-e35a8ca85485
# ╠═5cb10ab4-07d8-11eb-323a-355d87f5075d
# ╠═8360a24a-062d-11eb-2ae2-891d1573864c
# ╠═a4f8b8c6-0715-11eb-3453-9d4442e54113
# ╠═622e58a4-087f-11eb-12aa-c9e85acc2d8e
# ╠═9a0ce0ec-09db-11eb-3142-2d5c455cf2f8
# ╠═1e690c28-0df4-11eb-254b-8b81eed2eff3
# ╠═2a5d7276-0df4-11eb-2eed-39e04e00e1e5
# ╠═31efd0b2-0dfa-11eb-1e6b-c996cf6deecb
# ╠═73efba36-0e0b-11eb-040c-e91eefbf0aff
# ╠═ab63b618-0e18-11eb-3711-7377713dd74f
# ╠═ba570318-0e0c-11eb-2d0b-67a9242e2204
# ╠═65fa15da-0e19-11eb-3802-b3e48ba5bf33
# ╠═0f9b2e2a-0e1b-11eb-2ced-674a5af761bb
# ╠═4352aaee-0e1c-11eb-0077-07b0aeeb97e2
