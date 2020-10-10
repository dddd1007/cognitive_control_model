### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 383dd75c-060e-11eb-1fd1-85f6eb33a9af
using DataFrames, DataFramesMeta, CSV, Statistics, StatsBase

# ╔═╡ 49632e92-060e-11eb-05a8-9b0fccf669bd
include("/Users/dddd1007/project2git/cognitive_control_model/models/DataImporter.jl")

# ╔═╡ 97d7aa1e-0ad5-11eb-2713-23a719879a30
include("/Users/dddd1007/project2git/cognitive_control_model/models/RLModels.jl")

# ╔═╡ a60d9102-060a-11eb-1c04-fdb0ce5006ac
md"# 编写模型各功能"

# ╔═╡ ec09183e-060a-11eb-2690-c9aa2c7e2a31
md"## 测试DataImporter模块； 导入数据"

# ╔═╡ 30a9fb66-0644-11eb-08d7-310a38336154
#导入一个被试的数据开始分析
all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv");

# ╔═╡ 64a66ae6-077d-11eb-2e8c-5bdbc301926d
head(all_data)

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

# ╔═╡ 46a7cac6-0acf-11eb-2ba8-a175c4a140e9
begin
	α_v = rand([0.1:0.01:1.0;])
	β_v = rand([0.1:0.01:5.0;])
	α_s = rand([0.1:0.01:1.0;])
	β_s = rand([0.1:0.01:5.0;])
	decay = rand([0.1:0.01:1.0;])
	sub1_agent = Learner_basic(α_v, β_v, α_s, β_s, decay)
	rl_learning(sub1_env, sub1_agent, sub1_subinfo)
end

# ╔═╡ e53f76ee-08ad-11eb-00fd-8b7ea211b136
begin
	number_iterations = 10000
	result_table = zeros(Float64, (number_iterations,6))
	Threads.@threads for i in 1:number_iterations
		α_v = rand([0.1:0.01:1.0;])
		β_v = rand([0.1:0.01:5.0;])
		α_s = rand([0.1:0.01:1.0;])
		β_s = rand([0.1:0.01:5.0;])
		decay = rand([0.1:0.01:1.0;])
		sub1_agent = Learner_basic(α_v, β_v, α_s, β_s, decay)
		result_table[i,:] = rl_learning(sub1_env, sub1_agent, sub1_subinfo)
	end
end

# ╔═╡ 2412cc44-0ac3-11eb-337e-cd2299f61f01
StatsBase.summarystats(result_table[:,6])

# ╔═╡ fb79a89c-09db-11eb-15e8-6d7305e1a0f2
md"### 模型2 学习 S-R 联结的强化学习模型, 使用 SoftMax 决策, 错误试次下学习率不同"

# ╔═╡ bc5253a8-0ad2-11eb-069e-a332aff4bb20
sub1_subinfo.corrections

# ╔═╡ Cell order:
# ╠═a60d9102-060a-11eb-1c04-fdb0ce5006ac
# ╟─ec09183e-060a-11eb-2690-c9aa2c7e2a31
# ╠═383dd75c-060e-11eb-1fd1-85f6eb33a9af
# ╠═49632e92-060e-11eb-05a8-9b0fccf669bd
# ╠═97d7aa1e-0ad5-11eb-2713-23a719879a30
# ╠═30a9fb66-0644-11eb-08d7-310a38336154
# ╠═64a66ae6-077d-11eb-2e8c-5bdbc301926d
# ╠═91240684-060e-11eb-3e4a-d1c1878d8c3a
# ╠═ba1d75b4-077d-11eb-22de-e35a8ca85485
# ╠═5cb10ab4-07d8-11eb-323a-355d87f5075d
# ╠═8360a24a-062d-11eb-2ae2-891d1573864c
# ╠═a4f8b8c6-0715-11eb-3453-9d4442e54113
# ╠═622e58a4-087f-11eb-12aa-c9e85acc2d8e
# ╠═9a0ce0ec-09db-11eb-3142-2d5c455cf2f8
# ╠═46a7cac6-0acf-11eb-2ba8-a175c4a140e9
# ╠═e53f76ee-08ad-11eb-00fd-8b7ea211b136
# ╠═2412cc44-0ac3-11eb-337e-cd2299f61f01
# ╠═fb79a89c-09db-11eb-15e8-6d7305e1a0f2
# ╠═bc5253a8-0ad2-11eb-069e-a332aff4bb20
