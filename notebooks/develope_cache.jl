### A Pluto.jl notebook ###
# v0.12.3

using Markdown
using InteractiveUtils

# ╔═╡ 49632e92-060e-11eb-05a8-9b0fccf669bd
begin
	push!(LOAD_PATH, "/Users/dddd1007/project2git/cognitive_control_model/models")
	using DataManipulate
	include("/Users/dddd1007/project2git/cognitive_control_model/models/RLModels.jl")
end

# ╔═╡ 383dd75c-060e-11eb-1fd1-85f6eb33a9af
using DataFrames, DataFramesMeta, CSV, Statistics, StatsBase, GLM, Plots

# ╔═╡ a60d9102-060a-11eb-1c04-fdb0ce5006ac
md"# 编写模型各功能"

# ╔═╡ ec09183e-060a-11eb-2690-c9aa2c7e2a31
md"## 测试DataImporter模块； 导入数据"

# ╔═╡ 57f65b86-0d26-11eb-3de0-b5af36db7c1e


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
end

# ╔═╡ e580923a-0c6a-11eb-1207-65691ca00888
result1 = rl_learning_sr(sub1_env, sub1_agent, sub1_subinfo)

# ╔═╡ fb79a89c-09db-11eb-15e8-6d7305e1a0f2
md"### 模型2 学习 S-R 联结的强化学习模型, 使用 SoftMax 决策, 错误试次下学习率不同"

# ╔═╡ 87c3559e-0cfe-11eb-16db-7746f3add5e5
md"## 编写模型诊断部分"

# ╔═╡ 9422943a-0cfe-11eb-09de-438abdf68a93
x = result1["p_softmax_history"]

# ╔═╡ b7939784-0cfe-11eb-0977-5996c25272e9
y = sub1_subinfo.RT

# ╔═╡ bdebbde2-0d16-11eb-1059-f3d7784ba3f6
data = DataFrame(x = x, y = y);

# ╔═╡ bdb28b52-0cfe-11eb-2bd3-a5504eaa1f42
probe = lm(@formula(y~x), data)

# ╔═╡ 3d19c4d8-0d17-11eb-1f8f-53597a42ed2f
plot(x,y,seriestype = :scatter)

# ╔═╡ f1c31cdc-0d16-11eb-131d-4b112501e444
aic(probe)

# ╔═╡ 1e7a2ca2-0d17-11eb-205d-cf451ad6738f
bic(probe)

# ╔═╡ 17964e92-0d18-11eb-3562-b91129af4c89
dof_residual(probe)

# ╔═╡ 2af07aa8-0d18-11eb-1f23-ef8e8e1ff744
coef(probe)[2]

# ╔═╡ 8b09261a-0d18-11eb-14d5-c586f78204cd
r2(probe)

# ╔═╡ Cell order:
# ╠═a60d9102-060a-11eb-1c04-fdb0ce5006ac
# ╟─ec09183e-060a-11eb-2690-c9aa2c7e2a31
# ╠═383dd75c-060e-11eb-1fd1-85f6eb33a9af
# ╠═49632e92-060e-11eb-05a8-9b0fccf669bd
# ╟─57f65b86-0d26-11eb-3de0-b5af36db7c1e
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
# ╠═e580923a-0c6a-11eb-1207-65691ca00888
# ╠═fb79a89c-09db-11eb-15e8-6d7305e1a0f2
# ╠═87c3559e-0cfe-11eb-16db-7746f3add5e5
# ╠═9422943a-0cfe-11eb-09de-438abdf68a93
# ╠═b7939784-0cfe-11eb-0977-5996c25272e9
# ╠═bdebbde2-0d16-11eb-1059-f3d7784ba3f6
# ╠═bdb28b52-0cfe-11eb-2bd3-a5504eaa1f42
# ╠═3d19c4d8-0d17-11eb-1f8f-53597a42ed2f
# ╠═f1c31cdc-0d16-11eb-131d-4b112501e444
# ╠═1e7a2ca2-0d17-11eb-205d-cf451ad6738f
# ╠═17964e92-0d18-11eb-3562-b91129af4c89
# ╠═2af07aa8-0d18-11eb-1f23-ef8e8e1ff744
# ╠═8b09261a-0d18-11eb-14d5-c586f78204cd
