### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 383dd75c-060e-11eb-1fd1-85f6eb33a9af
using DataFrames, DataFramesMeta, CSV

# ╔═╡ 49632e92-060e-11eb-05a8-9b0fccf669bd
include("/Users/dddd1007/project2git/cognitive_control_model/models/DataImporter.jl")

# ╔═╡ a60d9102-060a-11eb-1c04-fdb0ce5006ac
md"# 编写模型各功能的测试notebook"

# ╔═╡ ec09183e-060a-11eb-2690-c9aa2c7e2a31
md"## DataImporter模块"

# ╔═╡ 30a9fb66-0644-11eb-08d7-310a38336154
#导入一个被试的数据开始分析
all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv")

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
sub1_data = @where(all_data, :Subject .== "sub01_Yangmiao")

# ╔═╡ da4a7022-07e0-11eb-2231-cd7ee67d5287
tryparse.(Float64, sub1_data.RT)

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

# ╔═╡ Cell order:
# ╠═a60d9102-060a-11eb-1c04-fdb0ce5006ac
# ╠═ec09183e-060a-11eb-2690-c9aa2c7e2a31
# ╠═383dd75c-060e-11eb-1fd1-85f6eb33a9af
# ╠═49632e92-060e-11eb-05a8-9b0fccf669bd
# ╠═30a9fb66-0644-11eb-08d7-310a38336154
# ╠═64a66ae6-077d-11eb-2e8c-5bdbc301926d
# ╠═91240684-060e-11eb-3e4a-d1c1878d8c3a
# ╠═ba1d75b4-077d-11eb-22de-e35a8ca85485
# ╠═5cb10ab4-07d8-11eb-323a-355d87f5075d
# ╠═da4a7022-07e0-11eb-2231-cd7ee67d5287
# ╠═8360a24a-062d-11eb-2ae2-891d1573864c
# ╠═a4f8b8c6-0715-11eb-3453-9d4442e54113
