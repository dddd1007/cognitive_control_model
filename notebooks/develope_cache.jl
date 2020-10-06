### A Pluto.jl notebook ###
# v0.11.14

using Markdown
using InteractiveUtils

# ╔═╡ 383dd75c-060e-11eb-1fd1-85f6eb33a9af
using DataFrames, DataFramesMeta, CSV

# ╔═╡ a60d9102-060a-11eb-1c04-fdb0ce5006ac
md"# 编写模型各功能的测试notebook"

# ╔═╡ ec09183e-060a-11eb-2690-c9aa2c7e2a31
md"## DataImporter模块"

# ╔═╡ 49632e92-060e-11eb-05a8-9b0fccf669bd
# Init Class system

"""
    ExpEnv

The **experiment environment** which the learner will to learn.
"""
struct ExpEnv
    stim_color::Array{Int64,1}
    stim_loc::Array{Int64,1}
    stim_correct_action::Array{Int64,1}
    stim_action_congruency::Array{Int64,1}
    env_type::Array{String,1}
    sub_tag::Array{String,1}
end

# ╔═╡ 4963a7fa-060e-11eb-31b5-65118c468abc
"""
    RealSub

All of the actions the **real subject** have done.
"""
struct RealSub
    response::Array{Int64,1}
    RT::Array{Float64,1}
    corrections::Array{Int64,1}
    sub_tag::Array{String,1}
end

# ╔═╡ 496dc474-060e-11eb-179c-67211ab71736
foo1 = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/unit_test/sub01_Yangmiao_s.csv")

# ╔═╡ 4964815c-060e-11eb-278e-618c1f91b32e
"""
    transform_data!(raw_data, transform_rule)

Convert the string tag into binary format.

# Examples
```julia
# Define the trasnform rule
begin
	code_rule = Dict("red" => "0" , "green" => "1")
	contigency = Dict("con" => "1", "inc" => "0")
	Type_rule = Dict("hit" => "1", "incorrect" => "0")
	location = Dict("left" => "0", "right" => "1")
	transform_rule = Dict("color" => code_rule, "Type" => Type_rule, 
		                  "location" => location, "contigency" => contigency)
end
# Excute the transform
transform_data!(experiment_data, transform_rule)
```
"""
function transform_data!(raw_data::DataFrame, transform_rule::Dict)
	for rules in transform_rule
        colname = rules.first
		replace_rules = rules.second
		for replace_pair in replace_rules
			replace!(raw_data[!, colname], replace_pair)
		end
		
		if !isa(raw_data[!, colname],Array{Int64})
			raw_data[!, colname] = parse.(Int,raw_data[!, colname])
		end
	end
end

# ╔═╡ f5b413a8-0642-11eb-3674-4d229350dba2
transformed_data = foo1

# ╔═╡ 8360a24a-062d-11eb-2ae2-891d1573864c
begin
    env_idx_dict = Dict("stim_color" => "color", "stim_loc" => "location", 
		                "stim_action_congruency" => "contigency", 
		                "env_type" => "condition", "sub_tag" => "Subject")
	sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
		                "corrections" => "Type", "sub_tag" => "Subject")
	task_rule = Dict(0 => 0, 1 => 1)
end

# ╔═╡ 2a07bfd8-0639-11eb-0250-ddfd340341f7
"""
    transform_sub!(transformed_data::DataFrame, env_idx_dict::Dict, 
		                  sub_idx_dict::Dict, task_rule::Dict)

Init the env and subject objects for simulation.

# Examples
```julia
# Define the trasnform rule
begin
    env_idx_dict = Dict("stim_color" => "color", "stim_loc" => "location", 
		                "stim_action_congruency" => "contigency", 
		                "env_type" => "condition", "sub_tag" => "Subject")
	sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
		                "corrections" => "Type", "sub_tag" => "Subject")
	task_rule = Dict(0 => 0, 1 => 1)
end
# Excute the transform
env, sub = init_env_realsub(transformed_data, env_idx_dict, sub_idx_dict, task_rule)
```
"""
function init_env_sub(transformed_data::DataFrame, env_idx_dict::Dict, 
		                  sub_idx_dict::Dict, task_rule::Dict)
	# Generate right reaction
	begin
		right_action = transformed_data[!, env_idx_dict["stim_color"]]
		for rule in task_rule
			replace!(right_action, rule)
		end
	end
	
	exp_env = ExpEnv(transformed_data[!, env_idx_dict["stim_color"]],
		             transformed_data[!, env_idx_dict["stim_loc"]],
		             right_action, 
		             transformed_data[!, env_idx_dict["stim_action_congruency"]],
		             transformed_data[!, env_idx_dict["env_type"]],
		             transformed_data[!, env_idx_dict["sub_tag"]])
	real_sub = RealSub(transformed_data[!, sub_idx_dict["response"]],
		               transformed_data[!, sub_idx_dict["RT"]],
		               transformed_data[!, sub_idx_dict["corrections"]],
		               transformed_data[!, sub_idx_dict["sub_tag"]])
	println("The env and sub info of " * transformed_data[!, env_idx_dict["sub_tag"]][1] * " is generated")
	
	return (exp_env, real_sub)
end	

# ╔═╡ f0f5125a-063c-11eb-2062-79ff1d203126
env, sub = init_env_sub(transformed_data, env_idx_dict, sub_idx_dict, task_rule)

# ╔═╡ 30a9fb66-0644-11eb-08d7-310a38336154
#导入一个被试的数据开始分析
all_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_sub_data_with_Guo_model_add_pe.csv")

# ╔═╡ 0f7c7282-060f-11eb-0bc0-6f2d4aeedf3a
transform_data!(foo1, transform_rule)

# ╔═╡ a4f8b8c6-0715-11eb-3453-9d4442e54113


# ╔═╡ 91240684-060e-11eb-3e4a-d1c1878d8c3a
# Define the trasnform rule
begin
	code_rule = Dict("red" => "0" , "green" => "1")
	contigency = Dict("con" => "1", "inc" => "0")
	Type_rule = Dict("hit" => "1", "incorrect" => "0")
	location = Dict("left" => "0", "right" => "1")
	transform_rule = Dict("color" => code_rule, "Type" => Type_rule, 
		                  "location" => location, "contigency" => contigency)
end

# ╔═╡ ab4f8e22-0715-11eb-3e7f-d5300a6f5d3c
# Define the trasnform rule
begin
	code_rule = Dict("red" => "0" , "green" => "1")
	contigency = Dict("con" => "1", "inc" => "0")
	Type_rule = Dict("hit" => "1", "incorrect" => "0")
	location = Dict("left" => "0", "right" => "1")
	transform_rule = Dict("color" => code_rule, "Type" => Type_rule, 
		                  "location" => location, "contigency" => contigency)
end

# ╔═╡ Cell order:
# ╠═a60d9102-060a-11eb-1c04-fdb0ce5006ac
# ╠═ec09183e-060a-11eb-2690-c9aa2c7e2a31
# ╠═383dd75c-060e-11eb-1fd1-85f6eb33a9af
# ╠═49632e92-060e-11eb-05a8-9b0fccf669bd
# ╠═4963a7fa-060e-11eb-31b5-65118c468abc
# ╠═496dc474-060e-11eb-179c-67211ab71736
# ╠═4964815c-060e-11eb-278e-618c1f91b32e
# ╠═91240684-060e-11eb-3e4a-d1c1878d8c3a
# ╠═0f7c7282-060f-11eb-0bc0-6f2d4aeedf3a
# ╠═f5b413a8-0642-11eb-3674-4d229350dba2
# ╠═8360a24a-062d-11eb-2ae2-891d1573864c
# ╠═2a07bfd8-0639-11eb-0250-ddfd340341f7
# ╠═f0f5125a-063c-11eb-2062-79ff1d203126
# ╠═30a9fb66-0644-11eb-08d7-310a38336154
# ╠═ab4f8e22-0715-11eb-3e7f-d5300a6f5d3c
# ╠═a4f8b8c6-0715-11eb-3453-9d4442e54113
