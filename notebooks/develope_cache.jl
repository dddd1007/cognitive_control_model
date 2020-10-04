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
    stim_color::Array{Int64}
    stim_loc::Array{Int64}
    stim_correct_action::Array{Int64}
    stim_action_congruency::Array{Float64,1}
    subtag::String
    envtype::Array{String}
end

# ╔═╡ 4963a7fa-060e-11eb-31b5-65118c468abc
"""
    RealSub

All of the actions the **real subject** have done.
"""
struct RealSub
    respons::Array{Int64}
    RT::Array{Float64}
    corrections::Array{Int64}
    subtag::String
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
    code_rule = Dict("con" => "1", "inc" => "0")
    Type_rule = Dict("hit" => "1", "incorrect" => "0")
    transform_rule = Dict("contigency" => code_rule, 
                          "Type" => Type_rule)
end

# Excute the transform
transform_data!(experiment_data, transform_rule)
```
"""
function transform_data!(raw_data, transform_rule)
	for rules in transform_rule
        colname = rules.first
		replace_rules = rules.second
		for replace_pair in replace_rules
			replace!(raw_data[!, colname], replace_pair)
		end
		raw_data[!, colname] = parse.(Int,raw_data[!, colname])
	end
end

# ╔═╡ 91240684-060e-11eb-3e4a-d1c1878d8c3a
# Define the trasnform rule
begin
	code_rule = Dict("con" => "1", "inc" => "0")
	Type_rule = Dict("hit" => "1", "incorrect" => "0")
	transform_rule = Dict("contigency" => code_rule, "Type" => Type_rule)
end

# ╔═╡ 0f7c7282-060f-11eb-0bc0-6f2d4aeedf3a
transform_data!(foo1, transform_rule)

# ╔═╡ 44c6ff22-061c-11eb-0f55-fb95273336b5
begin
	x = [1,2,3]
	y = [2,3,4]
	function f(x, y, a)
		for i in x
			for z in y
				eval(:(println(a)))
			end
		end
	end
	f(x, y, "a")
end

# ╔═╡ 5e62a886-061d-11eb-1dc6-6368bd08a0d5
foo1

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
# ╠═44c6ff22-061c-11eb-0f55-fb95273336b5
# ╠═5e62a886-061d-11eb-1dc6-6368bd08a0d5
