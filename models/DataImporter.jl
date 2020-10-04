#=
Data importer for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

导入研究数据并初始化学习环境
尝试使用UTF8命名函数和类型系统, 便于阅读者理解(划掉, 便于装逼)
=#

using DataFrames, DataFramesMeta
import CSV

# Init Class system

"""
    ExpEnv

The **experiment environment** which the learner will to learn.
"""
struct ExpEnv
    stim_color::Array{Int}
    stim_loc::Array{Int}
    stim_correct_action::Array{Int}
    stim_action_congruency::Array{Float}
    subtag::String
    envtype::Array{String}
end

"""
    RealSub

All of the actions the **real subject** have done.
"""
struct RealSub
    respons::Array{Int}
    RT::Array{Float}
    corrections::Array{Int}
    subtag::String
end

# Define functions
"""
    transform_data!(raw_data, transform_rule)

Convert the string tag into binary format.

# Examples
```julia
# Define the trasnform rule
begin
    code_rule = Dict("con" => "1", "inc" => "0")
    Type_rule = Dict("hit" => "1", "incorrect" => "0")
    transform_rule = Dict("contigency" => code_rule, "Type" => Type_rule)
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

function init_expenv_realsub(transformed_data)
    
end