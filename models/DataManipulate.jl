#=
Data importer for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

导入研究数据并初始化学习环境
=#

#============================================================================ 
# Global: Define RLModels with Softmax                                      #
============================================================================#
module DataManipulate

using DataFrames, DataFramesMeta, GLM
import CSV
export evaluate_relation, init_env_sub, update_options_weight_matrix, transform_data!


#### 定义数据导入的函数
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

        if !isa(raw_data[!, colname], Array{Int64})
            raw_data[!, colname] = parse.(Int, raw_data[!, colname])
        end
    end
end

end