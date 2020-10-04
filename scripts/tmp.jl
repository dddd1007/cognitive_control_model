import CSV

include("../models/DataImporter.jl")

## 测试读取文件

file_path = "data/unit_test/sub01_Yangmiao_s.csv"
foo = CSV.read(file_path)
foo

## 测试转换规则
code_rule = Dict("con" => "1", "inc" => "0")
Type_rule = Dict("hit" => "1", "miss" => "0")
transform_rule = Dict("Code" => code_rule, "Type" => Type_rule)

transform_rule.keys

code_rule.keys