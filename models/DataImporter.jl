#=
Data importer for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

导入研究数据并初始化学习环境
尝试使用UTF8命名函数和类型系统, 便于阅读者理解(划掉, 便于装逼)
=#

import DataFrames, DataFramesMeta, CSV

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

# Define functions

function transform_data(raw_data, translation_rule::Dict)
    
end

function init_expenv_realsub(transformed_data)
    
end

foo = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/unit_test/sub01_Yangmiao_s.csv")
