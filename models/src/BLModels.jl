#=
# Bayesian Learner for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-10-28

This model is based on Tim Behrens' bayesian learner model, which try to explain 
the mechanism of parameter estimation in human's brain.
=#

using Distributions

normal_dis = Normal()

x = [3,6,9,12,15]
y = [4,7,10,13,16]

import GLM, DataFrames
test_data = DataFrames.DataFramex =(x, y)
GLM.lm(y~x, test_data)