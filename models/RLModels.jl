#=
Reinforcement Learning Models for Lingwang's lab

Author: Xiaokai Xia (xia@xiaokai.me)
Date: 2020-09-28
Version: 0.0.009

This model try to using reinforcement learning and Softmax to simulate the learning process in
Simon Tasks.

There are two models with four different methods:

## Two models

1. Reinforcement learning are used to learning the value of each rules
2. Softmax are used to select action by the value which RL learnt

## Four methods

- What to learn
    - Abstract concepts (Con, Inc)
    - S-R association
- The information attenuation
    - Have a Decay
    - Without Decay
- How to treat the error trials
    - Have a CCC to change action
    - Without CCC
- How to react to error trials
    - Change RL model's learning rate
    - Change Softmax model's Q-value
    - Both
=#

module RLModels

# 设置导入后的信息显示
