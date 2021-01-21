# 认知控制的相关模型

本文件夹下的文件按功能不同, 通过 Module 隔离各个函数

- DataManipulate.jl 数据导入, 变量哑元化转换
- RLModels_basic.jl 强化学习模型的基本类型定义, 基本的计算函数
    - RLModels_SoftMax.jl 拟合过程中存在显式 SoftMax 决策过程的强化学习模型
    - RLModels_no_SoftMax.jl 拟合过程当中仅将决策结果的 学习后 Value 提取出的强化学习模型
- Bayesian_basic.jl 贝叶斯模型的基本类型定义, 与基本的计算函数
    - 