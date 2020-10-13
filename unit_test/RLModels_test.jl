using Test
import CSV

include("../models/DataImporter.jl")
include("../models/RLModels.jl")

# 测试函数计算的基本要素
@testset "Basic elements of Calculations" begin
    
    # 冲突水平的计算
    test1 = [0.6;0.4;0.7;0.3]
    @test calc_CCC(test1, (0,0)) ≈ 0.2

    # 价值矩阵更新
    weight_matrix = zeros(Float64, (2, 4))
    weight_matrix[1,:] = [0.5,0.5,0.5,0.5]
    weight_matrix[2,:] = update_options_weight_matrix(weight_matrix[1,:] , 0.5, 0.9, (0,0))
    @test weight_matrix[2,:] == [0.75, 0.5, 0.5, 0.5]
    
    weight_matrix = zeros(Float64, (2, 4))
    weight_matrix[1,:] = [0.5,0.5,0.6,0.6]
    weight_matrix[2,:] = update_options_weight_matrix(weight_matrix[1,:] , 0.5, 0.9, (0,0))
    print(weight_matrix)
    @test weight_matrix[2,:] == [0.75, 0.5, 0.51, 0.51]
end

weight_matrix = zeros(Float64, (2, 4))
weight_matrix[1,:] = [0.5,0.5,0.5,0.5]
weight_matrix[1,:]
x = [1,2,3];
y = [2,3,4];

result = update_options_weight_matrix(weight_matrix[1,:], 0.5, 0.9, (0,0)) # = [0.75, 0.5, 0.45, 0.45]
convert(Array, result) == [0.75, 0.5, 0.5, 0.5]
result2 = convert(Array, result)
typeof(convert(Array, result))

evaluate_relation(x,y)