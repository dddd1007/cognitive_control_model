using Test
import CSV

include("../models/DataImporter.jl")
include("../models/RLModels.jl")
## 测试读取文件

test1 = [0.6;0.4;0.7;0.3]
@test calc_CCC(test1, (0,0)) ≈ 0.2

x = [1,2,3];
y = [2,3,4];

evaluate_relation(x,y)