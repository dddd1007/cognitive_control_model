using Models, DataFrames, DataFramesMeta, CSV

raw_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv", DataFrame)
estimate_parameter = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/optim_model_wang/2a1d1ccc_wang.csv", DataFrame)

# for sub_num in 1:16
sub_num = 1
raw_sub_data = @where(raw_data, :Subject_num == sub_num)
sub_optim_params = @where(estimate_parameter, :sub_num == sub_num)
