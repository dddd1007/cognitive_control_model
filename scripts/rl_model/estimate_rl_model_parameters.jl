include("/Users/dddd1007/project2git/cognitive_control_model/Models/rl_model_estimate_by_stim/func_estimate_rl_model.jl")

using CSV, DataFramesMeta

# 导入数据

all_sub_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv",
                        DataFrame)
sub_num_list = unique(all_sub_data[!, "Subject_num"]);

# 拟合 rl_ab 模型
α_grid = collect(0.01:0.01:1.0);
all_sub_param_likelihoood = DataFrame(α = [], likelihood = [], aic = [], sub = [])

for i in collect(1:length(sub_num_list))
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])
    stim_feature_seq = single_sub_data[!, :congruency_num];
    for α in α_grid
        data = rl_ab_data(α, stim_feature_seq)
        likelihood = calc_rl_fit_goodness(data)
        aic = calc_rl_fit_goodness(data, fit_idx = "aic")
        push!(all_sub_param_likelihoood, (α, goodness, aic, sub_num_list[i]))
    end
end

CSV.write(all_sub_param_likelihoood, "/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_param_goodness.csv")

# 拟合 rl_ab_volatility 模型
α_s_grid = collect(0.01:0.01:1.0);
α_v_grid = collect(0.01:0.01:1.0);
all_sub_param_likelihoood_volatility = DataFrame(α_s = [], α_v = [], likelihood = [], sub = [])
