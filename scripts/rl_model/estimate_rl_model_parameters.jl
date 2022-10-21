include("/Users/dddd1007/project2git/cognitive_control_model/Models/rl_model_estimate_by_stim/func_estimate_rl_model.jl")

using CSV, DataFramesMeta

# 导入数据

all_sub_data = CSV.read("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv",
                        DataFrame)
sub_num_list = unique(all_sub_data[!, "Subject_num"]);


#################### Model Estimation ####################
# 拟合 rl_ab 模型
α_grid = collect(0.01:0.01:1.0);
all_sub_param_likelihoood = DataFrame(; α=[], likelihood=[], aic=[], sub=[])

for i in collect(1:length(sub_num_list))
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])
    stim_feature_seq = single_sub_data[!, :congruency_num]
    for α in α_grid
        data = rl_ab_data(α, stim_feature_seq)
        likelihood = calc_rl_fit_goodness(data)
        aic = calc_rl_fit_goodness(data; fit_idx="aic")
        push!(all_sub_param_likelihoood, (α, likelihood, aic, sub_num_list[i]))
    end
end
optim_params_set = DataFrame([g[findmax(g.likelihood)[2], :]
                              for g in groupby(all_sub_param_likelihoood, :sub)])

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_param_goodness.csv",
          all_sub_param_likelihoood)
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_param_set.csv",
          optim_params_set)

for i in collect(1:length(sub_num_list))
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])
    stim_feature_seq = single_sub_data[!, :congruency_num]
    exp_volatility_seq = single_sub_data[!, :volatility_num]
    for α_s in α_s_grid
        for α_v in α_v_grid
            data = rl_ab_volatility_data(α_s, α_v, stim_feature_seq, exp_volatility_seq)
            likelihood = calc_rl_fit_goodness(data)
            aic = calc_rl_fit_goodness(data; fit_idx="aic")
            push!(all_sub_param_likelihoood_volatility,
                  (α_s, α_v, likelihood, aic, sub_num_list[i]))
        end
    end
end
optim_params_set = DataFrame([g[findmax(g.likelihood)[2], :]
                              for g in groupby(all_sub_param_likelihoood_volatility, :sub)])

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_volatility_param_goodness.csv",
          all_sub_param_likelihoood_volatility)
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_volatility_param_set.csv",
          optim_params_set)

## 拟合 rl_sr 模型
α_grid = collect(0.01:0.01:1.0);
all_sub_param_likelihoood = DataFrame(; α=[], likelihood=[], aic=[], sub=[])

for i in collect(1:length(sub_num_list))
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])
    stim_loc_seq = single_sub_data[!, :stim_loc_num]
    reaction_loc_seq = single_sub_data[!, :correct_action]
    for α in α_grid
        data = rl_sr_data(α, stim_loc_seq, reaction_loc_seq)
        likelihood = calc_rl_fit_goodness(data)
        aic = calc_rl_fit_goodness(data; fit_idx="aic")
        push!(all_sub_param_likelihoood, (α, likelihood, aic, sub_num_list[i]))
    end
end
optim_params_set = DataFrame([g[findmax(g.likelihood)[2], :]
                              for g in groupby(all_sub_param_likelihoood, :sub)])

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_param_goodness.csv",
          all_sub_param_likelihoood)
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_param_set.csv",
          optim_params_set)

## 拟合 rl_sr_volatility 模型
α_s_grid = collect(0.01:0.01:0.5);
α_v_grid = collect(0.01:0.01:0.5);
all_sub_param_likelihoood_volatility = DataFrame(; α_s=[], α_v=[], likelihood=[], aic=[],
                                                 sub=[])

for i in collect(1:length(sub_num_list))
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])
    stim_loc_seq = single_sub_data[!, :stim_loc_num]
    reaction_loc_seq = single_sub_data[!, :correct_action]
    exp_volatility_seq = single_sub_data[!, :volatility_num]
    for α_s in α_s_grid
        for α_v in α_v_grid
            data = rl_sr_volatility_data(α_s, α_v, stim_loc_seq, reaction_loc_seq,
                                         exp_volatility_seq)
            likelihood = calc_rl_fit_goodness(data)
            aic = calc_rl_fit_goodness(data; fit_idx="aic")
            push!(all_sub_param_likelihoood_volatility,
                  (α_s, α_v, likelihood, aic, sub_num_list[i]))
        end
    end
end
optim_params_set = DataFrame([g[findmax(g.likelihood)[2], :]
                              for g in groupby(all_sub_param_likelihoood_volatility, :sub)])

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_volatility_param_goodness.csv",
          all_sub_param_likelihoood_volatility)
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_volatility_param_set.csv",
          optim_params_set)

## 拟合 rl_sr_sep_alpha 模型
α_l_grid = collect(0.01:0.01:1.0);
α_r_grid = collect(0.01:0.01:1.0);
all_sub_param_likelihoood = DataFrame(; α_l=[], α_r=[], likelihood=[], aic=[], sub=[])

for i in collect(1:length(sub_num_list))
    # i = 1
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])
    stim_loc_seq = single_sub_data[!, :stim_loc_num]
    reaction_loc_seq = single_sub_data[!, :correct_action]
    for α_l in α_l_grid
        for α_r in α_r_grid
            # α_l = 0.5
            # α_r = 0.5
            data = rl_sr_sep_alpha_data(α_l, α_r, stim_loc_seq, reaction_loc_seq)
            likelihood_result = calc_rl_fit_goodness(data)
            aic_result = calc_rl_fit_goodness(data; fit_idx="aic")
            push!(all_sub_param_likelihoood,
                  (α_l, α_r, likelihood_result, aic_result, sub_num_list[i]))
        end
    end
end
optim_params_set = DataFrame([g[findmax(g.likelihood)[2], :]
                              for g in groupby(all_sub_param_likelihoood, :sub)])

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_param_goodness.csv",
          all_sub_param_likelihoood)
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_param_set.csv",
          optim_params_set)

## 拟合 rl_sr_sep_alpha_volatility 模型
all_sub_param_likelihoood_volatility = DataFrame(; α_s_l=[], α_s_r=[], α_v_l=[], α_v_r=[],
                                                 likelihood=[], aic=[], mse=[], sub=[])
all_params_combine = collect(Iterators.product(0.1:0.01:0.4, 0.1:0.01:0.4, 0.1:0.01:0.4,
                                               0.1:0.01:0.4));


for i in collect(1:length(sub_num_list))
    println("===== Starting estimate sub" * string(sub_num_list[i]) * " =====")
    single_sub_data = @subset(all_sub_data, :Subject_num .== sub_num_list[i])
    stim_loc_seq = single_sub_data[!, :stim_loc_num]
    reaction_loc_seq = single_sub_data[!, :correct_action]
    exp_volatility_seq = single_sub_data[!, :volatility_num]

    # init data saving vectors
    α_s_l_list = Vector{Float64}(undef, length(all_params_combine))
    α_s_r_list = Vector{Float64}(undef, length(all_params_combine))
    α_v_l_list = Vector{Float64}(undef, length(all_params_combine))
    α_v_r_list = Vector{Float64}(undef, length(all_params_combine))
    likelihood_list = Vector{Float64}(undef, length(all_params_combine))
    mse_list = Vector{Float64}(undef, length(all_params_combine))
    aic_list = Vector{Float64}(undef, length(all_params_combine))
    sub_list = Vector{Float64}(undef, length(all_params_combine))

    Threads.@threads for j in collect(1:length(all_params_combine))
        (α_s_l, α_s_r, α_v_l, α_v_r) = all_params_combine[j]
        data = rl_sr_sep_alpha_volatility_data(α_s_l, α_s_r, α_v_l, α_v_r, stim_loc_seq,
                                               reaction_loc_seq, exp_volatility_seq)
        likelihood = calc_rl_fit_goodness(data)
        aic = calc_rl_fit_goodness(data; fit_idx="aic")
        mse = calc_rl_fit_goodness(data; fit_idx="mse")

        α_s_l_list[j] = α_s_l
        α_s_r_list[j] = α_s_r
        α_v_l_list[j] = α_v_l
        α_v_r_list[j] = α_v_r
        likelihood_list[j] = likelihood
        mse_list[j] = mse
        aic_list[j] = aic
        sub_list[j] = sub_num_list[i]
    end
    single_sub_param_likelihoood_volatility = DataFrame(; α_s_l=α_s_l_list, α_s_r=α_s_r_list,
                                                        α_v_l=α_v_l_list, α_v_r=α_v_r_list,
                                                        likelihood=likelihood_list, aic=aic_list,
                                                        mse=mse_list, sub=sub_list)
    all_sub_param_likelihoood_volatility = vcat(all_sub_param_likelihoood_volatility, single_sub_param_likelihoood_volatility)
    println("===== Finished estimate sub" * string(sub_num_list[i]) * " =====")
end

optim_params_set = DataFrame([g[findmax(g.likelihood)[2], :]
                              for g in groupby(all_sub_param_likelihoood_volatility, :sub)])

CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_volatility_param_goodness.csv",
          all_sub_param_likelihoood_volatility)
CSV.write("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_sep_alpha_volatility_param_set.csv",
          optim_params_set)
