using DataFramesMeta, DataFrames, GLM, StatsBase, Models
using CSV: CSV
using cognitive_control_model: cognitive_control_model
# set the generated file locations
csvpath = joinpath(dirname(pathof(cognitive_control_model)), "..", "data", "output",
                   "RLModels", "AB")

# import all data
all_data = CSV.read(joinpath(dirname(pathof(cognitive_control_model)), "..", "data",
                             "input", "pure_all_data.csv"));
all_data = @where(all_data, :Response .!= "NA")

# Remove the subject who always move the head
sub27 = @where(all_data, :Subject_num .== 27)

begin
    color_rule = Dict("red" => "0", "green" => "1")
    congruency_rule = Dict("con" => "1", "inc" => "0")
    Type_rule = Dict("hit" => "1", "incorrect" => "0", "miss" => "0")
    loc_rule = Dict("left" => "0", "right" => "1")
    transform_rule = Dict("stim_color" => color_rule, "Type" => Type_rule,
                          "stim_loc" => loc_rule, "congruency" => congruency_rule)
end
Models.DataManipulate.transform_data!(all_data, transform_rule)
begin
    env_idx_dict = Dict("stim_task_related" => "stim_color",
                        "stim_task_unrelated" => "stim_loc",
                        "stim_action_congruency" => "congruency",
                        "correct_action" => "correct_action", "env_type" => "condition",
                        "sub_tag" => "Subject")
    sub_idx_dict = Dict("response" => "Response", "RT" => "RT", "corrections" => "Type",
                        "sub_tag" => "Subject")
end

# For analysis each subject
#Threads.@threads 
for sub_num in 1:36
    #sub_num = 11
    println("========= Begin Sub " * repr(sub_num) * " ==========")

    if sub_num == 27 || sub_num == 6
        continue
    end

    params_basic = zeros(1, 4)
    params_error = zeros(1, 6)
    params_CCC = zeros(1, 9)
    params_basic = zeros(1, 4)
    params_error = zeros(1, 6)
    params_CCC = zeros(1, 9)

    each_sub_data = @where(all_data, :Subject_num .== sub_num)
    each_env, each_subinfo = Models.RLModels.init_env_sub(each_sub_data, env_idx_dict,
                                                          sub_idx_dict)

    # basic model
    println("= Begin basic model of " * repr(sub_num) * " =")
    optim_param, eval_result, verbose_table = Models.Optim.RL_NoSoftMax_basic_AB(each_env,
                                                                                 each_subinfo,
                                                                                 10000)

    params_basic[1:3] .= optim_param
    params_basic[4] = eval_result
    params_basic_table = DataFrame(params_basic, [:α_v, :α_s, :decay, :MSE])
    CSV.write(csvpath * "/sub_" * repr(sub_num) * "_params_basic.csv", params_basic_table)
    CSV.write(csvpath * "/sub_" * repr(sub_num) * "_params_basic_verbose.csv",
              verbose_table)

    println("! End basic model of " * repr(sub_num) * " =")

    # error model
    println("= Begin error model of " * repr(sub_num) * " =")
    optim_param, eval_result, verbose_table = Models.Optim.RL_NoSoftMax_witherror_AB(each_env,
                                                                                     each_subinfo,
                                                                                     10000)

    params_error[1:5] .= optim_param
    params_error[6] = eval_result
    params_error_table = DataFrame(params_error,
                                   [:α_v, :α_s, :α_v_error, :α_s_error, :decay, :MSE])
    CSV.write(csvpath * "/sub_" * repr(sub_num) * "_params_error.csv", params_error_table)
    CSV.write(csvpath * "/sub_" * repr(sub_num) * "_params_error_verbose.csv",
              verbose_table)

    println("! End error model of " * repr(sub_num) * " =")

    # CCC model
    println("= Begin CCC model of " * repr(sub_num) * " =")
    optim_param, eval_result, verbose_table = Models.Optim.RL_NoSoftMax_withCCC_AB(each_env,
                                                                                   each_subinfo,
                                                                                   10000)

    params_CCC[1:8] .= optim_param
    params_CCC[9] = eval_result
    params_CCC_table = DataFrame(params_CCC,
                                 [:α_v, :α_s, :α_v_error, :α_s_error, :α_v_CCC, :α_s_CCC,
                                  :CCC, :decay, :MSE])
    CSV.write(csvpath * "/sub_" * repr(sub_num) * "_params_CCC.csv", params_CCC_table)
    CSV.write(csvpath * "/sub_" * repr(sub_num) * "_params_CCC_verbose.csv", verbose_table)

    println("! End CCC model of " * repr(sub_num) * " =")
end