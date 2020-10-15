# set the generated file locations
imgpath = "/home/dddd1007/project2git/cognitive_control_model/data/output/RLModels/img_rl_model_hyperopt/"
csvpath = "/home/dddd1007/project2git/cognitive_control_model/data/output/RLModels/"

import CSV
using DataFrames, DataFramesMeta, Plots

include("/home/dddd1007/project2git/cognitive_control_model/src/optim.jl")

# import all data
all_data = CSV.read("/home/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv");
all_data = @where(all_data, :Response .!= "NA")

# Remove the subject who always move the head
sub27 = @where(all_data, :Subject_num .== 27)

begin
    color_rule = Dict("red" => "0" , "green" => "1")
    congruency_rule = Dict("con" => "1", "inc" => "0")
    Type_rule = Dict("hit" => "1", "incorrect" => "0", "miss" => "0")
    loc_rule = Dict("left" => "0", "right" => "1")
    transform_rule = Dict("stim_color" => color_rule, "Type" => Type_rule, 
        "stim_loc" => loc_rule, "congruency" => congruency_rule)
end
transform_data!(all_data, transform_rule)
begin
    env_idx_dict = Dict("stim_task_related" => "stim_color", 
                        "stim_task_unrelated" => "stim_loc", 
                        "stim_action_congruency" => "congruency", 
                        "correct_action" => "correct_action",
                        "env_type" => "condition", "sub_tag" => "Subject")
    sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
                        "corrections" => "Type", "sub_tag" => "Subject")
end

params_basic = zeros(36, 4)
params_error = zeros(36, 6)
params_CCC = zeros(36, 9)

# For analysis each subject
for sub_num in 1:36
    println(sub_num)

    if sub_num == 27
        continue
    end

    each_sub_data = @where(all_data, :Subject_num .== sub_num);
    each_env, each_subinfo = init_env_sub(each_sub_data, env_idx_dict, sub_idx_dict);
    
    # basic model
    optim_param, eval_result, vis = hyperopt_rllearn_basic(each_env, each_subinfo, 100000)

    params_basic[sub_num, 1:3] .= optim_param
    params_basic[sub_num, 4]   = eval_result
    
    savefig(vis, imgpath * "basic/" * each_env.sub_tag[1] * ".png")

    # error model
    optim_param, eval_result, vis = hyperopt_rllearn_witherror(each_env, each_subinfo, 1000000)

    params_error[sub_num, 1:5] .= optim_param
    params_error[sub_num, 6]   = eval_result
    
    savefig(vis,imgpath * "error/" * each_env.sub_tag[1] * ".png")

    # CCC model
    optim_param, eval_result, vis = hyperopt_rllearn_withCCC(each_env, each_subinfo, 1000000)

    params_CCC[sub_num, 1:8] .= optim_param
    params_CCC[sub_num, 9]   = eval_result
    
    savefig(vis,imgpath * "CCC/" * each_env.sub_tag[1] * ".png")
end

params_basic_table = DataFrame(params_basic, [:α_v, :α_s, :decay, :MSE])
params_error_table = DataFrame(params_error, [:α_v, :α_s, :α_v_error, :α_s_error, :decay, :MSE])
params_CCC_table   = DataFrame(params_CCC,   [:α_v, :α_s, :α_v_error, :α_s_error, :α_v_CCC, :α_s_CCC, :CCC, :decay, :MSE])

CSV.write(csvpath * "params_basic.csv", params_basic_table)
CSV.write(csvpath * "params_error.csv", params_error_table)
CSV.write(csvpath * "params_CCC.csv",   params_CCC_table)