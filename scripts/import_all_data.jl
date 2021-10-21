using DataFramesMeta, DataFrames, GLM, StatsBase, Models
using CSV: CSV
using cognitive_control_model: cognitive_control_model

# import all data
all_data = CSV.read(joinpath(dirname(pathof(cognitive_control_model)), "..", "data",
                             "input", "pure_all_data.csv"), DataFrame);
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
DataManipulate.transform_data!(all_data, transform_rule)
begin
    env_idx_dict = Dict("stim_task_related" => "stim_color",
                        "stim_task_unrelated" => "stim_loc",
                        "stim_action_congruency" => "congruency",
                        "correct_action" => "correct_action", "env_type" => "condition",
                        "sub_tag" => "Subject")
    sub_idx_dict = Dict("response" => "Response", "RT" => "RT", "corrections" => "Type",
                        "sub_tag" => "Subject", "prop_seq" => "prop")
end
