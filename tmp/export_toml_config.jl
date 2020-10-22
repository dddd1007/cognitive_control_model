using TOML

color_rule = Dict("red" => "0", "green" => "1")
congruency_rule = Dict("con" => "1", "inc" => "0")
Type_rule = Dict("hit" => "1", "incorrect" => "0", "miss" => "0")
loc_rule = Dict("left" => "0", "right" => "1")
dummy_rules = Dict("stim_color" => color_rule, "Type" => Type_rule,
                      "stim_loc" => loc_rule, "congruency" => congruency_rule)
TOML.print(dummy_rules)

env_idx_dict = Dict("stim_task_related" => "stim_color", 
                        "stim_task_unrelated" => "stim_loc", 
                        "stim_action_congruency" => "congruency", 
                        "correct_action" => "correct_action",
                        "env_type" => "condition", "sub_tag" => "Subject")
sub_idx_dict = Dict("response" => "Response", "RT" => "RT", 
                        "corrections" => "Type", "sub_tag" => "Subject")
env_sub_idx = Dict("env_idx_dict" => env_idx_dict, "sub_idx_dict" => sub_idx_dict)
TOML.print(env_sub_idx)