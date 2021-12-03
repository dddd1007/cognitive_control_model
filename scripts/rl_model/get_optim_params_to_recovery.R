library(tidyverse)
# import data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv")

# ab model
params_set_result <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_param_goodness.csv")
params_set_result %>%
    group_by(sub) %>%
    summarise(max_likelihood = max(likelihood)) -> a
# ab_v model
params_set_result_v <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_ab_volatility_param_goodness.csv")
params_set_result_v %>%
    group_by(sub) %>%
    summarise(max_likelihood = max(likelihood)) -> b
# sr model
params_set_result <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_param_goodness.csv")
params_set_result %>%
    group_by(sub) %>%
    summarise(max_likelihood = max(likelihood)) -> c
# sr_v model
params_set_result_v <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model_estimate_by_stim/rl_sr_volatility_param_goodness.csv")
params_set_result_v %>%
    group_by(sub) %>%
    summarise(max_likelihood = max(likelihood)) -> d
