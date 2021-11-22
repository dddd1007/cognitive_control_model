library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

sr_estimate_RT_as_obj_func_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr_estimate_RT_as_obj_func.stan"

sr_estimate_RT_as_obj_func_learner <- cmdstan_model(sr_estimate_RT_as_obj_func_stanfile)

output_dir <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/SR_RT_as_obj_func/"

##### helper func
generate_keep_seq <- function(input_dataframe, filter_type = "no_error"){
    keep_index <- seq(1, nrow(input_dataframe))
    foo_dataframe <- cbind(input_dataframe, keep_index)
    result_dataframe <- filter(foo_dataframe, Type == "hit")
    if (filter_type == "no outlier"){
        result_dataframe <- result_dataframe %>%
                        group_by(congruency, prop, condition) %>%
                        filter(abs(RT - mean(RT)) < (sd(RT) * 3))
    }
    keep_seq <- result_dataframe$keep_index
    return(keep_seq)
}
#####
##### Estimate Model
#####

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")

# Estimate each sub
sub_num_list <- unique(raw_data$Subject_num)
# sub_num_list <- sub_num_list[-1]

# for (i in sub_num_list) {
    i = 1
    print(paste0("Estimating model for subject ", i))
    single_sub_table <- filter(raw_data, Subject_num == i)
    
    N <- nrow(single_sub_table)
    stim_space_loc <- single_sub_table$stim_loc_num
    corr_reaction <- single_sub_table$correct_action
    real_reaction <- single_sub_table$Response
    RT <- single_sub_table$RT
    keep_seq <- generate_keep_seq(single_sub_table, filter_type = "no_error")

    data_list <- list(N = N,
                        stim_space_loc = stim_space_loc,
                        corr_reaction = corr_reaction,
                        real_reaction = real_reaction,
                        RT = RT,
                        keep_seq = keep_seq,
                        keep_seq_len = length(keep_seq))

    file_save_path <- paste0(output_dir, "sub_", as.character(i))
    dir.create(file_save_path)
    fit <- sr_estimate_RT_as_obj_func_learner$sample(
        data = data_list,
        chains = 4,
        parallel_chains = 4,
        refresh = 100,
        save_warmup = 0,
        output_dir = file_save_path
    )
# }