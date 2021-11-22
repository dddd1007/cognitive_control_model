library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

sr_1k1v_with_RT_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr_1k1v_estimate_RT_by_trial.stan"

sr_1k1v_with_RT_learner <- cmdstan_model(sr_1k1v_with_RT_stanfile)

output_dir <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/estimate_with_RT/"

#####
##### Estimate Model
#####

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")

# Estimate each sub
sub_num_list <- unique(raw_data$Subject_num)
sub_num_list <- sub_num_list[-1]

for (i in sub_num_list) {
    print(paste0("Estimating model for subject ", i))
    single_sub_table <- filter(raw_data, Subject_num == i)

    N <- nrow(single_sub_table)
    stim_space_loc <- single_sub_table$stim_loc_num
    corr_reaction <- single_sub_table$correct_action
    real_reaction <- single_sub_table$Response
    RT <- single_sub_table$RT
    data_list <- list(N = N, 
                        stim_space_loc = stim_space_loc, 
                        corr_reaction = corr_reaction, 
                        real_reaction = real_reaction, 
                        RT = RT)

    file_save_path <- paste0(output_dir, "sub_", as.character(i))
    dir.create(file_save_path)
    fit1 <- sr_1k1v_with_RT_learner$sample(
        data = data_list,
        chains = 4,
        parallel_chains = 4,
        refresh = 100,
        save_warmup = 0,
        output_dir = file_save_path
    )
}