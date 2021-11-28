library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

ab_with_RT_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_ab_estimate_RT_by_trial.stan"

ab_with_RT_learner <- cmdstan_model(ab_with_RT_stanfile)

output_dir <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/ab_with_RT/"

#####
##### Estimate Model
#####

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")

# Estimate each sub
sub_num_list <- unique(raw_data$Subject_num)

for (i in sub_num_list) {
    print(paste0("Estimating model for subject ", i))
    single_sub_table <- filter(raw_data, Subject_num == i)

    N <- nrow(single_sub_table)
    y <- as.numeric(str_replace(str_replace(single_sub_table$congruency, 'con', '1'), 'inc', '0'))
    RT <- single_sub_table$RT
    data_list <- list(N = N,
                      y = y,
                      RT = RT)

    file_save_path <- paste0(output_dir, "sub_", as.character(i))
    dir.create(file_save_path)
    fit1 <- ab_with_RT_learner$sample(
        data = data_list,
        chains = 4,
        parallel_chains = 4,
        refresh = 100,
        save_warmup = 0,
        output_dir = file_save_path
    )
}