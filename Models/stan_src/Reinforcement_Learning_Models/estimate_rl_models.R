###
### Prepartion
###

## Import packages and check the runtime environment

library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")

# Estimate each sub
sub_num_list <- unique(raw_data$Subject_num)

## Import the stan model
rl_ab_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Models/stan_src/Reinforcement_Learning_Models/rl_ab.stan"
rl_sr_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Models/stan_src/Reinforcement_Learning_Models/rl_sr.stan"
rl_sr_volatile_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Models/stan_src/Reinforcement_Learning_Models/rl_sr_volatile.stan"

rl_ab_model <- cmdstan_model(rl_ab_stanfile)
rl_sr_model <- cmdstan_model(rl_sr_stanfile)
rl_sr_volatile_model <- cmdstan_model(rl_sr_volatile_stanfile)

###
### Estimate Models
###

## rl_ab
for (i in sub_num_list) {
    single_sub_table <- filter(raw_data, Subject_num == i)

    N <- nrow(single_sub_table)
    consistency <- as.numeric(str_replace(str_replace(single_sub_table$congruency, 'con', '1'), 'inc', '0'))
    data_list <- list(N = N, consistency = consistency)

    fit <- rl_ab_model$sample(
        data = data_list,
        chains = 4,
        refresh = 500,
        output_dir = file_save_path_1k1v)
}
