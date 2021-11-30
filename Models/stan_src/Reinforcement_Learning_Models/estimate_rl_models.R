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

# Set output directory
output_dir <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model/"
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
    ab_output_dir <- paste0(output_dir, "rl_ab/sub", i, "/")
    dir.create(ab_output_dir)

    N <- nrow(single_sub_table)
    consistency <- as.numeric(str_replace(str_replace(single_sub_table$congruency, 'con', '1'), 'inc', '0'))
    data_list <- list(N = N, consistency = consistency)

    fit <- rl_ab_model$sample(
        data = data_list,
        chains = 4,
        parallel_chains = 4,
        refresh = 100,
        save_warmup = 0,
        output_dir = ab_output_dir)
}

## rl_sr
for (i in sub_num_list) {
    single_sub_table <- filter(raw_data, Subject_num == i)
    sr_output_dir <- paste0(output_dir, "rl_sr/sub", i, "/")
    dir.create(sr_output_dir)

    N <- nrow(single_sub_table)
    stim_space_loc <- single_sub_table$stim_loc_num
    corr_react <- single_sub_table$correct_action
    data_list <- list(N = N, space_loc = stim_space_loc, corr_react = corr_react)

    fit <- rl_sr_model$sample(
        data = data_list,
        chains = 4,
        parallel_chains = 4,
        refresh = 100,
        save_warmup = 0,
        output_dir = sr_output_dir)
}

## rl_sr_volatile
for (i in sub_num_list) {
    single_sub_table <- filter(raw_data, Subject_num == i)
    sr_volatile_output_dir <- paste0(output_dir, "rl_sr_volatile/sub", i, "/")
    dir.create(sr_volatile_output_dir)
    
    N <- nrow(single_sub_table)
    stim_space_loc <- single_sub_table$stim_loc_num
    corr_react <- single_sub_table$correct_action
    volatility <- as.numeric(str_replace(str_replace(single_sub_table$condition, 'v', '1'), 's', '0'))
    data_list <- list(N = N, space_loc = stim_space_loc, corr_react = corr_react, volatility = volatility)

    fit <- rl_sr_volatile_model$sample(
        data = data_list,
        chains = 4,
        parallel_chains = 4,
        refresh = 100,
        save_warmup = 0,
        output_dir = sr_volatile_output_dir)
}
