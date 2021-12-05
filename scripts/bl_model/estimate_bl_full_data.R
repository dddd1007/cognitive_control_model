library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

bl_ab_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Models/stan_src/Bayesian_Models/bayesian_learner_ab.stan"
bl_sr_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Models/stan_src/Bayesian_Models/bayesian_learner_sr_1k1v.stan"

bl_ab_learner <- cmdstan_model(bl_ab_stanfile)
bl_sr_learner <- cmdstan_model(bl_sr_stanfile)

output_ab <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bl_estimate_by_full_data/ab/"
output_sr <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bl_estimate_by_full_data/sr/"


#####
##### Estimate Model
#####

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv")

# Estimate each sub
sub_num_list <- unique(raw_data$Subject_num)

for (i in sub_num_list) {
  print(paste0("========== Begin estimate sub", as.character(i), " =========="))
  single_sub_table <- filter(raw_data, Subject_num == i)

  # sr model
  N <- nrow(single_sub_table)
  corr_react <- single_sub_table$correct_action
  space_loc <- single_sub_table$stim_loc_num
  data_list <- list(N = N, corr_react = corr_react, space_loc = space_loc)

  file_save_path_sr <- paste0(output_sr, "sub_", as.character(i))
  dir.create(file_save_path_sr)
  fit <- bl_sr_learner$sample(
    data = data_list,
    chains = 4,
    parallel_chains = 4,
    refresh = 500,
    max_treedepth = 15,
    output_dir = file_save_path_sr
  )

  # ab model
  y <- single_sub_table$congruency_num
  data_list <- list(N = N, y = y)
  file_save_path_ab <- paste0(output_ab, "sub_", as.character(i))
  dir.create(file_save_path_ab)
  fit2 <- bl_ab_learner$sample(
    data = data_list,
    chains = 4,
    parallel_chains = 4,
    refresh = 500,
    max_treedepth = 15,
    output_dir = file_save_path_ab
  )
}
