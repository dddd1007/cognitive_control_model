library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

sr_1k1v_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Models/stan_src/Bayesian_Models/bayesian_learner_sr_1k1v_neg_v.stan"

sr_1k1v_learner <- cmdstan_model(sr_1k1v_stanfile)

output_dir_1k1v <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/1k1v_neg_v/"

#####
##### Estimate Model
#####

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")

# Estimate each sub
sub_num_list <- unique(raw_data$Subject_num)

for (i in sub_num_list) {
  if (i < 8) {next}
  
  print(paste0("==== Begin Subject ", as.character(i), " ====="))
  single_sub_table <- filter(raw_data, Subject_num == i)

  N <- nrow(single_sub_table)
  corr_react <- single_sub_table$correct_action
  space_loc <- single_sub_table$stim_loc_num
  data_list <- list(N = N, corr_react = corr_react, space_loc = space_loc)

  # model1 1k1v
  file_save_path_1k1v <- paste0(output_dir_1k1v, "sub_", as.character(i))
  dir.create(file_save_path_1k1v)
  fit1 <- sr_1k1v_learner$sample(
    data = data_list,
    chains = 4,
    parallel_chains = 4,
    refresh = 500,
    output_dir = file_save_path_1k1v
  )
}
