library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

ab_learner_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_ab.stan"
sr_learner_stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan"

bayesian_ab_learner <- cmdstan_model(ab_learner_stanfile)
bayesian_sr_learner <- cmdstan_model(sr_learner_stanfile)

#####
##### Estimate Model
#####

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")
sub1_data <- filter(raw_data, Subject_num == 1)

## part 1 ab models
N = nrow(sub1_data)
y = as.numeric(str_replace(str_replace(sub1_data$congruency, 'con', '1'), 'inc', '0'))
data_list <- list(N = N, y = y)

fit <- bayesian_ab_learner$sample(
  data = data_list,
  chains = 4,
  parallel_chains = 4,
  refresh = 500
)

result_table <- fit$summary()
ab_v <- result_table$mean[str_detect(result_table$variable, "v")]
cor.test(ab_v, sub1_data$I_hats)
ab_v
sub1_data$I_hats

ab_r <- result_table$mean[str_detect(result_table$variable, "r")]
cor.test(ab_r, sub1_data$RT)
