library(cmdstanr)
library(posterior)
library(bayesplot)

check_cmdstan_toolchain()

file <- "~/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_sr.stan"

bayesian_sr_learner <- cmdstan_model(file)