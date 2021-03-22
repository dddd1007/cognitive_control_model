library(tidyverse)

xia_data <- read.csv("/Users/dddd1007/project2git/fmri_analysis2_nipype_type/data/inputs/behavioral_data/wang_2a1d1CCC_P.csv")
wang_dir <- "/Users/dddd1007/project2git/cognitive_control_model/data/input/optim_model_wang"

xia_sub1 <- filter(xia_data, data)
for i in 2:36
