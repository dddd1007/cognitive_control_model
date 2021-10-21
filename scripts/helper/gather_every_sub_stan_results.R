library(tidyverse)
data_loc <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/ab/process/extracted_data/"
sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")

sub_num_list <- sort(unique(sub_data$Subject_num))
all_data_list <- list()
count_num <- 1

for (i in sub_num_list) {
    data_file <- read.csv(paste0(data_loc, "sub_", i, "_ab_learner.csv"))
    all_data_list[[count_num]] <- cbind(sub_num = i, data_file)
    count_num <- count_num + 1
}

all_data_result <- bind_rows(all_data_list, .id = "column_label")
all_data_result <- mutate(all_data_result, PE = 1 - r_selected)
write.csv(all_data_result, "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/ab/process/extracted_data/gatherd/all_data_gatherd.csv")