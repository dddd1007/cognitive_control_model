library(tidyverse)
data_loc <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bl_estimate_by_full_data/sr/extracted_data/"
sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv")

sub_num_list <- sort(unique(sub_data$Subject_num))
all_data_list <- list()
count_num <- 1

for (i in sub_num_list) {
    data_file <- read.csv(paste0(data_loc, "sub_", i, "_sr_learner.csv"))
    all_data_list[[count_num]] <- cbind(sub_num = i, data_file)
    count_num <- count_num + 1
}

all_data_result <- bind_rows(all_data_list, .id = "column_label")
all_data_result <- mutate(all_data_result, PE = 1 - r_selected)
write.csv(all_data_result, paste0(data_loc, "bl_sr_all_data_gathered.csv"))
