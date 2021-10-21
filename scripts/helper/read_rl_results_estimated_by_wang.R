library(tidyverse)

sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")
rl_wang_verbose_loc <- "/Users/dddd1007/project2git/cognitive_control_model/ref_code/Lingwang_CCC/model_results_final"

sub_num_list <- sort(unique(sub_data$sub_num))
sub_data_list <- list()
type_list <- c("/RLCC_model_results_AB_Q_WOB_sub_",
               "/RLCC_model_results_SR_Q_D_V_WOB_sub_",
               "/RLCC_model_results_SR_Q_V_WOB_sub_")


for (sub_num in unique(sub_data$Subject_num)) {
    single_sub_data <- filter(sub_data, Subject_num == sub_num)
    tmp_result_list <- list()
    count_num <- 1
    for (file_type in type_list) {
        read_loc <- paste0(rl_wang_verbose_loc, file_type, sub_num, ".csv")
        tmp_result_list[[count_num]] <- read.csv(read_loc)
        count_num <- count_num + 1
    }
    
    # 将单个被试的数据与模型结果合并
    sub_data_list[[sub_num]] <- cbind(single_sub_data,
                                      AB = tmp_result_list[[1]]$P,
                                      SR_Decay = tmp_result_list[[2]]$P,
                                      SR = tmp_result_list[[3]]$P)
    #####
}
all_sub_data <- bind_rows(sub_data_list, .id = "column_label")
write.csv(all_sub_data, "/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_rl_model.csv")
