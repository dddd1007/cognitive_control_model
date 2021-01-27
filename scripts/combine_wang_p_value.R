library(tidyverse)
all_sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data.csv")

result_list <- list()
for (i in 1:18) {
    filepath = paste0("/Users/dddd1007/project2git/cognitive_control_model/ref_code/Lingwang_CCC/model_results/1. add_block/RLCC_model_results_SR_Q_D_alphaCCC_V_WOB_sub_", i, ".csv")
    sub_data <- filter(all_sub_data, Subject_num == i, Type == "hit" | Type == "incorrect")
    p_value <- read.csv(filepath)$P
    foo = cbind(sub_data, p_value)
    result_list[[i]] <- foo
}

Wang_2a1d1CCC_P <- do.call(rbind, result_list)

write.csv(Wang_2a1d1CCC_P, "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/wang_2a1d1CCC_P.csv")
