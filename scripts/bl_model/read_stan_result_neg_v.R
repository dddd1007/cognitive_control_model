library(cmdstanr)
library(tidyverse)

read_loc <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/1k1v_neg_v/"
save_loc <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/1k1v_neg_v/extracted_data/"
sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")


for (sub_num in unique(sub_data$Subject_num)) {
    # 读取单个被试的数据
    csv_files <- dir(paste0(read_loc, "sub_", sub_num),
        pattern = ".csv", full.names = TRUE
    )
    single_sub_data <- filter(sub_data, Subject_num == sub_num)
    stan_data_list <- list()
    for (i in seq_len(length(csv_files))) {
        stan_data_list[[i]] <- read_csv(csv_files[i],
            comment = "#",
            num_threads = readr_threads()
        )
    }

    stan_data <- bind_rows(stan_data_list, .id = "column_label")

    # 分条件将数据读取并求均值

    # r_l指刺激物空间位置在左侧时，右手按键的概率
    rlr_data <- select(stan_data, starts_with("r_l"))
    rlr_mean <- apply(rlr_data, 2, mean)
    rlr_mean <- c(0.5, rlr_mean) # stan 的估计结果为当前试次更新后的, 因此需要向后推一个试次
    rll_mean <- 1 - rlr_mean

    # r_r指刺激物空间位置在右侧时，右手按键的概率
    rrr_data <- select(stan_data, starts_with("r_r"))
    rrr_mean <- apply(rrr_data, 2, mean)
    rrr_mean <- c(0.5, rrr_mean)
    rrl_mean <- 1 - rrr_mean

    # 获取 v 的列表
    v_data <- select(stan_data, starts_with("v"))
    v_mean <- apply(v_data, 2, mean)

    # 根据被试应当正确的行为选出对应的r

    r_selected <- vector(mode = "numeric", length = nrow(single_sub_data))
    for (i in seq_len(nrow(single_sub_data))) {
        tmp <- single_sub_data[i, ]
        if (tmp$stim_loc == "left" & tmp$correct_action == 0) {
            r_selected[i] <- rll_mean[i]
        } else if (tmp$stim_loc == "left" & tmp$correct_action == 1) {
            r_selected[i] <- rlr_mean[i]
        } else if (tmp$stim_loc == "right" & tmp$correct_action == 0) {
            r_selected[i] <- rrl_mean[i]
        } else if (tmp$stim_loc == "right" & tmp$correct_action == 1) {
            r_selected[i] <- rrr_mean[i]
        }
    }
    #####
    result <- data.frame(
        ll = rll_mean[1:(length(rll_mean) - 1)],
        lr = rlr_mean[1:(length(rlr_mean) - 1)],
        rl = rrl_mean[1:(length(rrl_mean) - 1)],
        rr = rrr_mean[1:(length(rrr_mean) - 1)],
        v = v_mean, r_selected = r_selected
    )
    result_filename <- paste0(save_loc, "sub_", sub_num, "_sr_learner.csv")
    write_csv(result, result_filename)
}
