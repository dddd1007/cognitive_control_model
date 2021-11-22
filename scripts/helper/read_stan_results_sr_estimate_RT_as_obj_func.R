library(tidyverse)

sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")
stan_verbose_loc <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/SR_RT_as_obj_func"

sub_num_list <- sort(unique(sub_data$sub_num))

read_loc <- paste0(stan_verbose_loc, "/")
for (sub_num in unique(sub_data$Subject_num)) {
    print(paste0("Extract data from subject ", sub_num))
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
    rll_mean <- 1 - rlr_mean

    # r_r指刺激物空间位置在右侧时，右手按键的概率
    rrr_data <- select(stan_data, starts_with("r_r"))
    rrr_mean <- apply(rrr_data, 2, mean)
    rrl_mean <- 1 - rrr_mean

    # 获取 v 的列表
    v_data <- select(stan_data, starts_with("v"))
    v_mean <- apply(v_data, 2, mean)

    # 获取 RT 的 linear model 的估计结果
    alpha <- mean(stan_data$alpha)
    beta  <- mean(stan_data$beta)
    sigma <- mean(stan_data$sigma)
    decay_rate <- mean(stan_data$decay_rate)

    # 根据被试的行为选出对应的r

    r_selected <- vector(mode = "numeric", length = nrow(single_sub_data))
    for (i in seq_len(nrow(single_sub_data))) {
        tmp <- single_sub_data[i, ]
        if (tmp$stim_loc == "left" & tmp$Response == 0) {
            r_selected[i] <- rll_mean[i]
        } else if (tmp$stim_loc == "left" & tmp$Response == 1) {
            r_selected[i] <- rlr_mean[i]
        } else if (tmp$stim_loc == "right" & tmp$Response == 0) {
            r_selected[i] <- rrl_mean[i]
        } else if (tmp$stim_loc == "right" & tmp$Response == 1) {
            r_selected[i] <- rrr_mean[i]
        }
    }
    #####
    result <- data.frame(
        ll = rll_mean, lr = rlr_mean, rl = rrl_mean, rr = rrr_mean,
        v = v_mean, r_selected = r_selected
    )
    result_filename <- paste0(read_loc, "extracted_data/",
                              "sub_", sub_num, "_sr_learner.csv")
    print(paste0("Save data to ", result_filename))
    write_csv(result, result_filename)
}
