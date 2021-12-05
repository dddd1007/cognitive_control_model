library(tidyverse)

sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_sub_data.csv")
stan_verbose_loc <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bl_estimate_by_full_data/ab/"

sub_num_list <- sort(unique(sub_data$Subject_num))
for (sub_num in sub_num_list) {
    # sub_num = 22
    print(paste0("Extract data from subject ", sub_num))
    # 读取单个被试的数据
    csv_files <- dir(paste0(stan_verbose_loc, "sub_", sub_num),
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
    r_con_data <- select(stan_data, starts_with("r"))
    r_con_mean <- apply(r_con_data, 2, mean)
    r_inc_mean <- 1 - r_con_mean
    # con/inc 向量实际上为更新后的数值, 代表对下个试次的估计, 故应在前补 0.5
    r_con_mean <- c(0.5, r_con_mean)
    r_inc_mean <- c(0.5, r_inc_mean)

    v_data <- select(stan_data, starts_with("v"))
    v_mean <- apply(v_data, 2, mean)

    # 根据被试的行为选出对应的r

    r_selected <- vector(mode = "numeric", length = nrow(single_sub_data))
    for (i in seq_len(nrow(single_sub_data))) {
        tmp <- single_sub_data[i, ]
        if (tmp$congruency == "con") {
            r_selected[i] <- r_con_mean[i]
        } else if (tmp$congruency == "inc") {
            r_selected[i] <- r_inc_mean[i]
        }
    }

    result <- data.frame(
        r_con = r_con_mean[1:(length(r_con_mean) - 1)],
        r_inc = r_inc_mean[1:(length(r_con_mean) - 1)],
        v = v_mean, r_selected = r_selected
    )
    result_filename <- paste0(
        stan_verbose_loc, "extracted_data/",
        "sub_", sub_num, "_ab_learner.csv"
    )
    print(paste0("Save data to ", result_filename))
    write_csv(result, result_filename)
}