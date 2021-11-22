library(tidyverse)

sub_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")
stan_verbose_loc <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/ab_with_RT/"

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
    r_con_data <- select(stan_data, starts_with("r"))
    r_con_mean <- apply(r_con_data, 2, mean)
    r_inc_mean <- 1 - r_con_mean

    v_data <- select(stan_data, starts_with("v"))
    v_mean <- apply(v_data, 2, mean)

    # 获取 RT 的 linear model 的估计结果
    alpha <- mean(stan_data$alpha)
    beta  <- mean(stan_data$beta)
    sigma <- mean(stan_data$sigma)


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
        r_con = r_con_mean, r_inc = r_inc_mean,
        v = v_mean, r_selected = r_selected
    )
    result_filename <- paste0(
        stan_verbose_loc, "extracted_data/",
        "sub_", sub_num, "_ab_with_RT_learner.csv"
    )
    print(paste0("Save data to ", result_filename))
    write_csv(result, result_filename)
}
