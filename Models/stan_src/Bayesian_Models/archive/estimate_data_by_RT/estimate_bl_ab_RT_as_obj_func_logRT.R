library(cmdstanr)
library(posterior)
library(bayesplot)
library(tidyverse)

check_cmdstan_toolchain()

stanfile <- "/Users/dddd1007/project2git/cognitive_control_model/Bayesian_Models/bayesian_learner_ab_estimate_RT_as_obj_func.stan"
learner <- cmdstan_model(stanfile)
output_dir <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/bayesian_learner_samplers/AB_RT_as_obj_func_logRT/"

##### helper func
generate_keep_seq <- function(input_dataframe, filter_type = "no_error"){
    keep_index <- seq(1, nrow(input_dataframe))
    foo_dataframe <- cbind(input_dataframe, keep_index)
    result_dataframe <- filter(foo_dataframe, Type == "hit")
    if (filter_type == "no outlier"){
        result_dataframe <- result_dataframe %>%
                        group_by(congruency, prop, condition) %>%
                        filter(abs(RT - mean(RT)) < (sd(RT) * 3))
    }
    keep_seq <- result_dataframe$keep_index
    return(keep_seq)
}
#####
##### Estimate Model
#####

# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")

# Estimate each sub
sub_num_list <- unique(raw_data$Subject_num)
# sub_num_list <- sub_num_list[-1]

for (i in sub_num_list) {
    print(paste0("Estimating model for subject ", i))
    single_sub_table <- filter(raw_data, Subject_num == i)
    
    N <- nrow(single_sub_table)
    y <- as.numeric(str_replace(str_replace(single_sub_table$congruency, 'con', '1'), 'inc', '0'))
    RT <- log(single_sub_table$RT) # !!! Here I transform RT to logRT !!!
    keep_seq <- generate_keep_seq(single_sub_table, filter_type = "no_error")

    data_list <- list(N = N,
                      y = y,
                      RT = log(RT),
                      keep_seq = keep_seq,
                      keep_seq_len = length(keep_seq))

    file_save_path <- paste0(output_dir, "sub_", as.character(i))
    dir.create(file_save_path)
    fit <- learner$sample(
        data = data_list,
        chains = 4,
        parallel_chains = 4,
        refresh = 100,
        save_warmup = 0,
        output_dir = file_save_path
    )
}

read_loc <- output_dir
extracted_data_output_dir <- paste0(read_loc, "/extracted_data/")
if (!dir.exists(extracted_data_output_dir)){
    dir.create(extracted_data_output_dir)
}
for (sub_num in sub_num_list) {
    print(paste0("Extract data from subject ", sub_num))
    # 读取单个被试的数据
    csv_files <- dir(paste0(read_loc, "sub_", sub_num),
        pattern = ".csv", full.names = TRUE
    )
    single_sub_data <- filter(raw_data, Subject_num == sub_num)
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
        extracted_data_output_dir,
        "sub_", sub_num, "_ab_with_RT_learner.csv"
    )
    print(paste0("Save data to ", result_filename))
    write_csv(result, result_filename)
}

data_loc <- extracted_data_output_dir
all_data_list <- list()
count_num <- 1
dir.create(paste0(data_loc, "/gatherd"))

for (i in sub_num_list) {
    data_file <- read.csv(paste0(data_loc, "sub_", i, "_ab_with_RT_learner.csv"))
    all_data_list[[count_num]] <- cbind(sub_num = i, data_file)
    count_num <- count_num + 1
}

all_data_result <- bind_rows(all_data_list, .id = "column_label")
all_data_result <- mutate(all_data_result, PE = 1 - r_selected)
write.csv(all_data_result, paste0(data_loc, "gatherd/all_data_gatherd.csv"))
