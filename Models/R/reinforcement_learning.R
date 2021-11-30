# Modeling
rl_ab_model <- function(alpha, consistency_seq) {
    result_vector <- vector(mode = "numeric", length = length(consistency_seq) + 1)
    result_vector[1] <- 0.5
    
    # update
    for (i in 1:length(consistency_seq)) {
        result_vector[i + 1] <- result_vector[i] + alpha * (consistency_seq[i] - result_vector[i])
    }
    return(result_vector)
} 

rl_sr_model <- function(alpha, stim_space_loc, corr_reaction) {
    result_vector_stim_l <- vector(mode = "numeric", length = length(stim_space_loc) + 1)
    result_vector_stim_r <- vector(mode = "numeric", length = length(stim_space_loc) + 1)
    r_selected <- vector(mode = "numeric", length = length(stim_space_loc) + 1)

    result_vector_stim_l[1] <- 0.5
    result_vector_stim_r[1] <- 0.5
    r_selected[1] <- 0.5

    # update
    for (i in 1:length(stim_space_loc)) {
         if (stim_space_loc[i] == "0") {
            result_vector_stim_l[i + 1] <- result_vector_stim_l[i] + alpha * (corr_reaction[i] - result_vector_stim_l[i])
            result_vector_stim_r[i + 1] <- result_vector_stim_r[i]
            r_selected[i + 1] <- result_vector_stim_l[i + 1]
         } else {
            result_vector_stim_l[i + 1] <- result_vector_stim_l[i]
            result_vector_stim_r[i + 1] <- result_vector_stim_r[i] + alpha * (corr_reaction[i] - result_vector_stim_r[i])
            r_selected[i + 1] <- result_vector_stim_r[i + 1]
         }
    }

    return(r_selected)
}

rl_sr_volatile_model <- function(alpha_s, alpha_v, stim_space_loc, corr_reaction, volatility_seq) {
    result_vector_stim_l <- vector(mode = "numeric", length = length(stim_space_loc) + 1)
    result_vector_stim_r <- vector(mode = "numeric", length = length(stim_space_loc) + 1)
    r_selected <- vector(mode = "numeric", length = length(stim_space_loc) + 1)

    result_vector_stim_l[1] <- 0.5
    result_vector_stim_r[1] <- 0.5
    r_selected[1] <- 0.5

    # update
    for (i in 1:length(stim_space_loc)) {
        if (stim_space_loc[i] == "0") {
            if (volatility_seq[i] == "0") {
                result_vector_stim_l[i + 1] <- result_vector_stim_l[i] + alpha_s * (corr_reaction[i] - result_vector_stim_l[i])
            } else {
                result_vector_stim_l[i + 1] <- result_vector_stim_l[i] + alpha_v * (corr_reaction[i] - result_vector_stim_l[i])
            }
            result_vector_stim_r[i + 1] <- result_vector_stim_r[i]
            r_selected[i + 1] <- result_vector_stim_l[i + 1]
        } else {
            result_vector_stim_l[i + 1] <- result_vector_stim_r[i]
            if (volatility_seq[i] == "0") {
                result_vector_stim_r[i + 1] <- result_vector_stim_r[i] + alpha_s * (corr_reaction[i] - result_vector_stim_r[i])
            } else {
                result_vector_stim_r[i + 1] <- result_vector_stim_r[i] + alpha_v * (corr_reaction[i] - result_vector_stim_r[i])
            }
            r_selected[i + 1] <- result_vector_stim_r[i + 1]
        }
    }
    return(r_selected)
}

# obj_func
rl_ab_obj_func <- function(alpha, consistency_seq) {
    predict_seq <- rl_ab_model(alpha, consistency_seq)
    model_data <- data.frame(predict_seq = predict_seq[-length(predict_seq)], consistency_seq)
    model <- glm(consistency_seq ~ predict_seq, family = binomial, data = model_data)
    return(AIC(model))
}

rl_sr_obj_func <- function(alpha, stim_space_loc, corr_reaction) {
    predict_seq <- rl_sr_model(alpha, stim_space_loc, corr_reaction)
    model_data <- data.frame(predict_seq = predict_seq[-length(predict_seq)], corr_reaction)
    model <- glm(corr_reaction ~ predict_seq, family = binomial, data = model_data)
    return(AIC(model))
}

rl_sr_volatile_obj_func <- function(alpha_s, alpha_v, stim_space_loc, corr_reaction, volatility_seq) {
    predict_seq <- rl_sr_volatile_model(alpha_s, alpha_v, stim_space_loc, corr_reaction, volatility_seq)
    model_data <- data.frame(predict_seq = predict_seq[-length(predict_seq)], corr_reaction)
    model <- glm(corr_reaction ~ predict_seq, family = binomial, data = model_data)
    return(AIC(model))
}

# Import data
# Load data
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/all_data_with_I_hats.csv")
sub_num_list <- unique(raw_data$Subject_num)
# Set output directory
output_dir <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/rl_model/"

# Grid search
## rl_ab
alpha_grid <- seq(0, 1, 0.01)

rl_ab_result <- list()
# for (i in 1:length(sub_num_list)) {
   i = 1
    single_sub_data <- filter(raw_data, Subject_num == sub_num_list[i])
    consistency_seq <- as.numeric(str_replace(str_replace(single_sub_data$congruency, 'con', '1'), 'inc', '0'))
    
    alpha_vector <- vector(mode = "numeric", length = length(alpha_grid))
    AIC_vector <- vector(mode = "numeric", length = length(alpha_grid))
    for (alpha_num in 1:length(alpha_grid)) {
        obj_func_value <- rl_ab_obj_func(alpha_grid[alpha_num], consistency_seq)
        # print(paste0("alpha:", alpha, "  AIC:", obj_func_value))
        alpha_vector[alpha_num] <- alpha_grid[alpha_num]
        AIC_vector[alpha_num] <- obj_func_value
    }
    rl_ab_result[[i]] <- data.frame(alpha = alpha_vector, AIC = AIC_vector, sub = sub_num_list[i])
}

## rl_sr
rl_sr_result <- list()
for (i in 1:length(sub_num_list)) {
    single_sub_data <- filter(raw_data, Subject_num == sub_num_list[i])
    stim_space_loc <- single_sub_data$stim_loc_num
    corr_reaction <- single_sub_data$correct_action
    
    alpha_vector <- vector(mode = "numeric", length = length(alpha_grid))
    AIC_vector <- vector(mode = "numeric", length = length(alpha_grid))
    for (alpha_num in 1:length(alpha_grid)) {
        obj_func_value <- rl_sr_obj_func(alpha_grid[alpha_num], stim_space_loc, corr_reaction)
        # print(paste0("alpha:", alpha_grid[alpha_num], "  AIC:", obj_func_value))
        alpha_vector[alpha_num] <- alpha_grid[alpha_num]
        AIC_vector[alpha_num] <- obj_func_value
    }
    rl_sr_result[[i]] <- data.frame(alpha = alpha_vector, AIC = AIC_vector, sub = sub_num_list[i])
}
