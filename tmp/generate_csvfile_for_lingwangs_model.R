library(tidyverse)
library(fastDummies)
all_data <- read_csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data_transformed.csv")

renamer <- function(input_dataframe){
    input_dataframe <- select(input_dataframe, stim_loc,Type,Response,RT, congruency, Trial, correct_action, prop, condition)
    input_dataframe <- input_dataframe %>%
        mutate(error_outlier = abs(Type - 1)) %>%
        rename(stimPosit = stim_loc, correction = Type, 
               respPosit = correct_action, 
               hand = Response, trial_index = Trial)
    postError <- c(0, input_dataframe$error_outlier)
    postError <- postError[-length(postError)]
   
    input_data <- cbind(input_dataframe, postError)
    
    return(input_data)
}

change_outlier <- function(input_dataframe){
    RT_value <- input_dataframe$RT
    sd_RT <- sd(RT_value)
    mean_RT <- mean(RT_value)
    outlier_data <- rep(0, nrow(input_dataframe))
    index_outlier_3sd <- which(RT_value < mean_RT -3*sd_RT | RT_value > mean_RT + 3*sd_RT)
    index_error <- which(input_dataframe$correction == 0)
    outlier_data[index_outlier_3sd] <- 2
    outlier_data[index_error] <- 1
    
    input_dataframe$error_outlier <- outlier_data
    return(input_dataframe)
}

change_block_index <- function(input_dataframe) {
    prop_value <- input_dataframe$prop
    block_value <- rep(0, length(prop_value))
    block_index <- 1
    last_prop <- prop_value[1]
    for (i in 1:length(prop_value)) {
        if (last_prop != prop_value[i]) {
            block_index <- block_index + 1
            last_prop = prop_value[i]
        }
        block_value[i] <- block_index 
    }
    input_dataframe$nblock <- block_value
    result_dataframe <- dummy_cols(input_dataframe, select_columns = "nblock")
    return(result_dataframe)
}

savepath = "/Users/dddd1007/project2git/cognitive_control_model/ref_code/Lingwang_CCC/data/wang_model_input/"
for (i in 1:18) {
    foo = filter(all_data, Subject_num == i)
    bar = change_outlier(change_block_index(renamer(foo)))
    write_csv(bar, paste0(savepath, "sub_", i, "_prepared_data.csv"))
}
