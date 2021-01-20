library(tidyverse)
baseline_margin <- read_csv("/Users/dddd1007/project2git/cognitive_control_model/ref_code/Lingwang_CCC/data/sub01.csv") %>%
    select(block1,block2,block3,block4,block5,block6)
all_data <- read_csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data_transformed.csv")

renamer <- function(input_dataframe, baseline_margin){
    input_dataframe <- select(input_dataframe, stim_loc,Type,Response,RT, congruency, run, Trial, correct_action)
    input_dataframe <- input_dataframe %>%
        mutate(error_outlier = abs(Type - 1)) %>%
        rename(stimPosit = stim_loc, correction = Type, respPosit = correct_action, nblock = run, hand = Response)
    postError <- c(0, input_dataframe$error_outlier)
    postError <- postError[-length(postError)]
    baseline_margin = baseline_margin[1:nrow(input_dataframe),]

    input_data <- cbind(input_dataframe, postError, baseline_margin)
    
    return(input_dataframe)
}

savepath = "/Users/dddd1007/project2git/cognitive_control_model/ref_code/Lingwang_CCC/data/"
for (i in 28:36) {
    foo = filter(all_data, Subject_num == i)
    bar = renamer(foo, baseline_margin)
    write_csv(bar, paste0(savepath, "sub_", i, "_prepared_data.csv"))
}
