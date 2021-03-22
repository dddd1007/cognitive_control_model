library(tidyverse)
filepath <- "/Users/dddd1007/project2git/cognitive_control_model/data/input/optim_model_wang"

file_names <- dir(filepath, full.names = TRUE)

detect_nrow <- function(file_name) {
    foo <- read.csv(file_name)
    foo_row <- nrow(foo)

    return(foo_row)
}

for (i in file_names) {
    print(detect_nrow(i))
}
