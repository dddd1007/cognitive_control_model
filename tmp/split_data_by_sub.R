## 导入数据

library(tidyverse)
raw_data <- read_csv("/Users/dddd1007/project2git/cognitive_control_model/data/input/pure_all_data_transformed.csv")

for (i in 1:18){
    foo <- filter(raw_data, Subject_num == i)
    filename <- paste0("/Users/dddd1007/project2git/cognitive_control_model/data/input/data_by_sub/sub_", as.character(i), "_prepared_data.csv")
    write_csv(foo, filename)
}
