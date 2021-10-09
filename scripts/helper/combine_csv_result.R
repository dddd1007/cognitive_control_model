csvpath <- "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/SR"


basic_list <- dir(csvpath, full.names = T, pattern = "basic.csv")
error_list <- dir(csvpath, full.names = T, pattern = "error.csv")
CCC_list <- dir(csvpath, full.names = T, pattern = "CCC.csv")

library(plyr)
basic_set <- ldply(basic_list, read.csv, header=TRUE)
error_set <- ldply(error_list, read.csv, header=TRUE)
CCC_set <- ldply(CCC_list, read.csv, header=TRUE)

write.csv(basic_set, "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/SR_basic.csv")
write.csv(error_set, "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/SR_error.csv")
write.csv(CCC_set, "/Users/dddd1007/project2git/cognitive_control_model/data/output/RLModels/SR_CCC.csv")