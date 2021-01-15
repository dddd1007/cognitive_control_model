## 导入数据

library(tidyverse)
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/subdata_with_CCC.csv")

leading_index <- c()
for(i in 1:(nrow(raw_data)-1)){
  leading_table <- raw_data[i,]
  following_table <- raw_data[i+1,]

  if(leading_table$stim_loc == following_table$stim_loc & leading_table$stim_color == following_table$stim_color){
      leading_index <- append(leading_index, i)
  }
}

conflict_index <- c()
for (i in 1:nrow(raw_data)) {
   if (raw_data[i,]$if_below_CCC) {
       conflict_index <- append(conflict_index, i)
   }
}

same_and_conflict_index <- intersect(leading_index, conflict_index)
same_no_conflict_index <- setdiff(leading_index, conflict_index)

RT_conflict <- c()
RT_same <- c()

for (i in same_and_conflict_index) {
    RT_conflict <- append(RT_conflict, (raw_data[i+1,]$RT - raw_data[i,]$RT))
}

for (i in same_no_conflict_index) {
    RT_same <- append(RT_same, (raw_data[i+1,]$RT - raw_data[i,]$RT))
}

t.test(RT_conflict, RT_same)

library(perm)
permTS(RT_conflict, sample(RT_same, length(RT_conflict)), alternative="less")

library(coin)
oneway_test(RT_conflict~sample(RT_same, length(RT_conflict)))
