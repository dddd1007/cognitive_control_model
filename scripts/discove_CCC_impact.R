## 导入数据

library(tidyverse)
raw_data <- read.csv("/Users/dddd1007/project2git/cognitive_control_model/data/output/summary/subdata_with_CCC_wang_2a1d1CCC.csv")

################### CCC 对RT的影响

### 上一个是 CCC 的trial 和其他的trial之间反应时的差异
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

prep_CCC <- c("false", raw_data$if_below_CCC)
prep_CCC <- as.factor(prep_CCC[-length(prep_CCC)])

lm_prep_data <- cbind(raw_data, prep_CCC)

### 上一个是 CCC 对当前 trial 的反应时是否有影响
lm(RT ~ prep_CCC, data = lm_prep_data) %>% 
  summary()

lm(RT ~ prep_CCC + prop, lm_prep_data) %>% 
  summary()

### 选出当前的 CCC 相同的相同刺激物后续 trial , 查看其反应时
data_add_total_index <- cbind(lm_prep_data, total_index = 1:nrow(lm_prep_data))

CCC_current_table <- filter(data_add_total_index, if_below_CCC == "true")
CCC_current_index <- CCC_current_table$total_index

CCC_next_index <- c()

for(i in CCC_current_index){
  cache_stim_loc   = data_add_total_index[i, ]$stim_loc
  cache_stim_color = data_add_total_index[i, ]$stim_color
  
  for(j in 1){
    if(data_add_total_index[i+j, ]$stim_loc == cache_stim_loc & data_add_total_index[i+j, ]$stim_color == cache_stim_color){
      CCC_next_index[i] = i+j
      break
    }else{
      next
    }
  }
}

CCC_next_table = data_add_total_index[na.omit(CCC_next_index), ]

t.test(CCC_current_table$RT, CCC_next_table$RT, alternative = "greater")
############# CCC 出现的模式
raw_data %>%
  filter(if_below_CCC == "true") %>%
  group_by(Subject_num) %>%
  summarize(prop_inc = 1-mean(congruency))

raw_data %>% 
  filter(if_below_CCC == "true") %>%
  group_by(Subject_num, prop) %>%
  summarize(prop_inc = 1-mean(congruency))
