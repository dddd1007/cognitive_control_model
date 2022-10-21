raw_data <- read.csv("/Users/dddd1007/Desktop/new_sub_data.csv")
library(tidyverse)
data_without_outlier <- raw_data %>%
  filter(correction == 1) %>%
  group_by(Subject, cong, prop, volatility) %>%
  na.omit() %>%
  filter(abs(RT - mean(RT)) < (sd(RT) * 3))

result_list <- list()
result_count = 1
for (sub in unique(data_without_outlier$Subject)) {
  for (vol in unique(data_without_outlier$volatility)) {
    for (prop_i in unique(data_without_outlier$prop)) {
      sub_con_data <- filter(data_without_outlier, 
                             Subject == sub, 
                             volatility == vol, 
                             prop == prop_i,
                             cong == "con") %>% 
                      arrange(RT)
      sub_inc_data <- filter(data_without_outlier, 
                             Subject == sub, 
                             volatility == vol, 
                             prop == prop_i,
                             cong == "inc") %>% 
                      arrange(RT)
      
      # 三等分变量
      quantile_con <- quantile(sub_con_data$RT, probs = c(1,2,3)/3)
      quantile_inc <- quantile(sub_inc_data$RT, probs = c(1,2,3)/3)
      
      # 分别计算每部份的 Simon 效应
      SE_part_1 <- mean(sub_inc_data$RT[sub_inc_data$RT <= quantile_inc[1]]) - 
                   mean(sub_con_data$RT[sub_con_data$RT <= quantile_con[1]])
      SE_part_2 <- mean(sub_inc_data$RT[sub_inc_data$RT <= quantile_inc[2] &
                                        sub_inc_data$RT >  quantile_inc[1]]) - 
                   mean(sub_con_data$RT[sub_con_data$RT <= quantile_con[2] &
                                        sub_con_data$RT >  quantile_con[1]])
      SE_part_3 <- mean(sub_inc_data$RT[sub_inc_data$RT >  quantile_inc[2]]) - 
                   mean(sub_con_data$RT[sub_con_data$RT >  quantile_con[2]])
      
      # 保存数据
      result_list[[result_count]] <- cbind(sub, vol, prop_i, 
                                           SE_part_1, SE_part_2, SE_part_3)
      result_count = result_count + 1
    }
  }
}

# 聚合数据
do.call(rbind.data.frame, result_list) %>% 
  pivot_longer(cols = starts_with("SE_"), 
               names_to = "size_by_RT", 
               values_to = "Simon_effect") %>%
  na.omit() -> delta_plot_table_RT
delta_plot_table_RT$Simon_effect <- as.numeric(delta_plot_table_RT$Simon_effect) 
delta_plot_table_RT %>%
  group_by(vol, prop_i, size_by_RT) %>%
  summarise(mean_Simon_effect = mean(Simon_effect, na.rm = TRUE)) -> delta_plot_table_RT_for_ggplot
delta_plot_table_RT_for_ggplot$size_by_RT <- as.numeric(
  str_remove(delta_plot_table_RT_for_ggplot$size_by_RT, "SE_part_"))
delta_plot_table_RT_for_ggplot %>%
  ggplot(aes(x = size_by_RT, y = mean_Simon_effect)) +
  geom_point(aes(color = prop_i, shape = vol), size = 3) +
  geom_line(aes(color = prop_i, linetype = vol))

delta_plot_table_RT %>%
  filter(prop_i == 20, size_by_RT == "SE_part_1") %>%
  na.omit() %>%
  pivot_wider(names_from = vol, values_from = Simon_effect) -> data_for_ttest
t.test(x = data_for_ttest$v, y = data_for_ttest$s)