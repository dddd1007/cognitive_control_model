library(tidyverse)
library(here)
source(here("script", "read_logfile.R"))
source(here("script", "check_first_onset.R"))
source(here("script", "read_all_logfile_in_folder.R"))

logdir = here("input", "behavior_data")
all_sub_data <- read_all_logfile_in_folder(logdir)

# 1. 将文件名生成为条件
path_head <- str_extract(all_sub_data$filename, ".*/behavior_data/")[1]
path_tail <- ".log"

all_sub_data$factors <- all_sub_data$filename %>% 
  str_remove_all(path_head) %>% 
  str_remove_all(path_tail)

# 2. 提取被试编号
all_sub_data$Subject_num <- str_extract_all(all_sub_data$Subject,
                                            "[0-9][0-9]") %>%
  unlist() %>% 
  as.numeric()

# 2. 将con与inc标记出来

# 被试 1 - 16 与 被试 35、36
part1_data <- all_sub_data %>% 
  filter(Subject_num <= 16 | Subject_num == 35 | Subject_num == 36)

part2_data <- all_sub_data %>% 
  filter(Subject_num > 16 & Subject_num <35)

# 查看trial数是否正确
print("组1各被试下的trials数")
summary(as.factor(part1_data$Subject_num))

# 对于各组被试添加正确规则信息
part1_data$rule <- "redleft-greenright"
part2_data$rule <- "redright-greenleft"

# 对于两组被试的code进行不同的修改
part1_data$Code %>% 
  str_replace_all("red_left", "con") %>% 
  str_replace_all("red_right", "inc") %>% 
  str_replace_all("green_left", "inc") %>% 
  str_replace_all("green_right", "con") -> part1_data$congruency
part2_data$Code %>% 
  str_replace_all("red_left", "inc") %>% 
  str_replace_all("red_right", "con") %>% 
  str_replace_all("green_left", "con") %>% 
  str_replace_all("green_right", "inc") -> part2_data$congruency

# 增加正确行为反应
part1_data$Code %>% 
  str_replace_all("red_left", "0") %>% 
  str_replace_all("red_right", "0") %>% 
  str_replace_all("green_left", "1") %>% 
  str_replace_all("green_right", "1") -> part1_data$correct_action

part2_data$Code %>% 
  str_replace_all("red_left", "1") %>% 
  str_replace_all("red_right", "1") %>% 
  str_replace_all("green_left", "0") %>% 
  str_replace_all("green_right", "0") -> part2_data$correct_action

## 合并数据
all_sub_data <- rbind(part1_data, part2_data)

# 3. 增加 run 的信息
source(here("script", "add_run_number.R"))
all_sub_data <- all_sub_data %>% 
  split(.$Subject) %>% 
  map(add_run_num) %>% 
  do.call(rbind,.)

# 4. 修正Trials信息, 分开颜色信息与刺激位置, 修正被试行为反应, 0为左手, 1为右手, 
#    NaN为缺失值
all_sub_data$Trial <- str_remove_all(rownames(all_sub_data), "sub.._.*[.]")
all_sub_data %>% 
  separate(Code, c("stim_color", "stim_loc"), sep = "_") -> all_sub_data
all_sub_data$Response[is.na(all_sub_data$Response)] <- NaN
all_sub_data$Response <- as.numeric(as.character(all_sub_data$Response)) - 1

all_sub_data$RT[is.na(all_sub_data$RT)] <- NaN

# 5. 删去无用变量并导出数据
all_sub_data %>% 
  select(-Event.Type, -`RT.Uncertainty` : -`ReqDur`) %>% 
  write.csv(here("output","pure_all_data.csv"))
