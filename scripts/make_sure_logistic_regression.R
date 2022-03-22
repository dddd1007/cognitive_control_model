y_80 <- c(1,1,1,0,0,1,1,1,1,1)
y_60 <- c(1,1,1,0,0,1,1,0,0,1)

x_80_80 <- c(80,80,80,20,20,80,80,80,80,80)
x_60_80 <- c(60,60,60,40,40,60,60,60,60,60)
x_60_60 <- c(60,60,60,40,40,60,60,40,40,60)
x_80_60 <- c(80,80,80,20,20,80,80,20,20,80)

mydata <- data.frame(y_80, y_60, x_80_80, x_80_60, x_60_60, x_60_80)

m1 <- glm(y_80~x_80_80, family = binomial(), data = mydata)
m2 <- glm(y_80~x_60_80, family = binomial(), data = mydata)
m3 <- glm(y_60~x_80_60, family = binomial(), data = mydata)
m4 <- glm(y_60~x_60_60, family = binomial(), data = mydata)

anova(m1, m2)
anova(m3, m4)
# 预期 m1 > m2, m4 > m3


