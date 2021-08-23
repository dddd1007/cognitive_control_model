library(rstan)

N = 5
y = c(1,1,0,1,1)
my_data = list(N, y)

fit = stan(file = "~/project2git/cognitive_control_model/tmp/stantest.stan", data = my_data)