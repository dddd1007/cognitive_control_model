data {
  int<lower=1> N;         // trial number
  int corr_react[N];      // sub's correct reaction
  int space_loc[N];       // stim space location
}

parameters {
  real alpha;
  real<lower=0.01, upper=0.99> p_l[N + 1]; // Probability for stim on left
  real<lower=0.01, upper=0.99> p_r[N + 1]; // Probability for stim on right
}

model {
  p_l[1] = 0.5;
  p_r[1] = 0.5;
  for(t in 1:N){
    if(space_loc[t] == 0){
        p_l[t + 1] = p_l[t] + alpha * (corr_react - p_l[t]);
        p_r[t + 1] = p_r[t];
        consistency[t] ~ bernoulli(p_l[t + 1]);
    }else{
        p_l[t + 1] = p_l[t];
        p_r[t + 1] = p[t] + alpha * (consistency[t] - p[t]);
        consistency[t] ~ bernoulli(p_r[t + 1]); // Use a logistic regression to estimate alpha
  }
}