data {
  int<lower=1> N;         // trial number
  int corr_react[N];      // sub's correct reaction
  int space_loc[N];       // stim space location
}

parameters {
  real alpha;
}

model {
  vector[N+1] p_l;
  vector[N+1] p_r;
  p_l[1] = 0.5;
  p_r[1] = 0.5;
  for(t in 1:N){
    if(space_loc[t] == 0){
        p_l[t + 1] = p_l[t] + alpha * (corr_react[t] - p_l[t]);
        p_r[t + 1] = p_r[t];
        corr_react[t] ~ bernoulli(p_l[t + 1]);
    }else{
        p_l[t + 1] = p_l[t];
        p_r[t + 1] = p_r[t] + alpha * (corr_react[t] - p_r[t]);
        corr_react[t] ~ bernoulli(p_r[t + 1]); // Use a logistic regression to estimate alpha
    }
  }
}