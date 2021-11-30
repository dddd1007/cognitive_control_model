data {
  int<lower=1> N;         // trial number
  int corr_react[N];      // sub's correct reaction
  int space_loc[N];       // stim space location
  int volatility[N];      // stim volatility (0=stable, 1=volatile)
}

parameters {
  real<lower = 0, upper = 1> alpha_s;
  real<lower = 0, upper = 1> alpha_v;
}

model {
  vector[N+1] p_l;    // Probability for stim on left
  vector[N+1] p_r;    // Probability for stim on right
  p_l[1] = 0.5;
  p_r[1] = 0.5;
  for(t in 1:N){
    if(space_loc[t] == 0){
        if(volatility[t] == 0){
            p_l[t + 1] = p_l[t] + alpha_s * (corr_react[t] - p_l[t]);
        }else{
            p_l[t + 1] = p_l[t] + alpha_v * (corr_react[t] - p_l[t]);
        }
        p_r[t + 1] = p_r[t];
        corr_react[t] ~ bernoulli(p_l[t + 1]);
    }else{
        p_l[t + 1] = p_l[t];
        if(volatility[t] == 0){
            p_r[t + 1] = p_r[t] + alpha_s * (corr_react[t] - p_r[t]);
        }else{
            p_r[t + 1] = p_r[t] + alpha_v * (corr_react[t] - p_r[t]);
        }
        p_r[t + 1] = p_r[t];
        corr_react[t] ~ bernoulli(p_r[t + 1]); // Use a logistic regression to estimate alpha
    }
  }  
}