data {
  int<lower=0> N; // trial number
  int y[N];
}

parameters {
  real k;
  real<lower=0.1, upper=0.9> r[N];
  real v[N];
}

model {
  k~uniform(-10,10);
  for(t in 1:N){
    if(t == 1){
      v[t] ~ uniform(0,100);
      r[t] ~ normal(0.5,0.3);
    }
    else{
      v[t] ~ normal(v[t-1],exp(k));
      r[t] ~ beta_proportion(r[t-1],exp(v[t]));
    }
    y[t] ~ bernoulli(r[t]);
  }
}
