data {
  int<lower=1> N; // trial number
  int y[N];
  real RT[N];     // reaction time
}

parameters {
  real k;
  real<lower=0.01, upper=0.99> r[N];
  real v[N];
  real alpha;                          // linear model parameters for RT ~ P
  real beta;
  real sigma;
}

model {
  k~uniform(-10,10);
  for(t in 1:N){
    if(t == 1){
      v[t] ~ uniform(0,100);
      r[t] ~ normal(0.5,0.45);
    }
    else{
      v[t] ~ normal(v[t-1],exp(k));
      r[t] ~ beta_proportion(r[t-1],exp(v[t]));
    }
    y[t] ~ bernoulli(r[t]);

    if(y[t] == 1){
      RT[t] ~ normal(alpha + beta * r[t], sigma);
    }else if(y[t] == 0){
      RT[t] ~ normal(alpha + beta * (1 - r[t]), sigma);
    }
  }
}
