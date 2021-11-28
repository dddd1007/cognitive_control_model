data {
  int<lower=1> N; // trial number
  int keep_seq_len;
  int y[N];
  real RT[N];     // reaction time
  int keep_seq[keep_seq_len];
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
  vector[N] r_selected;
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
      r_selected[t] = r[t];
    }else if(y[t] == 0){
      r_selected[t] = (1-r[t]);
    }
  }
  RT[keep_seq] ~ normal(alpha + beta * r_selected[keep_seq],sigma);
}
