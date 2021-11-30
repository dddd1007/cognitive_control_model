data {
  int<lower=1> N; // trial number
  int consistency[N];
}

parameters {
  real<lower = 0, upper = 1> alpha;
}

model {
  vector[N+1] p;
  p[1] = 0.5;
  for(t in 1:N){
    p[t + 1] = p[t] + alpha * (consistency[t] - p[t]);
    consistency[t] ~ bernoulli(p[t + 1]); // Use a logistic regression to estimate alpha
  }
}
