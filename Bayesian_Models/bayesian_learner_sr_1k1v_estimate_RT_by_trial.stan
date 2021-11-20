data {
  int<lower=1> N;       // trial number
  int stim_space_loc[N];     // stim space location
  int corr_reaction[N]; // sub's correct reaction
  int real_reaction[N]; // real reaction
  real RT[N];           // reaction time
}

parameters {
  real k;
  real<lower=0.01, upper=0.99> r_l[N]; // Probability for stim on left
  real<lower=0.01, upper=0.99> r_r[N]; // Probability for stim on right
  real v[N];
  real alpha;                          // linear model parameters for RT ~ P
  real beta;
  real sigma;
}

// It is equivalent between trial by trail or vectorization form, so I write them in for loop
// Ref from https://mc-stan.org/docs/2_28/stan-users-guide/linear-regression.html

model {
  k~uniform(-10,10);
  for(t in 1:N){
    if(t == 1){
      v[t] ~ uniform(-100,100);
      r_l[t] ~ normal(0.5,0.45);
      r_r[t] ~ normal(0.5,0.45);
      
      // estiamte RT
      if(stim_space_loc[t] == 0){
        if(real_reaction[t] == 0){
          RT[t] ~ normal(alpha + beta * r_l[t], sigma);
        }else if(real_reaction[t] == 1){
          RT[t] ~ normal(alpha + beta * (1 - r_l[t]), sigma);
        }
      }else if(stim_space_loc[t] == 1){
        if(real_reaction[t] == 0){
          RT[t] ~ normal(alpha + beta * r_r[t], sigma);
        }else if(real_reaction[t] == 1){
          RT[t] ~ normal(alpha + beta * (1 - r_r[t]), sigma);
        }
      }
    }else{
      v[t] ~ normal(v[t-1],exp(k));
      if(stim_space_loc[t] == 0){
        r_l[t] ~ beta_proportion(r_l[t-1],exp(v[t]));
        r_r[t] ~ beta_proportion(r_r[t-1],exp(v[t]));
        corr_reaction[t] ~ bernoulli(r_l[t]);
        
        if(real_reaction[t] == 0){
          RT[t] ~ normal(alpha + beta * r_l[t], sigma);
        }else if(real_reaction[t] == 1){
          RT[t] ~ normal(alpha + beta * (1 - r_l[t]), sigma);
        }

      }else if(stim_space_loc[t] == 1){
        r_l[t] ~ beta_proportion(r_l[t-1],exp(v[t]));
        r_r[t] ~ beta_proportion(r_r[t-1],exp(v[t]));
        corr_reaction[t] ~ bernoulli(r_r[t]);

        if(real_reaction[t] == 0){
          RT[t] ~ normal(alpha + beta * r_r[t], sigma);
        }else if(real_reaction[t] == 1){
          RT[t] ~ normal(alpha + beta * (1 - r_r[t]), sigma);
        }
      }
    }
  }
}