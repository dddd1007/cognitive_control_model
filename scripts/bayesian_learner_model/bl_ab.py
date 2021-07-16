import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

class bayesian_ab_learner:
    def __init__(self, observed_data):
        self.observed_data = observed_data

    def reset(self):
        self.bayesian_learner = pm.Model()
        with self.bayesian_learner:
            k = pm.Normal("k", mu = 1, sigma = 0.5, testval = 0.6)
            k_ = pm.Deterministic('k_hat', pm.math.exp(k))
            v = pm.GaussianRandomWalk("v", mu = 0.07, sigma = k_, testval = 0.05, shape = len(self.observed_data))
            v_ = pm.Deterministic('v_hat', pm.math.exp(v))

            r = []
            for ti in range(len(self.observed_data)):
                if ti == 0:
                    # Testvals are used to prevent -inf initial probability
                    r.append(pm.Beta(f'r{ti}', 1, 1))
                else:
                    w = r[ti-1]
                    k = 1 / v_[ti-1]
                    r.append(pm.Beta(f'r{ti}', alpha=w*(k-2) + 1, beta=(1-w)*(k-2) + 1, testval = 0.5))

            r = pm.Deterministic('r', pm.math.stack(r))
            y = pm.Bernoulli("y", p = r, observed = self.observed_data)

    def fit(self):
        with self.bayesian_learner:
            self.trace = pm.sample(init="adapt_diag")

        self.r = self.trace['r']
        self.v_hat = self.trace['v_hat']
        self.k = self.trace['k']

    def plot_history(self):
        hdi = az.hdi(self.trace['r'], hdi_prob=.50)
        x = range(len())
        plt.scatter(x, self.trace['r'].mean(0), label='r (posterior mean)')
        plt.plot(x, np.mean(self.trace['v_hat'], axis = 0), alpha = 0.3, label = 'v_hat')
        plt.fill_between(x, hdi[:, 0], hdi[:, 1], color='C1', alpha=.3, label='r (posterior 50% hdi)')
        plt.legend();