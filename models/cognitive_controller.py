# TODO 需要让 session 2 的计算继承 session 1 的估计结果

import numpy as np
import pymc3 as pm
import pandas as pd
import math

class decision_maker:
    def receive_values(self, value_vector, subject_choice_option, total_options, tau, trial_index):
        self.value_vector = value_vector
        self.subject_choice_option = subject_choice_option
        self.total_options = total_options
        self.tau = tau
        self.op_options = list(self.total_options)
        self.op_options.remove(self.subject_choice_option)
        self.trial_index = trial_index

    def decision(self, debug = False):
        self.decision_maker_debug_dict = {'error' : 'The key parameter of debug is False'}
        self.p_softmax = np.exp((3 ** self.tau - 1) * self.value_vector[self.subject_choice_option - 1]) / \
                         (np.exp((3 ** self.tau - 1) * self.value_vector[self.subject_choice_option - 1]) +
                          np.exp((3 ** self.tau - 1) * sum([self.value_vector[x - 1] for x in self.op_options])))

        if debug:
            self.decision_maker_debug_dict = dict(trial_index = self.trial_index, value_vector = self.value_vector,
                                                  subject_choice_option = self.subject_choice_option, tau = self.tau,
                                                  op_options = self.op_options, p_softmax = self.p_softmax)
            return self.decision_maker_debug_dict
        else:
            return self.p_softmax

class bayesian_lerner:
    def __init__(self, observation, dim):
        self.observation = observation
        self.dim = dim # The number of dimensions of the value estimated in brain

    def fit(self):
        foo = pd.read_csv("/Users/dddd1007/project2git/cognitive_control_model/data/sub01_Yangmiao_s.csv")
        tag = {'con':1, 'inc':0}
        bar = [tag[x] for x in foo['contigency']]

        k_hat = [1]
        v_hat = [1]
        r_hat = [0.5]

        for observed_data in bar:
            with pm.Model() as bayesian_lerner_model:
                k = pm.Normal("k", mu = k_hat[-1], sigma = 1)
                k_ = pm.Deterministic('k_', pm.math.exp(k))
                v = pm.Normal("v", mu = v_hat[-1], sigma = k_)
                v_ = pm.Deterministic('v_', pm.math.exp(v))
                r = pm.Beta("r", mu = r_hat[-1], sd = v_)
                y = pm.Bernoulli("y", p = r, observed = observed_data)#shape = self.shape, observed = self.observation)

                trace = pm.sample()

                estimate_result = pm.sample_posterior_predictive(trace)

            bayesian_lerner_model.check_test_point()

            k_hat.append(estimate_result['k'].mean(axis = 0))
            v_hat.append(estimate_result['v'].mean(axis = 0))
            r_hat.append(estimate_result['r'].mean(axis = 0))

        print(bayesian_lerner_model.basic_RVs)

        bayesian_lerner_model.sample()
