# TODO 需要让 session 2 的计算继承 session 1 的估计结果

import numpy as np
import pymc3 as pm
import pandas as pd
import numba
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
        foo = pd.read_csv("/Users/dddd1007/project2git/cognitive_control_model/data/sub01_Yangmiao_v.csv")
        tag = {'con':1, 'inc':0}
        bar = [tag[x] for x in foo['contigency']]

        k_list = [np.array(0.5)]
        v_list = [np.array(0.5)]
        r_list = [np.array(0.5)]

        k_hat = []
        v_hat = []

        for observed_data in bar:
            with pm.Model() as bayesian_lerner_model:
                k = pm.Normal("k", mu = k_list[-1], sigma = 1)
                k_ = pm.Deterministic('k_hat', pm.math.exp(k))
                v = pm.Normal("v", mu = v_list[-1], sigma = k_)
                v_ = pm.Deterministic('v_hat', pm.math.exp(v))
                r = pm.Beta("r", mu = r_list[-1], sigma =v_)
                y = pm.Bernoulli("y", p = r, observed = observed_data) #, shape = self.dim)

                point_estimate = pm.find_MAP()

            k_list.append(point_estimate['k'])
            v_list.append(point_estimate['v'])
            r_list.append(point_estimate['r'])

            k_hat.append(point_estimate['k_hat'])
            v_hat.append(point_estimate['v_hat'])


