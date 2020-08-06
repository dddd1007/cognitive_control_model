# TODO 需要让 session 2 的计算继承 session 1 的估计结果

# TODO 使用 Numpy 来优化一下程序中涉及到的列表

import numpy as np
import pymc3 as pm
import pandas as pd

# TODO 编写提取数据的类

class data_reader:
    """Convert the csv data to np.array to be used in model estimation

    Attributes:
          data: A pandas DataFrame contain all useful data
          data_dict: A dict describe the relation between the item name what you want
                     and the true name in csv table which you import
    """
    def __init__(self, csv_file, data_dict):
        self.data = pd.read_csv(csv_file)
        self.data_dict = data_dict

    def extract_data(self, *item:str) -> dict:
        """Generate a dict contain all data needed in next step

        Args:
            *item: The data name you want to extract. It must be a key in data_dict (The para used
                   to create an instance)

        Returns:
            result_dict: A dict contained all data you need. Every element is a np.array
        """
        result_dict = {}
        for i in item:
            result_dict[i] = np.array(self.data[self.data_dict[item]])

        return result_dict

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

class bayesian_learner:
    def __init__(self, observation, stim_loc):
        self.observation = observation

class bayesian_abstract_learner(bayesian_learner):
    def bayesian_lerner_abstract(self):
        k_list = [1]
        v_list = [1]
        r_list = [0.5]

        k_cap = []
        v_cap = []

        for observed_data in self.observation:
            with pm.Model() as bayesian_lerner_model:
                k = pm.Normal("k", mu = k_list[-1], sigma = 1000)
                k_ = pm.Deterministic('k_cap', pm.math.exp(k))
                v = pm.Normal("v", mu = v_list[-1], sigma = k_)
                v_ = pm.Deterministic('v_cap', pm.math.exp(v))
                r = pm.Beta("r", alpha = (r_list[-1] / v_), beta = ((1-r_list[-1]) / v_))
                y = pm.Bernoulli("y", p = r, observed = observed_data)

                trace = pm.sample()

            k_list.append(trace['k'].mean())
            v_list.append(trace['v'].mean())
            r_list.append(trace['r'].mean())
            k_cap.append(trace['k_cap'].mean())
            v_cap.append(trace['v_cap'].mean())
            # TODO 增加 decay 部分， 每次更新趋近于 0.5
        del(k_list[0])
        del(v_list[0])
        del(r_list[0])

        parameters_dict = {'k_list': k_list, 'v_list': v_list, 'r_list': r_list,
                           'k_cap': k_cap, 'v_cap': v_cap}

        return parameters_dict
                


