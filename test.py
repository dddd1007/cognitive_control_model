import numpy as np
import pymc3 as pm
import pandas as pd

foo = pd.read_csv("/Users/dddd1007/project2git/cognitive_control_model/data/sub01_Yangmiao_v.csv")
tag = {'con':1, 'inc':0}
contigency = [tag[x] for x in foo['contigency']]

tag_loc = {'left':1, 'right':0}
locations = [tag_loc[x] for x in foo['location']]

behavior = []
for i in range(0, len(locations)):
    if contigency[i] == 1:
        behavior.append(locations[i])
    else:
        behavior.append(abs(locations[i] - 1))

zipped_data = zip(locations, behavior)
dim = 2

k_list = [1]
v_list = [1]
r_list = [0.5]

k_cap = []
v_cap = []

#for observed_data in bar:
for observed_data in zipped_data: 
    with pm.Model() as bayesian_lerner_model:
        k = pm.Normal("k", mu = k_list[-1], sigma = 1000)
        k_ = pm.Deterministic('k_cap', pm.math.exp(k))
        v = pm.Normal("v", mu = v_list[-1], sigma = k_)
        v_ = pm.Deterministic('v_cap', pm.math.exp(v))
        r = pm.Beta("r", alpha = (r_list[-1] / v_), beta = ((1-r_list[-1]) / v_))
        y = pm.Bernoulli("y", p = r, observed = observed_data, shape = dim)


        trace = pm.sample()

    k_list.append(trace['k'].mean())
    v_list.append(trace['v'].mean())
    r_list.append(trace['r'].mean())
    k_cap.append(trace['k_cap'].mean())
    v_cap.append(trace['v_cap'].mean())


result_dict['k_cap': k_cap