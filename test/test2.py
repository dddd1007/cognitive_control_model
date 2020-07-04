# Ref Libraries
import numpy as np
import pymc3 as pm
import pandas as pd

foo = pd.read_csv("/home/dddd1007/cognitive_control_model/data/sub01_Yangmiao_s.csv")

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
r1_list = [0.5]
r2_list = [0.5]

k_cap = []
v_cap = []
y1_list = []
y2_list = []

#for observed_data in bar:
for observed_data in list(zipped_data): 
    with pm.Model() as bayesian_lerner_model_2dim:
        k = pm.Normal("k", mu = k_list[-1], sigma = 1000)
        k_ = pm.Deterministic('k_cap', pm.math.exp(k))
        v = pm.Normal("v", mu = v_list[-1], sigma = k_)
        v_ = pm.Deterministic('v_cap', pm.math.exp(v))
        r1 = pm.Beta("r1", alpha = (r1_list[-1] / v_), beta = ((1-r1_list[-1]) / v_))
        r2 = pm.Beta("r2", alpha = (r2_list[-1] / v_), beta = ((1-r2_list[-1]) / v_))
        y1 = pm.Bernoulli("y_loc", p = r1, observed = observed_data[0])
        y2 = pm.Bernoulli("y_beh", p = r2, observed = observed_data[1])

        trace = pm.sample(cores=6, tune=1000)

    k_list.append(trace['k'].mean())
    v_list.append(trace['v'].mean())
    r1_list.append(trace['r1'].mean())
    r2_list.append(trace['r2'].mean())
    k_cap.append(trace['k_cap'].mean())
    v_cap.append(trace['v_cap'].mean())

del(k_list[0])
del(v_list[0])
del(r1_list[0])
del(r2_list[0])

results_model3 = {'k_list': k_list, 'v_list': v_list, 
                  'r1_list': r1_list, 'r2_list': r2_list, 
                  'k_cap': k_cap, "v_cap": v_cap}

results_model3_table = pd.DataFrame(results_model3)
results_model3_table.to_csv("/home/dddd1007/cognitive_control_model/data/output/multi_dim_bayesian_learner/model3.csv")