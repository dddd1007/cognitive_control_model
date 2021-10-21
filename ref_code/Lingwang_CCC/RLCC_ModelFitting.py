# coding: utf-8

# In[1]:


from RLCC_Models import RLCCModels
from obj_function import objective_function
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import socket
import sys
import warnings
import copy
from datetime import datetime
warnings.filterwarnings("ignore", category=FutureWarning, module="RLModels")


data_folder = './data_block/'
results_folder = './model_results_final/'

subjects = ['sub_1', 'sub_2', 'sub_3', 'sub_4', 'sub_5', 'sub_6', 'sub_7', 'sub_8', 'sub_9', 'sub_10', 
            'sub_11', 'sub_12', 'sub_13', 'sub_14', 'sub_15', 'sub_16', 'sub_17', 'sub_18', 
            'sub_19', 'sub_20', 'sub_21', 'sub_22', 'sub_23',
            'sub_24','sub_25','sub_26','sub_28','sub_29','sub_30','sub_31','sub_32',
            'sub_33', 'sub_34', 'sub_35', 'sub_36']

models = ['SR_Q_V_WOB', 'SR_Q_D_V_WOB', 'SR_Q_D_E_V_WOB', 'SR_Q_D_alphaCCC_V_WOB', 'SR_Q_D_E_alphaCCC_V_WOB', 'AB_Q_WOB', 'AB_Q_E_WOB', 'AB_Q_alphaCCC_WOB', 'AB_Q_E_alphaCCC_WOB']

#models = ['SR_Q_D_alphaCCC_V_WOB', 'SR_Q_D_E_alphaCCC_V_WOB']
#models = ['AB_Q_WOB', 'AB_Q_E_WOB', 'AB_Q_alphaCCC_WOB', 'AB_Q_E_alphaCCC_WOB']
#models = ['SR_Q_D_WOB', 'SR_Q_D_alphaCCC_WOB']

for s, subject in enumerate(subjects):
    print('********************')
    print(subject)
    #read data
    filename = data_folder  + subject + '_prepared_data.csv'
    data = pd.read_csv(filename)
    for m, model_type in enumerate(models):
        a = datetime.now()
        RLCCmodel = RLCCModels(model_type, data)
        RLCCmodel.subject = subject
        RLCCmodel.results_folder = results_folder

        for fit_with_error in range(0, 1):  #range(0, 2)
            for fit_with_logRT in range(0, 1):  #range(0, 2)
                RLCCmodel.fit_with_error = fit_with_error  # 0, GLM fit without error; 1, with error
                RLCCmodel.fit_with_logRT = fit_with_logRT  # 0, GLM fit raw RT; 1, with np.log(RT)

                print(model_type)
                regreMse = 10000000

                for n in range(0, 100):
                    #print('******************n is ', n, '******************')
                    RLCCmodel.initial_x0()
                    fit = minimize(objective_function, RLCCmodel.parameter_x0, args=(RLCCmodel), method='L-BFGS-B', bounds=RLCCmodel.model_para_bounds)

                    #fit = minimize(objective_function, x0, args=(RLmodel), method='Nelder-Mead', options={'disp': True})
                    if fit.fun < regreMse:
                        RLCCmodel.fitted_x = fit.x
                        RLCCmodel.fitted_mse = fit.fun
                        bestModel = copy.deepcopy(RLCCmodel)
                        regreMse = fit.fun

                b = datetime.now()
                print('time: ', (b - a).seconds)

                print(bestModel.fitted_mse, '\n')

                bestModel.save_data()
# # group results
RLCCmodel.group_parameter_results(subjects, models, '')
#RLCCmodel.group_parameter_results(subjects, models, 'logRT_', '_SR')
#RLCCmodel.group_parameter_results(subjects, models, 'with_error_', '_SR')
#RLCCmodel.group_parameter_results(subjects, models, 'logRT_with_error_', '_SR')

# do the model selection based on AIC
RLCCmodel.model_selection()