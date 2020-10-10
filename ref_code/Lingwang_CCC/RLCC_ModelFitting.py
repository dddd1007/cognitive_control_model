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


data_folder = './data/'
results_folder = './model_results/'

subjects = ['sub01','sub02','sub04','sub13','sub14','sub15','sub16','sub33',
            'sub34','sub35','sub36','sub37','sub38','sub39','sub40',
            'sub09','sub10','sub11','sub12','sub17','sub18','sub19','sub20',
            'sub41','sub42','sub43','sub44','sub45','sub46','sub47','sub48',
            'sub05','sub06','sub07','sub08','sub21','sub22','sub23','sub24',
            'sub25','sub26','sub27','sub28','sub29','sub30','sub31','sub32']

models = ['SR_Q_WOB', 'SR_Q_D_WOB', 'SR_Q_D_E_WOB', 'SR_Q_D_alphaCCC_WOB', 'SR_Q_D_E_alphaCCC_WOB']
#models = ['AB_Q_WOB', 'AB_Q_E_WOB', 'AB_Q_alphaCCC_WOB', 'AB_Q_E_alphaCCC_WOB']

for s, subject in enumerate(subjects):
    print('********************')
    print(subject)
    #read data
    filename = data_folder + 'behavior_prepared_modelfit_data_' + subject + '.csv'
    data = pd.read_csv(filename)
    for m, model_type in enumerate(models):
        a = datetime.now()
        RLCCmodel = RLCCModels(model_type, data)
        RLCCmodel.subject = subject
        RLCCmodel.results_folder = results_folder

        # for fit_with_error in range(0, 2):
        #     for fit_with_logRT in range(0, 2):
        #         RLCCmodel.fit_with_error = fit_with_error  # 0, GLM fit without error; 1, with error
        #         RLCCmodel.fit_with_logRT = fit_with_logRT  # 0, GLM fit raw RT; 1, with np.log(RT)
        #
        #         print(model_type)
        #         regreMse = 10000000
        #
        #         for n in range(0, 100):
        #             #print('******************n is ', n, '******************')
        #             RLCCmodel.initial_x0()
        #             fit = minimize(objective_function, RLCCmodel.parameter_x0, args=(RLCCmodel), method='L-BFGS-B', bounds=RLCCmodel.model_para_bounds)
        #
        #             #fit = minimize(objective_function, x0, args=(RLmodel), method='Nelder-Mead', options={'disp': True})
        #             if fit.fun < regreMse:
        #                 RLCCmodel.fitted_x = fit.x
        #                 RLCCmodel.fitted_mse = fit.fun
        #                 bestModel = copy.deepcopy(RLCCmodel)
        #
        #         b = datetime.now()
        #         print('time: ', (b - a).seconds)
        #
        #         print(bestModel.fitted_mse, '\n')
        #
        #         bestModel.save_data()
# # group results
RLCCmodel.group_parameter_results(subjects, models, '', '_SR')
RLCCmodel.group_parameter_results(subjects, models, 'logRT_', '_SR')
RLCCmodel.group_parameter_results(subjects, models, 'with_error_', '_SR')
RLCCmodel.group_parameter_results(subjects, models, 'logRT_with_error_', '_SR')

# do the model selection based on AIC
RLCCmodel.model_selection()