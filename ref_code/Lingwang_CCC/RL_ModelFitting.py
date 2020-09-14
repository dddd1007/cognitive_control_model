#!/usr/bin/env python
# coding: utf-8

# In[1]:


from models.RL_Models import RLModels 
from fitting.obj_function import objective_function
from fitting.best_model_regression import doFittedRegre
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
import sys

if sys.platform == 'win32':
	data_folder = 'Z:/Huang/orthogonal_simon_proportion_eeg/behavioral_data/model_fit/scripts/data/'
elif sys.platform == 'linux':
	data_folder = '/Data/Huang/orthogonal_simon_proportion_eeg/behavioral_data/model_fit/scripts/data/'

else:
	data_folder = '/Volumes/Data/Huang/orthogonal_simon_proportion_eeg/behavioral_data/model_fit/scripts/data/'

subjects = ['sub01','sub02','sub04','sub13'] 

models = ['OB_SAME_AB', 'OB_SAME_SR', 'OB_DIFF_AB', 'OB_DIFF_SR', 'OP_SAME_AB', 'OP_SAME_SR', 'OP_DIFF_AB', 
          'OP_DIFF_SR', 'OPerror_SAME_AB', 'OPerror_SAME_SR', 'OPerror_DIFF_AB', 'OPerror_DIFF_SR']

allParasResults = pd.DataFrame()
for s, subject in enumerate(subjects):
    #read data
    filename = data_folder + 'behavior_prepared_modelfit_data_' + subject + '.csv'
    data = pd.read_csv(filename)
    for m, model_type in enumerate(models):
        RLmodel = RLModels(model_type, data)
        print('********************')
        print(model_type)
        regreMse = 1000000000000000000000
        
        for n in range(0, 100):
            alpha = random.uniform(0.0001, 1)
            alpha2 = random.uniform(0.0001, 1)
            alpha3 = random.uniform(0.0001, 1)
            alpha_error = random.uniform(0.0001, 1)
            alpha_error2 = random.uniform(0.0001, 1)
            alpha_error3 = random.uniform(0.0001, 1)
            beta = random.uniform(0.0001, 200)
            decay = random.uniform(0.0001, 1)
#             x0 = models[model][0]
#             boundsVals = models[model][1] 
            if model_type == 'OB_SAME_AB':
                #alpha, beta
                x0 = np.array([alpha, beta])
                boundsVals = [(0.0001, 1), (0.0001, 200)]
                Q_name = ['Q']
                PE_name = ['PE']
                modelP_name = ['alpha', 'beta']
            elif model_type == 'OB_SAME_SR':
                #alpha, beta, decay
                x0 = np.array([alpha, beta, decay])
                boundsVals = [(0.0001, 1), (0.0001, 200), (0.0001, 1)]
                Q_name = ['Q1', 'Q2', 'Q3', 'Q4']
                PE_name = ['PE1', 'PE2', 'PE3', 'PE4']
                modelP_name = ['alpha', 'beta', 'decay']
            elif model_type == 'OB_DIFF_AB':
                #alpha[nblock], beta
                x0 = np.array([alpha, alpha2, alpha3, beta])
                boundsVals = [(0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 200)]
                Q_name = ['Q']
                PE_name = ['PE']
                modelP_name = ['alpha', 'alpha2', 'alpha3', 'beta']
            elif model_type == 'OB_DIFF_SR':
                #alpha[nblock], beta, decay
                x0 = np.array([alpha, alpha2, alpha3, beta, decay])
                boundsVals = [(0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 200), (0.0001, 1)]
                Q_name = ['Q1', 'Q2', 'Q3', 'Q4']
                PE_name = ['PE1', 'PE2', 'PE3', 'PE4']
                modelP_name = ['alpha', 'alpha2', 'alpha3', 'beta', 'decay']
            if model_type == 'OP_SAME_AB':
                #alpha, beta
                #note: OP_SAME_AB is actually identical to OB_SAME_AB
                x0 = np.array([alpha, beta])
                boundsVals = [(0.0001, 1), (0.0001, 200)]
                Q_name = ['Q']
                PE_name = ['PE']
                modelP_name = ['alpha', 'beta']
            elif model_type == 'OP_SAME_SR':
                #alpha, beta, decay
                x0 = np.array([alpha, beta, decay])
                boundsVals = [(0.0001, 1), (0.0001, 200), (0.0001, 1)]
                Q_name = ['Q1', 'Q2', 'Q3', 'Q4']
                PE_name = ['PE1', 'PE2', 'PE3', 'PE4']
                modelP_name = ['alpha', 'beta', 'decay']
            elif model_type == 'OP_DIFF_AB':
                #alpha[nblock], beta
                x0 = np.array([alpha, alpha2, alpha3, beta])
                boundsVals = [(0.0001, 1),  (0.0001, 1), (0.0001, 1), (0.0001, 200)]
                Q_name = ['Q']
                PE_name = ['PE']
                modelP_name = ['alpha', 'alpha2', 'alpha3', 'beta']
            elif model_type == 'OP_DIFF_SR':
                #alpha[nblock], beta, decay
                x0 = np.array([alpha, alpha2, alpha3, beta, decay])
                boundsVals = [(0.0001, 1),  (0.0001, 1), (0.0001, 1), (0.0001, 200), (0.0001, 1)]
                Q_name = ['Q1', 'Q2', 'Q3', 'Q4']
                PE_name = ['PE1', 'PE2', 'PE3', 'PE4']
                modelP_name = ['alpha', 'alpha2', 'alpha3', 'beta', 'decay']
            if model_type == 'OPerror_SAME_AB':
                #alpha, beta, alpha_error
                x0 = np.array([alpha, beta, alpha_error])
                boundsVals = [(0.0001, 1), (0.0001, 200), (0.0001, 1)]
                Q_name = ['Q']
                PE_name = ['PE']
                modelP_name = ['alpha', 'beta', 'alpha_error']
            elif model_type == 'OPerror_SAME_SR': 
                #alpha, beta, decay, alpha_error
                x0 = np.array([alpha, beta, decay, alpha_error])
                boundsVals = [(0.0001, 1), (0.0001, 200), (0.0001, 1), (0.0001, 1)]
                Q_name = ['Q1', 'Q2', 'Q3', 'Q4']
                PE_name = ['PE1', 'PE2', 'PE3', 'PE4']
                modelP_name = ['alpha', 'beta', 'decay', 'alpha_error']
            elif model_type == 'OPerror_DIFF_AB':
                #alpha(nblock), beta, alpha_error[nblock]
                x0 = np.array([alpha, alpha2, alpha3, beta, alpha_error, alpha_error2,
                               alpha_error3])
                boundsVals = [(0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 200), 
                              (0.0001, 1),(0.0001, 1), (0.0001, 1)]
                Q_name = ['Q']
                PE_name = ['PE']
                modelP_name = ['alpha', 'alpha2', 'alpha3', 'beta', 'alpha_error', 'alpha_error2',
                               'alpha_error3']
            elif model_type == 'OPerror_DIFF_SR':
                #alpha(nblock), beta, decay, alpha_error[nblock]
                x0 = np.array([alpha, alpha2, alpha3, beta, decay, alpha_error, alpha_error2,
                               alpha_error3])
                boundsVals = [(0.0001, 1), (0.0001, 1), (0.0001, 1), (0.0001, 200), (0.0001, 1),
                              (0.0001, 1),(0.0001, 1), (0.0001, 1)]
                Q_name = ['Q1', 'Q2', 'Q3', 'Q4']
                PE_name = ['PE1', 'PE2', 'PE3', 'PE4']
                modelP_name = ['alpha', 'alpha2', 'alpha3', 'beta', 'decay', 'alpha_error', 'alpha_error2',
                               'alpha_error3']
            
            fit = minimize(objective_function, x0, args=(RLmodel), method='L-BFGS-B', bounds=boundsVals,options={'disp': True})
            #fit = minimize(objective_function, x0, args=(RLmodel), method='Nelder-Mead', options={'disp': True})
            print(fit.fun)
            if fit.fun < regreMse:
                regreMse = fit.fun
                modelParas = fit.x
                bestModel = RLmodel
        PResult = pd.DataFrame(bestModel.PResult)
        PResult.columns = ['P']
#         QResult = pd.DataFrame(np.array(bestModel.QResult).reshape(-1,bestModel.QResult[0].size))
        QResult = bestModel.QResult
        QResult.reset_index(drop=True,inplace=True)
        QResult.columns = Q_name
        PEResult = bestModel.PEResult
        PEResult.reset_index(drop=True,inplace=True)
        PEResult.columns = PE_name
        modelVars = pd.concat([PResult, QResult, PEResult], axis=1)
        modelRsults = pd.concat([bestModel.data,modelVars], axis=1).to_csv(data_folder + 'model_results_' + model_type + '_' + subject + '.csv')
        fitParas = doFittedRegre(bestModel)
        fitParas['MSE'] = regreMse
        fitParas['model'] = model_type
        modelP = pd.DataFrame(modelParas).T
        modelP.columns = modelP_name
        parasResults = pd.concat([fitParas, modelP], axis=1)
        parasResults['subject'] = subject
        paraRes = parasResults
        paraRes.to_csv(data_folder + 'modelFit_results_'+ model_type + '_'  +subject + '.csv')
        allParasResults = pd.concat([allParasResults,parasResults])
allParasResults.to_csv(data_folder + 'modelFit_results.csv')
#         bestModel.save_data()
    
   
        

