# coding: utf-8

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from sklearn import preprocessing	
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def objective_function(parameters, model):

    if model.model_type[:2] == 'AB':
        model.AB_Learning(parameters)
    elif model.model_type[:2] == 'SR':
        model.SR_Learning(parameters)
    else:
        return None

    mse = glm_fit(model)

    return mse

def glm_fit(model, fitpara=False):
    allData = pd.concat([model.data, pd.DataFrame(model.PResult, columns=['P'])], axis=1)

    if model.fit_with_error == 0:
        # remove outlier and error trials
        correData = allData[allData['error_outlier'] == 0]
        
    else:
        # remove outlier trials only
        correData = allData[allData['error_outlier'] != 2]

    correData = correData.reset_index(drop=True)
    # xData = pd.concat(2*[correData['P']-1, correData['congruency'], correData['hand'], correData['postError'],
    #                    correData['block1'], correData['block2'], correData['block3']], axis=1)
    #xData = pd.concat(2 * [correData['P'] - 1, correData['congruency'], correData['Response']], axis=1)
    if 'nblock_14' in correData.columns:
        xData = pd.concat([correData['P'],
	                       correData['nblock_1'], correData['nblock_2'], correData['nblock_3'], correData['nblock_4'],
	                       correData['nblock_5'], correData['nblock_6'], correData['nblock_7'], correData['nblock_8'],
	                       correData['nblock_9'], correData['nblock_10'], correData['nblock_11'], correData['nblock_12'],
	                       correData['nblock_13'], correData['nblock_14']], axis=1)
    else:
        xData = pd.concat([correData['P'],
	                       correData['nblock_1'], correData['nblock_2'], correData['nblock_3'], correData['nblock_4'],
	                       correData['nblock_5'], correData['nblock_6'], correData['nblock_7'], correData['nblock_8'],
	                       correData['nblock_9'], correData['nblock_10'], correData['nblock_11'], correData['nblock_12'],
	                       correData['nblock_13']], axis=1)

    xData.columns.values[0] = 'P'

    if model.fit_with_logRT == 0:
        yData = correData['RT']
    else:
        yData = np.log(correData['RT'])

    linear_regression = LinearRegression()
    linear_regression.fit(xData, yData)
    rtPred = linear_regression.predict(xData)
    rtPred = pd.DataFrame(rtPred, columns={'rt'})
    mse = mean_squared_error(yData, rtPred)

    if fitpara:
        regre = sm.OLS(yData, xData)
        regreResuls = regre.fit()
        coef = regreResuls.params
        coef = pd.DataFrame(coef).T
        sigP = regreResuls.pvalues
        sigP = pd.DataFrame(sigP).T
        sigP.columns = sigP.columns.str.swapcase()
        rSquared = regreResuls.rsquared
        logLikeliHood = regreResuls.llf
        AIC = 2 * model.parameter_size - 2 * logLikeliHood
        BIC = np.log(yData.size) * model.parameter_size - 2 * logLikeliHood
        fitParas1 = pd.concat([coef, sigP], axis=1)
        fitParas2 = pd.DataFrame([rSquared, logLikeliHood, AIC, BIC]).T
        fitParas2.columns = ['rSquared', 'LLH', 'AIC', 'BIC']
        fitParas = pd.concat([fitParas1, fitParas2], axis=1)
        return fitParas
    else:
        return mse
