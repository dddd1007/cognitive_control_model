# coding: utf-8

import numpy as np
import pandas as pd
import sys
#import transplant
import obj_function

if sys.platform == 'linux':
    sys.path.append('./build/lib.linux-x86_64-3.7')
elif sys.platform == 'win32':
    sys.path.append('./build/lib.win-amd64-3.7')
import RLCC

class RLCCModels(object):
    def __init__(self, model_type, data):
        self.subject = []
        self.results_folder = []
        self.model_type = model_type
        self.data = data
        self.data.columns = self.data.columns.str.strip()
        self.trialNum = data['trial_index'].size
        self.PResult, self.QResult, self.PEResult = [],[],[]
        self.Q_name = []
        self.PE_name = []
        self.model_para = []
        self.model_para_names = []
        self.parameter_size = []

        self.parameter_x0 = []
        self.alpha_bound =(0.0001, 1)
        self.beta_bound = (0.0001, 200)
        self.ccc_bound = (-1, 0)

        self.fit_with_error = 0 # 0, GLM fit without error; 1, with error
        self.fit_with_logRT = 0 # 0, GLM fit raw RT; 1, with np.log(RT)

        self.fitted_x = []
        self.fitted_mse = []

        #AB model with beta
        if model_type == 'AB_Q':
            #alpha, beta
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_E':
            #alpha, beta, alpha_error
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta', 'alpha_error']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_E_alphaCCC':
            #alpha, beta, alpha_error, ccc, alpha_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta', 'alpha_error', 'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_E_betaCCC':
            #alpha, beta, alpha_error, ccc, beta_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta', 'alpha_error', 'ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.ccc_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_E_alphaCCC_betaCCC':
            #alpha, beta, alpha_error, ccc, alpha_ccc, beta_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta', 'alpha_error', 'ccc', 'alpha_ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_alphaCCC':
            #alpha, beta, ccc, alpha_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta', 'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.ccc_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_betaCCC':
            #alpha, beta, ccc, beta_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta', 'ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.ccc_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_alphaCCC_betaCCC':
            #alpha, beta, ccc, alpha_ccc, beta_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'beta', 'ccc', 'alpha_ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.ccc_bound, self.alpha_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)

        # AB model without beta
        if model_type == 'AB_Q_WOB':
            # alpha
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha']
            self.model_para_bounds = [self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_E_WOB':
            # alpha, alpha_error
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'alpha_error']
            self.model_para_bounds = [self.alpha_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_E_alphaCCC_WOB':
            # alpha, alpha_error, ccc, alpha_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha', 'alpha_error', 'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'AB_Q_alphaCCC_WOB':
            # alpha, ccc, alpha_ccc
            self.Q_name = ['Q']
            self.PE_name = ['PE']
            self.model_para_names = ['alpha',  'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.ccc_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)

        #SR model with beta
        if model_type == 'SR_Q':
            #alpha, beta
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound]
            self.parameter_size=len(self.model_para_names)
        elif model_type == 'SR_Q_D':
            #alpha, beta, decay
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_E':
            #alpha, beta, decay, alpha_error
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay', 'alpha_error']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_E_alphaCCC':
            #alpha, beta, decay, alpha_error, ccc, alpha_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay', 'alpha_error', 'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_E_betaCCC':
            #alpha, beta, decay, alpha_error, ccc, beta_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay', 'alpha_error', 'ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.alpha_bound, self.ccc_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_E_alphaCCC_betaCCC':
            #alpha, beta, decay, alpha_error, ccc, alpha_ccc, beta_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay', 'alpha_error', 'ccc', 'alpha_ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_alphaCCC':
            #alpha, beta, decay, ccc, alpha_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay', 'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_betaCCC':
            #alpha, beta, decay, ccc, beta_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay',  'ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.ccc_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_alphaCCC_betaCCC':
            #alpha, beta, decay, ccc, alpha_ccc, beta_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'beta', 'decay', 'ccc', 'alpha_ccc', 'beta_ccc']
            self.model_para_bounds = [self.alpha_bound, self.beta_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound, self.beta_bound]
            self.parameter_size = len(self.model_para_names)

        # SR model without beta
        if model_type == 'SR_Q_WOB':
            # alpha
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = [ 'alpha']
            self.model_para_bounds = [self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_WOB':
            # alpha, decay
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = [ 'alpha', 'decay']
            self.model_para_bounds = [self.alpha_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_E_WOB':
            # alpha, decay, alpha_error
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'decay', 'alpha_error']
            self.model_para_bounds = [self.alpha_bound, self.alpha_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_alphaCCC_WOB':
            # alpha, decay, ccc, alpha_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = ['alpha', 'decay', 'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.alpha_bound, self.ccc_bound, self.alpha_bound]
            self.parameter_size = len(self.model_para_names)
        elif model_type == 'SR_Q_D_E_alphaCCC_WOB':
            # alpha, decay, alpha_error, ccc, alpha_ccc
            self.Q_name = ['Q_ll', 'Q_lr', 'Q_rl', 'Q_rr']
            self.PE_name = ['PE_ll', 'PE_lr', 'PE_rl', 'PE_rr']
            self.model_para_names = [ 'alpha', 'decay', 'alpha_error', 'ccc', 'alpha_ccc']
            self.model_para_bounds = [self.alpha_bound, self.alpha_bound, self.alpha_bound, self.ccc_bound,
                                      self.alpha_bound]
            self.parameter_size = len(self.model_para_names)

    def SR_Learning(self, parameters):

        self.model_para = parameters
        model_type = self.model_type
        stimPosit, respPosit, correction = self.data['stimPosit'].tolist(), self.data['hand'].tolist(), \
                                                       self.data['correction'].tolist()
        if model_type[-3:] == 'WOB':
            P, Q, pe = RLCC.SR_Learning_without_beta(model_type, parameters.tolist(), stimPosit, respPosit, correction)
        else:
            P, Q, pe = RLCC.SR_Learning(model_type, parameters.tolist(), stimPosit, respPosit, correction)

        self.PResult = P
        self.QResult = Q
        self.PEResult = pe

    def AB_Learning(self, parameters):

        self.model_para = parameters
        model_type = self.model_type
        congruency, correction = self.data['congruency'].tolist(), self.data['correction'].tolist()

        if model_type[-3:] == 'WOB':
            P, Q, pe = RLCC.AB_Learning_without_beta(model_type, parameters.tolist(), congruency, correction)
        else:
            P, Q, pe = RLCC.AB_Learning(model_type, parameters.tolist(), congruency, correction)

        self.PResult = P
        self.QResult = Q
        self.PEResult = pe

    def initial_x0(self):

        len_x0 = len(self.model_para_bounds)
        x0_tmp = np.zeros(len_x0)

        for ix in range(len_x0):
            x0_tmp[ix] = np.random.uniform(self.model_para_bounds[ix][0], self.model_para_bounds[ix][1])

        self.parameter_x0 = x0_tmp

    def save_data(self):

        PResult = pd.DataFrame(self.PResult)
        PResult.columns = ['P']
        QResult = pd.DataFrame(np.array(self.QResult).reshape(-1, np.array(self.QResult[0]).size))
        QResult.reset_index(drop=True, inplace=True)
        QResult.columns = self.Q_name
        PEResult = pd.DataFrame(np.array(self.PEResult).reshape(-1, np.array(self.PEResult[0]).size))
        PEResult.reset_index(drop=True, inplace=True)
        PEResult.columns = self.PE_name
        modelVars = pd.concat([PResult, QResult, PEResult], axis=1)

        if self.fit_with_error == 0:
            prefix_str = ''
        else:
            prefix_str = 'with_error_'

        if self.fit_with_logRT == 0:
            prefix_str = '' + prefix_str
        else:
            prefix_str = 'logRT_' + prefix_str

        pd.concat([self.data, modelVars], axis=1).to_csv(self.results_folder + prefix_str + 'RLCC_model_results_' + self.model_type + '_' + self.subject + '.csv')

        fitParas = obj_function.glm_fit(self, True)
        fitParas['MSE'] = self.fitted_mse
        fitParas['model'] = self.model_type
        modelP = pd.DataFrame(self.fitted_x).T
        modelP.columns = self.model_para_names
        parasResults = pd.concat([fitParas, modelP], axis=1)
        parasResults['subject'] = self.subject
        parasResults.to_csv(self.results_folder + prefix_str + 'RLCC_parameter_results_' + self.model_type + '_' + self.subject + '.csv')

    def group_parameter_results(self, subjects, model_types, prefix_str = '', subfix_str = ''):

        rSquared_pd = pd.DataFrame()
        LLH_pd = pd.DataFrame()
        AIC_pd = pd.DataFrame()
        BIC_pd = pd.DataFrame()
        MSE_pd = pd.DataFrame()

        for m, model_type in enumerate(model_types):
            data_all = pd.DataFrame()
            for s, subject in enumerate(subjects):
                filename = self.results_folder + prefix_str + 'RLCC_parameter_results_' + model_type + '_' + subject + '.csv'
                data = pd.read_csv(filename)
                data_all = pd.concat([data_all, data], axis=0)

            data_all = data_all.loc[:, ~data_all.columns.str.match('Unnamed: 0')]
            data_all.to_csv(self.results_folder + 'Group_' + prefix_str + 'RLCC_parameter_results_' + model_type + '.csv', index=False)

            rSquared_pd = pd.concat([rSquared_pd, data_all['rSquared']], axis=1)
            rSquared_pd.columns.values[-1] = model_type

            LLH_pd = pd.concat([LLH_pd, data_all['LLH']], axis=1)
            LLH_pd.columns.values[-1] = model_type

            AIC_pd = pd.concat([AIC_pd, data_all['AIC']], axis=1)
            AIC_pd.columns.values[-1] = model_type

            BIC_pd = pd.concat([BIC_pd, data_all['BIC']], axis=1)
            BIC_pd.columns.values[-1] = model_type

            MSE_pd = pd.concat([MSE_pd, data_all['MSE']], axis=1)
            MSE_pd.columns.values[-1] = model_type

        pd.concat([data_all['subject'], rSquared_pd], axis=1).to_csv(self.results_folder + 'Model_comparison_rSquared_' + prefix_str + 'RLCC_parameter_results' + subfix_str + '.csv', index=False)
        pd.concat([data_all['subject'], LLH_pd], axis=1).to_csv(self.results_folder + 'Model_comparison_LLH_' + prefix_str + 'RLCC_parameter_results' + subfix_str + '.csv', index=False)
        pd.concat([data_all['subject'], AIC_pd], axis=1).to_csv(self.results_folder + 'Model_comparison_AIC_' + prefix_str + 'RLCC_parameter_results' + subfix_str + '.csv', index=False)
        pd.concat([data_all['subject'], BIC_pd], axis=1).to_csv(self.results_folder + 'Model_comparison_BIC_' + prefix_str + 'RLCC_parameter_results' + subfix_str + '.csv', index=False)
        pd.concat([data_all['subject'], MSE_pd], axis=1).to_csv(self.results_folder + 'Model_comparison_MSE_' + prefix_str + 'RLCC_parameter_results' + subfix_str + '.csv', index=False)
        print('end of group_parameter_results')

    def model_selection(self, prefix_str = ''):
        # filename = 'Model_comparison_AIC_' + prefix_str + 'RLCC_parameter_results' + '.csv'
        # matlab = transplant.Matlab(jvm=False, desktop=False)
        # matlab.model_selection(filename)
        # matlab.exit()
        print('\nrun model_selection(\'Model_comparison_AIC_"prefix_str"_RLCC_parameter_results.csv\') in Matlab')
        print('“prefix_str” could be \'\', logRT, with_error, logRT_with_error')
        print('the output is model_selection_"filename".csv in the folder of model_results')

