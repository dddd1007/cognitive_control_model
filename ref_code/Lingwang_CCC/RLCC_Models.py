#!/usr/bin/env python
# coding: utf-8

️import numpy as np
import pandas as pd
import sys
from numba import jit

class RLCCModels(object):
    def __init__(self, model_type, data):
        # 3 (Observer, Oprator, Operator_error)* 2 (same alpha for all blocks, diffent alpha for different blocks) * 2 (abstract learning or S-R learning)
        # OB_SAME_AB, OB_SAME_SR, OB_DIFF_AB, OB_DIFF_SR, OP_SAME_AB, OP_SAME_SR, OP_DIFF_AB, OP_DIFF_SR, OPerror_SAME_AB, OPerror_SAME_SR, OPerror_DIFF_AB, OPerror_DIFF_SR
        self.model_type = model_type
        self.data = data
        self.trialNum = data['trial_index'].size
        self.PResult, self.QResult, self.PEResult = [],[],[]
        self.para_alpha = []
        self.para_alpha_error = []
        self.para_beta = []
        self.para_decay = []
        self.stimPosit = []
        self.respPosit = []

        self.ccc = []

        if model_type == 'AB_Q':
            #alpha, beta
            self.parameter_size=2
        elif model_type == 'OB_SAME_SR':
            #alpha, beta, decay
            self.parameter_size=3
        elif model_type == 'OB_SAME_SR_CCC':
            #alpha, beta, decay, alpha_ccc, ccc
            self.parameter_size=5
        elif model_type == 'OB_SAME_SR_doubleUpdate':
            #alpha, beta
            self.parameter_size=2
        elif model_type == 'OB_DIFF_AB':
            #alpha[nblock], beta
            self.parameter_size=4
        elif model_type == 'OB_DIFF_SR':
            #alpha[nblock], beta, decay
            self.parameter_size=5
            
        if model_type == 'OP_SAME_AB':
            #alpha, beta
            #note: OP_SAME_AB is actually identical to OB_SAME_AB
            self.parameter_size=2
        elif model_type == 'OP_SAME_SR':
            #alpha, beta, decay
            self.parameter_size=3
        elif model_type == 'OP_SAME_SR_CCC':
            #alpha, beta, decay, alpha_ccc, ccc, alpha_error
            self.parameter_size=6
        elif model_type == 'OP_SAME_SR_doubleUpdate':
            #alpha, beta
            self.parameter_size=2
        elif model_type == 'OP_DIFF_AB':
            #alpha[nblock], beta
            self.parameter_size=4
        elif model_type == 'OP_DIFF_SR':
            #alpha[nblock], beta, decay
            self.parameter_size=5
        
        if model_type == 'OPerror_SAME_AB':
            #alpha, beta, alpha_error
            self.parameter_size=3 
        elif model_type == 'OPerror_SAME_SR': 
            #alpha, beta, decay, alpha_error
            self.parameter_size=4
        elif model_type == 'OPerror_SAME_SR_CCC':
            #alpha, beta, decay, alpha_error
            self.parameter_size=6
        elif model_type == 'OPerror_SAME_SR_doubleUpdate': 
            #alpha, beta, alpha_error
            self.parameter_size=3
        elif model_type == 'OPerror_DIFF_AB':
            #alpha(nblock), beta, alpha_error[nblock]
            self.parameter_size=7
        elif model_type == 'OPerror_DIFF_SR':
            #alpha(nblock), beta, decay, alpha_error[nblock]
            self.parameter_size=8
   
    def incongruent_abstract_learning_observer(self, x, nblock=1):
        alpha, beta = x[0], x[1]

        if np.array(alpha).size != nblock:
            print('numbers of alpha must equal to that of nblock')
            sys.exit(1)
       
        Q = np.array(0.5)
        self.para_alpha = alpha
        self.para_beta = beta
        self.PResult = []
        self.QResult = pd.DataFrame()
        self.PEResult = pd.DataFrame()

        for m in np.arange(0, self.trialNum):
            #1: incongruent, 0:congruent
            congruency = int(self.data['congruency'][m])

            #softmax
            P = np.exp(beta * Q)/(np.exp(beta* Q) + np.exp(beta * (1 - Q)))
            #update
            pe = np.array(congruency - Q)           
            
            if nblock>1:
                #mltipule learning rates for different blocks
                block_num = int(self.data['nblock'][m])
                Q = Q + alpha[block_num-1] * pe
                           
            else:
                Q = Q + alpha * pe
                           
            self.PResult.append(P)
            self.QResult = pd.concat([self.QResult, pd.DataFrame(Q.reshape(-1,Q.size))])
            self.PEResult = pd.concat([self.PEResult, pd.DataFrame(pe.reshape(-1,pe.size))])
                                  
                           
    
    def incongruent_abstract_learning_observer_nblock(self, x, nblock=1):
        alpha = x[0:nblock]
        beta = x[nblock]
        x0 = [alpha, beta]
        self.incongruent_abstract_learning_observer(x0, nblock)                          
                                  
    def SR_learning_observer(self, x, nblock=1):
        alpha, beta, decay = x[0], x[1], x[2]     
                        
        if np.array(alpha).size != nblock:
            print('numbers of alpha must equal to that of nblock')
            sys.exit(1)
       
        
        self.para_alpha = alpha
        self.para_beta = beta
        self.para_decay = decay
        Q = np.ones((2, 2), dtype=np.float64) * 0.5
        self.QResult = pd.DataFrame()
        self.PEResult = pd.DataFrame()
        self.PResult = []
        stimPosit, respPosit = [], []
        for m in np.arange(0, self.trialNum):
                       
            #  0:left, 1:right             
            stimPosit, respPosit = int(self.data['stimPosit'][m]), int(self.data['respPosit'][m])
            oppAction = 1-respPosit
           
            #softmax               
            P = np.exp(beta * Q[stimPosit, respPosit]) / (np.exp(beta * Q[stimPosit, respPosit]) + np.exp(beta * Q[stimPosit, oppAction]))
           
            #update
            if nblock > 1:
                #mltipule learning rates for different blocks
                block_num = int(self.data['nblock'][m])
                alpha_tmp=alpha[block_num-1]*P
            else:
                alpha_tmp=alpha*P
                      
                      
            pe = np.ones((2, 2), dtype=np.float64) * 0
            pe[stimPosit, respPosit] = 1 - Q[stimPosit, respPosit]
            Q[stimPosit, respPosit] = Q[stimPosit, respPosit] + alpha_tmp * pe[stimPosit, respPosit]
                           
            pe[stimPosit, oppAction] = 0 - Q[stimPosit, oppAction]
            Q[stimPosit, oppAction] = Q[stimPosit, oppAction] + alpha_tmp*pe[stimPosit, oppAction]
           
            # decay
            opp_stimPosit = 1 - stimPosit
            Q[opp_stimPosit, 0] = Q[opp_stimPosit, 0] + decay*(0.5 - Q[opp_stimPosit, 0])               
            Q[opp_stimPosit, 1] = Q[opp_stimPosit, 1] + decay*(0.5 - Q[opp_stimPosit, 1])               
           
                           
            self.PResult.append(P)
            self.QResult = pd.concat([self.QResult, pd.DataFrame(Q.reshape(-1,Q.size))])
            self.PEResult = pd.concat([self.PEResult, pd.DataFrame(pe.reshape(-1,pe.size))])
            self.stimPosit.append(stimPosit)
            self.respPosit.append(respPosit)

    def SR_learning_ccc_observer(self, x, nblock=1):
        alpha, beta, decay, alpha_ccc, ccc = x[0], x[1], x[2], x[3], x[4]

        if np.array(alpha).size != nblock:
            print('numbers of alpha must equal to that of nblock')
            sys.exit(1)

        self.para_alpha = alpha
        self.para_beta = beta
        self.para_decay = decay

        self.para_alpha_ccc = alpha_ccc
        self.para_ccc = ccc

        Q = np.ones((2, 2), dtype=np.float64) * 0.5
        self.QResult = pd.DataFrame()
        self.PEResult = pd.DataFrame()
        self.PResult = []
        self.stimPosit = []
        self.respPosit = []
        stimPosit, respPosit = [], []
        for m in np.arange(0, self.trialNum):

            #  0:left, 1:right
            stimPosit, respPosit = int(self.data['stimPosit'][m]), int(self.data['respPosit'][m])
            oppAction = 1 - respPosit

            # softmax
            P = np.exp(beta * Q[stimPosit, respPosit]) / (
                        np.exp(beta * Q[stimPosit, respPosit]) + np.exp(beta * Q[stimPosit, oppAction]))

            # update
            if nblock > 1:
                # mltipule learning rates for different blocks
                block_num = int(self.data['nblock'][m])
                alpha_tmp = alpha[block_num - 1]
            else:
                alpha_tmp = alpha

            conflict = 2 * P - 1

            if conflict < ccc:
                alpha_tmp = alpha_ccc


            pe = np.ones((2, 2), dtype=np.float64) * 0
            pe[stimPosit, respPosit] = 1 - Q[stimPosit, respPosit]
            Q[stimPosit, respPosit] = Q[stimPosit, respPosit] + alpha_tmp * pe[stimPosit, respPosit]

            pe[stimPosit, oppAction] = 0 - Q[stimPosit, oppAction]
            Q[stimPosit, oppAction] = Q[stimPosit, oppAction] + alpha_tmp * pe[stimPosit, oppAction]

            # decay
            opp_stimPosit = 1 - stimPosit
            Q[opp_stimPosit, 0] = Q[opp_stimPosit, 0] + decay * (0.5 - Q[opp_stimPosit, 0])
            Q[opp_stimPosit, 1] = Q[opp_stimPosit, 1] + decay * (0.5 - Q[opp_stimPosit, 1])

            self.PResult.append(P)
            self.QResult = pd.concat([self.QResult, pd.DataFrame(Q.reshape(-1, Q.size))])
            self.PEResult = pd.concat([self.PEResult, pd.DataFrame(pe.reshape(-1, pe.size))])
            self.stimPosit.append(stimPosit)
            self.respPosit.append(respPosit)

    def SR_learning_observer_nblock(self, x, nblock=1):
        alpha = x[0:nblock]
        beta = x[nblock]
        decay = x[nblock+1]                          
        x0 = [alpha, beta, decay]
        self.SR_learning_observer(x0, nblock) 
        
    def SR_learning_observer_doubleUpdate(self, x):
        alpha, beta = x[0], x[1]
              
        self.para_alpha = alpha
        self.para_beta = beta
        Q = np.ones((2, 2), dtype=np.float64) * 0.5
        self.QResult = pd.DataFrame()
        self.PEResult = pd.DataFrame()
        self.PResult = []
        stimPosit, respPosit = [], []
        for m in np.arange(0, self.trialNum):
                       
            #  0:left, 1:right             
            stimPosit, respPosit = int(self.data['stimPosit'][m]), int(self.data['respPosit'][m])
            oppAction = 1-respPosit
           
            #softmax               
            P = np.exp(beta * Q[stimPosit, respPosit]) / (np.exp(beta * Q[stimPosit, respPosit]) + np.exp(beta * Q[stimPosit, oppAction]))
           
            #update                    
            pe = np.ones((2, 2), dtype=np.float64) * 0
            pe[stimPosit, respPosit] = 1 -  Q[stimPosit, respPosit]
            Q[stimPosit, respPosit] = Q[stimPosit, respPosit] + alpha * pe[stimPosit, respPosit]
                           
            pe[stimPosit, oppAction] = 0 - Q[stimPosit, oppAction]
            Q[stimPosit, oppAction] = Q[stimPosit, oppAction]+ alpha*pe[stimPosit, oppAction]
                          
            self.PResult.append(P)
            self.QResult = pd.concat([self.QResult, pd.DataFrame(Q.reshape(-1,Q.size))])
            self.PEResult = pd.concat([self.PEResult, pd.DataFrame(pe.reshape(-1,pe.size))])
            self.stimPosit.append(stimPosit)
            self.respPosit.append(respPosit)                            
                                                                    
    def incongruent_abstract_learning_operator(self, x, x_error='', nblock=1):
        alpha, beta = x[0], x[1]
        alpha_error = x_error
                      
        if np.array(alpha).size != nblock:
            print('numbers of alpha must equal to that of nblock')
            sys.exit(1)
       
        Q = np.array(0.5)
        self.para_alpha = alpha
        self.para_alpha_error = alpha_error
        self.para_beta = beta
        
        self.PResult = []
        self.QResult = pd.DataFrame()
        self.PEResult = pd.DataFrame()
        
        for m in np.arange(0, self.trialNum):
            #1: incongruent, 0:congruent
            congruency = int(self.data['congruency'][m])
            correction = int(self.data['correction'][m])
            
            # note: cogruency is a actual action value
            if correction == 1:
                congruency = congruency
            elif correction == 0:
                congruency = 1 - congruency
           
            #softmax
            P = np.exp(beta * Q)/(np.exp(beta* Q) + np.exp(beta * (1 - Q)))
           
            #update
            pe = np.array(congruency - Q) 
            
            
            if nblock>1:
                #mltipule learning rates for different blocks
                block_num = int(self.data['nblock'][m])
               
                #different learning rates for correct or incorrect resp if needed
                if alpha_error != '':
                    if correction == 1:
                        Q = Q + alpha[block_num-1] * pe
                    elif correction == 0:
                        Q = Q + alpha_error[block_num-1] * pe
                else:
                    Q = Q + alpha[block_num-1] * pe 
                          
            else:
                if alpha_error != '':
                    if correction == 1:
                        Q = Q + alpha * pe
                    elif correction == 0:
                        Q = Q + alpha_error * pe
                else:
                    Q = Q + alpha * pe
                           
                           
            self.PResult.append(P)
            self.QResult = pd.concat([self.QResult, pd.DataFrame(Q.reshape(-1,Q.size))])
            self.PEResult = pd.concat([self.PEResult, pd.DataFrame(pe.reshape(-1,pe.size))])
    
                                  
    def incongruent_abstract_learning_operator_nblock(self, x, x_error='', nblock=1):
        alpha = x[0:nblock]
        beta = x[nblock]  
        x0 = [alpha, beta]
        self.incongruent_abstract_learning_operator(x0, x_error, nblock)                              

    def SR_learning_operator(self, x, x_error='', nblock=1):
        alpha, beta, decay = x[0], x[1], x[2]     
        alpha_error = x_error
                           
        if np.array(alpha).size != nblock:
            print('numbers of alpha must equal to that of nblock')
            sys.exit(1)
       
        Q = np.ones((2, 2), dtype=np.float64) * 0.5
        self.para_alpha = alpha
        self.para_alpha_error = alpha_error
        self.para_beta = beta
        self.para_decay = decay
        
        self.PResult = []
        self.QResult = []
        self.PEResult = []
        self.stimPosit, self.respPosit = [], []
        
        for m in np.arange(0, self.trialNum):
           
            #  0:left, 1:right
            # respPosit is actual response location
            stimPosit, respPosit = int(self.data['stimPosit'][m]), int(self.data['hand'][m])
            correction = int(self.data['correction'][m])
            oppAction = 1-respPosit
           
            #softmax              
            P = np.exp(beta * Q[stimPosit, respPosit])/ (np.exp(beta * Q[stimPosit, respPosit]) + np.exp(beta * Q[stimPosit, oppAction]))
           
            #update   
            if nblock>1:
                #mltipule learning rates for different blocks
                block_num = int(self.data['nblock'][m])
               
                #different learning rates for correct or incorrect resp if needed
                if alpha_error != '':
                    if correction == 1:
                        alpha_tmp = alpha[block_num-1]*P
                    elif correction == 0:
                        alpha_tmp = alpha_error[block_num-1]
                else:
                    alpha_tmp = alpha[block_num-1]*P 
                          
            else:
                if alpha_error != '':
                    if correction == 1:
                        alpha_tmp = alpha*P
                    elif correction == 0:
                        alpha_tmp = alpha_error
                else:
                    alpha_tmp = alpha*P
           
            pe = np.ones((2, 2), dtype=np.float64) * 0
            pe[stimPosit, respPosit] = correction -  Q[stimPosit, respPosit]
            Q[stimPosit, respPosit] = Q[stimPosit, respPosit] + alpha_tmp * pe[stimPosit, respPosit]
                           
            pe[stimPosit, oppAction] = (1 - correction) - Q[stimPosit, oppAction]
            Q[stimPosit, oppAction] = Q[stimPosit, oppAction]+ alpha_tmp*pe[stimPosit, oppAction]
           
            # decay
            opp_stimPosit = 1 - stimPosit
            Q[opp_stimPosit, 0] = Q[opp_stimPosit, 0] + decay*(0.5 - Q[opp_stimPosit, 0])               
            Q[opp_stimPosit, 1] = Q[opp_stimPosit, 1] + decay*(0.5 - Q[opp_stimPosit, 1])               
           
                           
            self.PResult.append(P)               
            self.QResult.append(Q)
            self.PEResult.append(pe)
            self.stimPosit.append(stimPosit)
            self.respPosit.append(respPosit)
    
    def SR_learning_operator_ccc(self, x, x_error='', nblock=1):
        alpha, beta, decay, alpha_ccc, ccc = x[0], x[1], x[2], x[3], x[4]     
        alpha_error = x_error
                           
        if np.array(alpha).size != nblock:
            print('numbers of alpha must equal to that of nblock')
            sys.exit(1)
       
        Q = np.ones((2, 2), dtype=np.float64) * 0.5
        self.para_alpha = alpha
        self.para_alpha_error = alpha_error
        self.para_beta = beta
        self.para_decay = decay
        
        self.PResult = []
        self.QResult = pd.DataFrame()
        self.PEResult = pd.DataFrame()
        self.stimPosit, self.respPosit = [], []
        
        for m in np.arange(0, self.trialNum):
           
            #  0:left, 1:right
            # respPosit is actual response location
            stimPosit, respPosit = int(self.data['stimPosit'][m]), int(self.data['hand'][m])
            correction = int(self.data['correction'][m])
            oppAction = 1-respPosit
           
            #softmax              
            P = np.exp(beta * Q[stimPosit, respPosit])/ (np.exp(beta * Q[stimPosit, respPosit]) + np.exp(beta * Q[stimPosit, oppAction]))
            
            conflict = 2 * P - 1
            
            # 0 is wrong
            #update
            if alpha_error != '':
                if correction == 1:
                    alpha_tmp = alpha
                elif correction == 0:
                    if conflict < ccc:
                        alpha_tmp = alpha_ccc
                    else:
                        alpha_tmp = alpha_error
            else:
                alpha_tmp = alpha

            pe = np.ones((2, 2), dtype=np.float64) * 0
            pe[stimPosit, respPosit] = correction -  Q[stimPosit, respPosit]
            Q[stimPosit, respPosit] = Q[stimPosit, respPosit] + alpha_tmp * pe[stimPosit, respPosit]
                           
            pe[stimPosit, oppAction] = (1 - correction) - Q[stimPosit, oppAction]
            Q[stimPosit, oppAction] = Q[stimPosit, oppAction]+ alpha_tmp*pe[stimPosit, oppAction]
           
            # decay
            opp_stimPosit = 1 - stimPosit
            Q[opp_stimPosit, 0] = Q[opp_stimPosit, 0] + decay*(0.5 - Q[opp_stimPosit, 0])               
            Q[opp_stimPosit, 1] = Q[opp_stimPosit, 1] + decay*(0.5 - Q[opp_stimPosit, 1])               
                          
            self.PResult.append(P)               
            self.QResult = pd.concat([self.QResult, pd.DataFrame(Q.reshape(-1,Q.size))])
            self.PEResult = pd.concat([self.PEResult, pd.DataFrame(pe.reshape(-1,pe.size))])
            self.stimPosit.append(stimPosit)
            self.respPosit.append(respPosit)
    
    
    def SR_learning_operator_nblock(self, x, x_error='', nblock=1):
        alpha = x[0:nblock]
        beta = x[nblock] 
        decay = x[nblock+1]
        x0 = [alpha, beta, decay]
        self.SR_learning_operator(x0, x_error, nblock)
    def SR_learning_operator_doubleUpdate(self, x, x_error=''):
        alpha, beta = x[0], x[1]   
        alpha_error = x_error
     
        Q = np.ones((2, 2), dtype=np.float64) * 0.5
        self.para_alpha = alpha
        self.para_alpha_error = alpha_error
        self.para_beta = beta
        
        self.PResult = []
        self.QResult = pd.DataFrame()
        self.PEResult = pd.DataFrame()
        self.stimPosit, self.respPosit = [], []
        
        for m in np.arange(0, self.trialNum):
           
            #  0:left, 1:right
            # respPosit is actual response location
            stimPosit, respPosit = int(self.data['stimPosit'][m]), int(self.data['hand'][m])
            correction = int(self.data['correction'][m])
            oppAction = 1-respPosit
           
            #softmax              
            P = np.exp(beta * Q[stimPosit, respPosit])/ (np.exp(beta * Q[stimPosit, respPosit]) + np.exp(beta * Q[stimPosit, oppAction]))
           
            #update                                        
            if alpha_error != '':
                if correction == 1:
                    alpha_tmp = alpha
                elif correction == 0:
                    alpha_tmp = alpha_error
            else:
                alpha_tmp = alpha
           
            pe = np.ones((2, 2), dtype=np.float64) * 0
            pe[stimPosit, respPosit] = correction -  Q[stimPosit, respPosit]
            Q[stimPosit, respPosit] = Q[stimPosit, respPosit] + alpha_tmp * pe[stimPosit, respPosit]
                           
            pe[stimPosit, oppAction] = (1 - correction) - Q[stimPosit, oppAction]
            Q[stimPosit, oppAction] = Q[stimPosit, oppAction]+ alpha_tmp*pe[stimPosit, oppAction]
                                      
            self.PResult.append(P)               
            self.QResult = pd.concat([self.QResult, pd.DataFrame(Q.reshape(-1,Q.size))])
            self.PEResult = pd.concat([self.PEResult, pd.DataFrame(pe.reshape(-1,pe.size))])
            self.stimPosit.append(stimPosit)
            self.respPosit.append(respPosit)
    
    def save_data(self):
        best_model_regression(self)
        
        
                                  

