import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from utils import *
from detection_model import Net
from sklearn import preprocessing
import torch.utils.data as data_utils
import torch.optim as optim

if __name__=='__main__':
    for epo in range(1,200):
        labels = np.load("labels.npy")
        pres = np.load('mean_distence/epo_%d_mean_distence.npy' % epo)
        threshold_T = np.zeros([labels.shape[0],32])
        threshold_F = np.zeros([labels.shape[0],32])
        threshold =  np.zeros([32])
        count1=0
        count2=0
        mea1=0
        mea2=0
        
        for i in range(labels.shape[0]):
            if labels[i]==0:
                threshold_F[i] = np.abs(pres[i][0]).max(axis=0)
            else:
                threshold_T[i] = np.abs(pres[i][0]).max(axis=0)
        # print('T_max:',threshold_T.max(axis=0),'\n', "F_max:",threshold_F.max(axis=0))
        # threshold = threshold_F.max(axis=0)
        P_value = []
        R_value = []
        F1_value = []
        for j in range(1, 501, 10):
            threshold = np.sort(threshold_F,axis=0)[-1 * j,:]
            TP = 0
            TN = 0
            FP = 0
            FN = 0
            for i in range(labels.shape[0]):
                if labels[i]==0:
                    if np.sum((np.abs(pres[i][0]).max(axis=0) - threshold)>0) > 0:
                        FP += 1
                    else:
                        TN += 1
                if labels[i]==1:
                    if np.sum((np.abs(pres[i][0]).max(axis=0) - threshold)>0) > 0:
                        TP += 1
                    else:
                        FN += 1
            p = TP / (TP + FP)
            r = TP / (TP + FN)
            f1 = (p * r) / (p + r) * 2
            P_value.append(p)
            R_value.append(r)
            F1_value.append(f1)
        max_f1_value = max(F1_value) # 求列表最大值
        max_idx = F1_value.index(max_f1_value) # 求最大值对应索引
        with open('modified_smd_result.txt', 'a') as f:    
            # print('epoch:This is epoch {}', epo+1, ' P:', P_value[max_idx],' R:',R_value[max_idx],' F1:', max_f1_value)
            print('epoch:This is epoch {}, the P is {}, the R is {}, the F1 is {}'.format(epo+1, P_value[max_idx], R_value[max_idx], max_f1_value), file=f)  
        
