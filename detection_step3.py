import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from utils import *
from att_u_net import Net
# from u_net import Net
from sklearn import preprocessing
import torch.utils.data as data_utils
import torch.optim as optim

# if __name__ == '__main__':  
#     for epo in range(1,51):
#         labels = np.load("labels_modified.npy")  
#         pres = np.load('mean_distence/epo_%d_mean_distence.npy' % epo)  
#         threshold_T = np.zeros([labels.shape[0], 48]) 
#         threshold_F = np.zeros([labels.shape[0], 48])  
#         threshold = np.zeros([48])  
#         count1 = 0  
#         count2 = 0  
#         mea1 = 0  
#         mea2 = 0  
  
#         for i in range(labels.shape[0]):  
#             if labels[i] == 0:  
#                 count1 += 1  
#                 mea1 += np.sum(np.abs(pres[i][0]))  
#             else:  
#                 count2 += 1  
#                 mea2 += np.sum(np.abs(pres[i][0]))  
  
#         if count1 == 0 or count2 == 0:  
#             print("Warning: One or both label categories have no data in epoch", epo)  
#             continue  # Skip this epoch if there is no data for one of the categories  
  
#         mea1 /= count1  
#         mea2 /= count2  
  
#         P_value = []  
#         R_value = []  
#         F1_value = []  
#         if mea1 < mea2:  # Ensure there is a valid range  
#             for t in np.arange(mea1, mea2, step=0.1):  
#                 TP = 0  
#                 TN = 0  
#                 FP = 0  
#                 FN = 0  
#                 for i in range(labels.shape[0]):  
#                     if labels[i] == 0:  
#                         if np.sum(np.abs(pres[i][0])) - t > 0:  
#                             FP += 1
#                         else:  
#                             TN += 1  
#                     else:  
#                         if np.sum(np.abs(pres[i][0])) - t > 0:  
#                             TP += 1  
#                         else:  
#                             FN += 1  
#                 if (TP + FP) > 0 and (TP + FN) > 0:  # Avoid division by zero  
#                     p = TP / (TP + FP)  
#                     r = TP / (TP + FN)  
#                     f1 = (2 * p * r) / (p + r)  # Corrected F1 formula  
#                     P_value.append(p)  
#                     R_value.append(r)  
#                     F1_value.append(f1)  

#         if F1_value:  # Check if F1_value is not empty  
#             max_f1_value = max(F1_value)  
#             max_idx = F1_value.index(max_f1_value) 
#             with open('ablation_result.txt', 'a') as f:  # 修正了单引号 
#                 print('epoch:This is epoch {}, the P is {}, the R is {}, the F1 is {}'.format(epo+1, P_value[max_idx], R_value[max_idx], max_f1_value), file=f)
#                 # print('epoch:', epo + 1, ' P:', P_value[max_idx], ' R:', R_value[max_idx], ' F1:', max_f1_value)  
#         else:
#             with open('ablation_result.txt', 'a') as f:  # 修正了单引号  
#                 print(("No valid F1 scores computed for epoch", epo), file=f)
if __name__=='__main__':
    # for epo in range(1,11):
    #     labels = np.load("labels_modified.npy")
    #     pres = np.load('mean_distence/epo_%d_mean_distence.npy' % epo)
    #     threshold_T = np.zeros([labels.shape[0],48])
    #     threshold_F = np.zeros([labels.shape[0],48])
    #     threshold =  np.zeros([48])
    #     for i in range(labels.shape[0]):
    #         # if labels[i]==0:
    #         #     threshold_F[i] = np.abs(pres[i][0]).max(axis=0)
    #         # else:
    #         #     threshold_T[i] = np.abs(pres[i][0]).max(axis=0)
    #         threshold_F[i] = np.abs(pres[i][0]).mean(axis=0)
    #     # print('T_max:',threshold_T.max(axis=0),'\n', "F_max:",threshold_F.max(axis=0))
    #     # threshold = threshold_F.max(axis=0)
    #     P_value = []
    #     R_value = []
    #     F1_value = []
    #     # for j in range(1, 501, 10):
    #     for j in range(1, 1001, 10):
    #         threshold = np.sort(threshold_F,axis=0)[-1 * j,:]
    #         # threshold = np.sort(threshold_F,axis=0)[j,:]
    #         TP = 0
    #         TN = 0
    #         FP = 0
    #         FN = 0
    #         for i in range(labels.shape[0]):
    #             if labels[i]==0:
    #                 if np.sum((np.abs(pres[i][0]).max(axis=0) - threshold)>0) > 0:
    #                     FP += 1
    #                 else:
    #                     TN += 1
    #             if labels[i]==1:
    #                 if np.sum((np.abs(pres[i][0]).max(axis=0) - threshold)>0) > 0:
    #                     TP += 1
    #                 else:
    #                     FN += 1
    #         p = TP / (TP + FP)
    #         r = TP / (TP + FN)
    #         f1 = (p * r) / (p + r) * 2
    #         P_value.append(p)
    #         R_value.append(r)
    #         F1_value.append(f1)
    #     max_f1_value = max(F1_value) # 求列表最大值
    #     max_idx = F1_value.index(max_f1_value) # 求最大值对应索引
    #     print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)
    #     with open('ablation_result.txt', 'a') as f:  # 修正了单引号 
    #         print('epoch:This is epoch {}, the P is {}, the R is {}, the F1 is {}'.format(epo+1, P_value[max_idx], R_value[max_idx], max_f1_value), file=f)
    #     # print('epoch:', epo+1, ' P:', P_value[max_idx],' R:',R_value[max_idx],' F1:', max_f1_value)
    # for epo in range(1, 11):  
    #     labels = np.load("labels_modified.npy")  
    #     pres = np.load(f'mean_distence/epo_{epo}_mean_distence.npy')  
      
    #     threshold_F = np.abs(pres[:, 0]).mean(axis=1)  # 计算每个样本的平均预测值作为特征  
      
    #     best_f1 = 0  
    #     best_threshold = 0  
    #     best_p, best_r = 0, 0  
      
    #     thresholds = np.sort(threshold_F)[::-1]  # 对阈值进行排序（从大到小）  
    #     for threshold in thresholds:  
    #         TP, TN, FP, FN = 0, 0, 0, 0  
    #         for i, label in enumerate(labels):  
    #             pred = threshold_F[i] > threshold  # 使用平均预测值是否大于阈值作为预测结果  
    #             if label == 0 and not pred:  
    #                 TN += 1  
    #             elif label == 0 and pred:  
    #                 FP += 1  
    #             elif label == 1 and pred:  
    #                 TP += 1  
    #             elif label == 1 and not pred:  
    #                 FN += 1  
          
    #         p = TP / (TP + FP) if (TP + FP) > 0 else 0  
    #         r = TP / (TP + FN) if (TP + FN) > 0 else 0  
    #         f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0  
          
    #         if f1 > best_f1:  
    #             best_f1 = f1  
    #             best_threshold = threshold  
    #             best_p, best_r = p, r  
      
    #     print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)  
    #     print(f'epoch: This is epoch {epo+1}, the best P is {best_p}, the best R is {best_r}, the best F1 is {best_f1}, with threshold {best_threshold}')  
    #     with open('ablation_result.txt', 'a') as f:  
    #         f.write(f'epoch: This is epoch {epo+1}, the best P is {best_p}, the best R is {best_r}, the best F1 is {best_f1}, with threshold {best_threshold}\n')

    for epo in range(2, 11):  
        labels = np.load("labels_modified.npy")  
        pres = np.load(f'mean_distence/epo_{epo}_mean_distence.npy')
        print(pres.shape)
        # pres = np.sum(np.abs(pres), axis=(-3,-2,-1))
        pres = np.linalg.norm(pres, axis=-1, ord=2)
        pres = np.linalg.norm(pres, axis=-1, ord=2)
        pres = np.linalg.norm(pres, axis=-1, ord=2)
        print(pres.shape)
        
        max_threshold = pres.max()
        min_threshold = pres.min()
        print(max_threshold, min_threshold)
        for threshold in np.arange(min_threshold, max_threshold, 1):
            TP, TN, FP, FN = 0, 0, 0, 0  
            for j in range(len(pres)):
                pred = pres[j] > threshold
                print(pres[j].shape)
                # coding
                if labels[j] == 0 and not pred:  
                    TN += 1  
                elif labels[j] == 0 and pred:  
                    FP += 1  
                elif labels[j] == 1 and pred:  
                    TP += 1  
                elif labels[j] == 1 and not pred:  
                    FN += 1  
            p = TP / (TP + FP) if (TP + FP) > 0 else 0  
            r = TP / (TP + FN) if (TP + FN) > 0 else 0  
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0 
            with open('ablation_result.txt', 'a') as f:  # 修正了单引号  
                print('epoch: This is epoch {}, the P is {}, the R is {}, the F1 is {}'.format(epo+1, p, r, f1), file=f) 
            # print(f'epoch: This is epoch {epo+1}, the P is {p}, the R is {r}, the F1 is {f1}', '  TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)  
        
                

            
    
      
    # # 假设 pres 的形状是 (n_samples, n_features)，我们只使用第一个特征的平均绝对值作为特征  
    #     threshold_F = np.abs(pres[:, 0]).mean(axis=0)  # 注意这里使用 axis=0，但因为我们只取了一列，所以其实没影响  
    # # 但是，我们不会直接使用 threshold_F 中的值作为阈值  
    #     print(threshold_F.shape)
    # # 创建一个新的阈值范围来搜索  
    #     min_val, max_val = threshold_F.min() - 1, threshold_F.max() + 1  # 可以根据需要调整范围  
    #     thresholds = np.linspace(min_val, max_val, 100)  # 生成 100 个均匀分布的阈值  
      
    #     best_f1 = 0  
    #     best_threshold = 0  
    #     best_p, best_r = 0, 0  
      
    #     for threshold in thresholds:  
    #         TP, TN, FP, FN = 0, 0, 0, 0  
    #         for i, label in enumerate(labels): 

    #             print("label:", label, "pred:",pres[i, 0])
    #         # 使用第一个特征的平均绝对值是否大于阈值作为预测结果  
    #         # 注意：这里我们使用的是 pres[i, 0] 的绝对值，然后取平均（但在这个循环里我们其实只需要 pres[i, 0] 的绝对值）  
    #         # 因为 threshold_F 已经是 pres[:, 0] 的平均绝对值了，所以这里我们应该直接使用 pres[i, 0] 的绝对值进行比较  
    #         # 但是为了保持一致性（尽管在这个特定情况下是冗余的），我们还是用 threshold_F[i]（它其实就等于 pres[i, 0] 的绝对值，如果 pres 只有一列的话）  
    #         # 注意：这里的 threshold_F[i] 应该是标量，因为 pres[:, 0] 是一维数组  
    #             pred = np.abs(pres[i, 0, ]) > threshold  # 修改这里，使用绝对值进行比较  
    #         # 注意：如果 pres 有多列，并且您想要使用多列的特征来做出预测，  
    #         # 那么您需要一个不同的方法来组合这些特征成一个单一的分数或决策。  
                
              
    #             if label == 0 and not pred:  
    #                 TN += 1  
    #             elif label == 0 and pred:  
    #                 FP += 1  
    #             elif label == 1 and pred:  
    #                 TP += 1  
    #             elif label == 1 and not pred:  
    #                 FN += 1  
          
    #         p = TP / (TP + FP) if (TP + FP) > 0 else 0  
    #         r = TP / (TP + FN) if (TP + FN) > 0 else 0  
    #         f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0  
          
    #         if f1 > best_f1:  
    #             best_f1 = f1  
    #             best_threshold = threshold  
    #             best_p, best_r = p, r  
      
    #     print('TP:', TP, 'FP:', FP, 'TN:', TN, 'FN:', FN)  
    #     print(f'epoch: This is epoch {epo+1}, the best P is {best_p}, the best R is {best_r}, the best F1 is {best_f1}, with threshold {best_threshold}')  
    #     with open('ablation_result.txt', 'a') as f:  
    #         f.write(f'epoch: This is epoch {epo+1}, the best P is {best_p}, the best R is {best_r}, the best F1 is {best_f1}, with threshold {best_threshold}\n')
        

        
