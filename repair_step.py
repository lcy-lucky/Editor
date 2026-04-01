import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from utils import *
from sklearn import preprocessing
import torch.utils.data as data_utils
import torch.optim as optim
import random
import scipy.sparse as sp
# from TCN_GCN import TransfomerTCN
from repair_model import bfGCNTCN
# from GCN import GCN

attack = pd.read_csv("../input/save-machine-testlabel-1-1.csv",sep=',')

attack = attack.drop(index=attack.loc[(attack['Normal/Attack']=='1')].index,axis=0)

attack = attack.drop(["col5","col8","col17","col18","col27","col29","Normal/Attack"] , axis=1)

attack = attack.astype(float)
attack_x = attack.values

attack = pd.DataFrame(attack_x)
print(attack.shape)
windows_attack = attack.values[np.arange(32)[None, :] + np.arange(attack.shape[0]-32)[:, None]]
test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float()), batch_size=128, shuffle=False, num_workers=0)


#Read normal data
normal = pd.read_csv("../input/machine-train1-1.csv",sep=',')
normal = normal.drop(["col5","col8","col17","col18","col27","col29"], axis=1)
print(normal.shape)

normal = normal.astype(float)

x = normal.values
normal = pd.DataFrame(x)

print(normal.shape)


BATCH_SIZE = 128
DEVICE = 0

window_size = 32
windows_normal = normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]


windows_normal_train = windows_normal[:int(np.floor(1. *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.7 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()     
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

#Masking can be applied either in a predefined manner based on specific requirements or randomly
def generate_mask(x): 
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            L1 = random.sample(range(1, x.shape[2]), 10)
            for k in L1:
                x[i,j,k] = 0    
    return x


def generate_mask_new(x):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            L1 = random.sample(range(1, x.shape[3]), 5)
            for k in L1:
                x[i,j,0,k] = 0
    # print("x.shape is", x.shape)    
    return x

def get_mre(y_pre,y_true, mask):
    err = torch.abs(y_pre - y_true) * mask
    return err.sum() / ((y_true * mask).sum() + 1e-8)

def get_mre_all_graph(y_pre,y_true, mask):
    err = (torch.abs(y_pre - y_true) * mask).sum()
    sum = (y_true * mask).sum() 
    return err, sum

if __name__=='__main__':
    
    adj = torch.ones(32,32)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if i==j:
                adj[i,j]=0
    
                        
    adj = normalize(adj)
    adj = torch.tensor(adj, dtype=torch.float).to(DEVICE)
    # print(adj)
    # adj = torch.ones(size=[40,40]).to(DEVICE)
    # for j in range(adj.shape[0]):
    #     for k in range(adj.shape[1]):
    #         if k == j:
    #             adj[j,k]=0
    
    # mask = torch.ones(size=[windows_normal_train.shape[0],windows_normal_train.shape[1], 1, windows_normal_train.shape[2]]).to(DEVICE)
    # mask = generate_mask_new(mask)
    
    # model = bfGCNTCN(gcn_nfeat=32, gcn_nhid1=64, gcn_class=32, gcn_dropout=0.2, 
    #                tcn_num_inputs=32, tcn_num_channels=[64,32], tcn_kernel_size=3, tcn_dropout=0.3).to(DEVICE)
    model = bfGCNTCN(gcn_nfeat=32, gcn_nhid1=64, gcn_class=32, gcn_dropout=0.2, tcn_num_inputs=32, tcn_num_channels=[64,32], tcn_kernel_size=3, tcn_dropout=0.3).to(DEVICE)
    # model = GCNTCN(gcn_nfeat=40, gcn_nhid1=400, gcn_nhid2=800, gcn_nhid3=400, gcn_class=40, gcn_dropout=0.2, 
    #                tcn_num_inputs=40, tcn_num_channels=[120,40], tcn_kernel_size=3, tcn_dropout=0.3).to(DEVICE)
    # model = GCN(nfeat=40, nhid1=120, nhid2=60, nclass=40, dropout=0.2).to(DEVICE)
    # model.load_state_dict(torch.load("model_para/trantcn_96_epo.pth"))
    model = model.train()
    criterion = torch.nn.L1Loss()
    mae_func = torch.nn.L1Loss(reduction='sum')
    mse_func = torch.nn.MSELoss(reduction='sum')
  
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4,betas=(0.9,0.99))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[5,10],gamma = 0.9)
    # train_mask = torch.ones([int(BATCH_SIZE*window_size),1,40])
    # train_mask = generate_mask(train_mask).to(DEVICE)
    for epoch in range(200):
        loss_sum = 0
        loss_ori = 0
        mae_sum = 0
        mse_sum = 0
        mre_sum = 0
        count = 0
        
        loss_test_sum = 0
        mae_test_sum = 0
        mse_test_sum = 0
        mre_test_sum = 0
        count_test = 0
        
        for i, [x] in enumerate(train_loader):
            # x = x.view([int(x.shape[0]*x.shape[1]),1,x.shape[2]])
            x = x.to(DEVICE)
            train_mask = torch.ones_like(x)
            train_mask = generate_mask(train_mask)
            optimizer.zero_grad()
           
            pre = model(x, train_mask, adj)
            loss = criterion(pre, x)
            loss_sum += loss
            loss.backward()
            optimizer.step()
            mae = mae_func(pre, x)
            mae_sum += mae
            # print(mse)
            mse = mse_func(pre, x)
            mse_sum += mse
            # print(mse)
            mre = get_mre(pre, x, 1-train_mask)
            mre_sum += mre
            
            count += 1 
        scheduler.step()
        
        with torch.no_grad():
            for i, [x] in enumerate(test_loader):
                x = x.to(DEVICE)
                test_mask = torch.ones_like(x)
                test_mask = generate_mask(test_mask)
                optimizer.zero_grad()
                # print(x.shape,train_mask.shape)
                pre= model(x, test_mask,adj)
                loss_test = criterion(pre, x)
                loss_test_sum += loss_test
                # loss.backward()
                # optimizer.step()
                mae_test = mae_func(pre, x)
                mae_test_sum += mae_test
                
                mse_test = mse_func(pre, x)
                mse_test_sum += mse_test
                # print(mse)
                mre_test = get_mre(pre, x, 1-test_mask)
                mre_test_sum += mre_test
                count_test += 1         
        
        with open('bf_smd_result.txt', 'a') as f:  
            print('train bf_tcn_gcn_smd: This is epoch {}, the loss is {}, the mae is {}, the mse is {}, the mre is {}'.format(epoch, loss_sum, mae_sum/count/BATCH_SIZE/window_size/10, mse_sum/count/BATCH_SIZE/window_size/10, mre_sum/count), file=f)  
            print('test bf_tcn_gcn_smd: This is epoch {}, the loss is {}, the mae is {}, the mse is {}, the mre is {}'.format(epoch, loss_test_sum, mae_test_sum/count_test/BATCH_SIZE/window_size/10, mse_test_sum/count_test/BATCH_SIZE/window_size/10, mre_test_sum/count_test), file=f)
        
