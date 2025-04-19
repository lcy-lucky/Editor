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
from TCN_GCN import GCNTCN
from GCN import GCN

attack = pd.read_csv("../input/SWaT_Dataset_Attack_v0.csv",sep=';')
attack = attack.head(40000)
# attack = pd.read_csv("../input/SWaT_Dataset_Attack_v0.csv",sep=';',skiprows=lambda x: x > 0 and x % 100 != 0)#, nrows=1000)
attack = attack.drop(index=attack.loc[(attack['Normal/Attack']=='Attack')].index,axis=1)
# attack = attack.drop(attack.loc[attack["Normal/Attack"] == "Attack"].index, inplace=True)
attack = attack.drop(["Timestamp" , "Normal/Attack"] , axis=1)
attack = attack.drop(attack.columns[[4, 10, 11, 13, 15, 29, 31, 32, 43, 48, 50]], axis=1)
print(attack.shape)
for i in list(attack): 
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)
min_max_scaler = preprocessing.MinMaxScaler()
attack_x = attack.values
n2 = pd.DataFrame(attack_x)

attack_x_scaled = min_max_scaler.fit_transform(attack_x)
attack = pd.DataFrame(attack_x_scaled)
windows_attack = attack.values[np.arange(12)[None, :] + np.arange(attack.shape[0]-12)[:, None]]
test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float()), batch_size=128, shuffle=False, num_workers=0)


#Read normal data
normal = pd.read_csv("../input/SWaT_Dataset_Normal_v1.csv",skiprows=lambda x: x > 0 and x % 10 != 0)#, nrows=1000)
normal = normal.head(40000)
normal = normal.drop(["Timestamp" , "Normal/Attack"] , axis=1)
normal = normal.drop(normal.columns[[4, 10, 11, 13, 15, 29, 31, 32, 43, 48, 50]], axis=1)
print(normal.shape)
# Transform all columns into float64
for i in list(normal): 
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)
# normal = normal.head(20000)

min_max_scaler = preprocessing.MinMaxScaler()
x = normal.values
n1 = pd.DataFrame(x)


x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)

# print(normal.shape)


BATCH_SIZE = 128
DEVICE = 0

window_size = 12
windows_normal = normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]

# windows_normal = np.array(normal).reshape([1300,36,40])
# print(windows_normal.shape)



windows_normal_train = windows_normal[:int(np.floor(1. *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.7 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float()), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


# def generate_mask(x):
#     for j in range(x.shape[0]):
#         L1 = random.sample(range(1, x.shape[2]), 10)
#         for k in L1:
#             x[j,0,k] = 0    
#     return x
def normalize(mx):
    rowsum = np.array(mx.sum(1)) #会得到一个（2708,1）的矩阵
    r_inv = np.power(rowsum, -1).flatten() #得到（2708，）的元祖
    #在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


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
    
    # adj = torch.zeros(40, 40)
    # corr = np.abs(n1.corr().values)
    # # print(corr)
    # for j in range(corr.shape[0]):
    #     for k in range(corr.shape[1]):
    #         if abs(corr[j,k]) < 0.9:
    #             corr[j,k]=0
    #         else:
    #             corr[j,k]=1
    #         if k == j:
    #             corr[j,k]=0
    # adj = torch.tensor(corr).to(DEVICE)
    adj = torch.ones(12,12)
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if i==j:
                adj[i,j]=0
    # adj = np.zeros(shape=[12,12])
    # for i in range(adj.shape[0]):
    #     for j in range(adj.shape[1]):
    #         if j==i-1 and 0<=i-1<=11:
    #             adj[i,j]=1
    #         if j==i+1 and 0<=i+1<=11:
    #             adj[i,j]=1
            
    #         if j==i-2 and 0<=i-2<=11:
    #             adj[i,j]=0.8
    #         if j==i+2 and 0<=i+2<=11:
    #             adj[i,j]=0.8
            
    #         if j==i-3 and 0<=i-3<=11:
    #             adj[i,j]=0.6
    #         if j==i+3 and 0<=i+3<=11:
    #             adj[i,j]=0.6
                        
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
    
    # model = GCNTCN(gcn_nfeat=40, gcn_nhid1=120, gcn_nhid2=60, gcn_class=40, gcn_dropout=0.2, 
    #                tcn_num_inputs=40, tcn_num_channels=[80,40], tcn_kernel_size=3, tcn_dropout=0.3).to(DEVICE)
    # model = GCNTCN(gcn_nfeat=40, gcn_nhid1=120, gcn_class=40, gcn_dropout=0.2, 
    #                tcn_num_inputs=40, tcn_num_channels=[80,40], tcn_kernel_size=3, tcn_dropout=0.3).to(DEVICE)
    model = GCNTCN(gcn_nfeat=40, gcn_nhid1=400, gcn_nhid2=800, gcn_nhid3=400, gcn_class=40, gcn_dropout=0.2, 
                   tcn_num_inputs=40, tcn_num_channels=[120,40], tcn_kernel_size=3, tcn_dropout=0.3).to(DEVICE)
    # model = GCN(nfeat=40, nhid1=120, nhid2=60, nclass=40, dropout=0.2).to(DEVICE)
    # model.load_state_dict(torch.load("model_para/trantcn_96_epo.pth"))
    model = model.train()
    criterion = torch.nn.L1Loss()
    mae_func = torch.nn.L1Loss(reduction='sum')
    mse_func = torch.nn.MSELoss(reduction='sum')
    # optimizer = optim.SGD(model.parameters(), lr = 1e-3, momentum = 0.01)
    # optimizer = torch.optim.Adam(model.parameters(),lr=2e-3,betas=(0.9,0.99))
    optimizer = torch.optim.Adam(model.parameters(),lr=8e-3,betas=(0.9,0.99))
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
            # train_mask = train_mask.to(DEVICE)
            # print(train_mask.shape, mask[i].shape)
            # pre = model(x, train_mask, adj)
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
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # torch.save(model.state_dict(), 'swat_gat_tcn/gat_tcn_{epo}_epo.pth'.format(epo=epoch))
        with open('size80_swat_gtn_result.txt', 'a') as f:  # 修正了单引号 
            print('train tcn_gcn_swat:This is epoch {}, the loss is {}, the mae is {}, the mse is {}, the mre is {}'.format(epoch, loss_sum, mae_sum/count/BATCH_SIZE/window_size/10, mse_sum/count/BATCH_SIZE/window_size/10, mre_sum/count), file=f)
            print('test tcn_gcn_swat:This is epoch {}, the loss is {}, the mae is {}, the mse is {}, the mre is {}'.format(epoch, loss_test_sum, mae_test_sum/count_test/BATCH_SIZE/window_size/10, mse_test_sum/count_test/BATCH_SIZE/window_size/10, mre_test_sum/count_test), file=f)
        # print('train tcn_gcn_swat:This is epoch {epo}, the loss is {loss}, the mae is {mae}, the mse is {mse}, the mre is {mre}'.format(epo=epoch, loss=loss_sum, mae=mae_sum/count/BATCH_SIZE/window_size/10, mse=mse_sum/count/BATCH_SIZE/window_size/10, mre=mre_sum/count))
        # print('test tcn_gcn_swat:This is epoch {epo}, the loss is {loss}, the mae is {mae}, the mse is {mse}, the mre is {mre}'.format(epo=epoch, loss=loss_test_sum, mae=mae_test_sum/count_test/BATCH_SIZE/window_size/10, mse=mse_test_sum/count_test/BATCH_SIZE/window_size/10, mre=mre_test_sum/count_test))
        # print('train gcn_add_lr:This is epoch {epo}, the loss is {loss}, the mse is {mse}, the mre is {mre}'.format(epo=epoch, loss=loss_sum, mse=mse_sum/count/BATCH_SIZE/window_size/10, mre=mre_sum/count))
        # print('test gcn_add_lr:This is epoch {epo}, the loss is {loss}, the mse is {mse}, the mre is {mre}'.format(epo=epoch, loss=loss_test_sum, mse=mse_test_sum/count_test/BATCH_SIZE/window_size/10, mre=mre_test_sum/count_test))        
        # print(optimizer.state_dict()['param_groups'][0]['lr'])
        # torch.save(model.state_dict(), 'tcn_model_para/gat_tcn_{epo}_epo.pth'.format(epo=epoch))
        # print('TranTcn:This is epoch {epo}, the loss is {loss}, the mse is {mse}, the mre is {mre}'.format(epo=epoch, loss=loss_sum, mse=mse_sum/count/BATCH_SIZE/window_size/10, mre=mre_sum/count))
