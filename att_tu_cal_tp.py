import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from utils import *
# from att_u_net import Net
from att_u_net_bf import bfNet
# from u_net import Net
from sklearn import preprocessing
import torch.utils.data as data_utils
import torch.optim as optim

#Read normal data
normal = pd.read_csv("input/SWaT_Dataset_Normal_v1.csv")#, nrows=1000)
normal = normal.drop(["Timestamp" , "Normal/Attack", "P201", "AIT501", "P202"] , axis = 1)
print(normal.shape)

# Transform all columns into float64
for i in list(normal): 
    normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)

min_max_scaler = preprocessing.MinMaxScaler()
x = normal.values
x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x_scaled)

#Read attack data
attack = pd.read_csv("input/modified_dataset.csv",sep=",")#, nrows=1000)
labels = [ float(label!= 'Normal' ) for label  in attack["Normal/Attack"].values]
attack = attack.drop(["Normal/Attack"] , axis=1)
print(attack.shape)
# attack = attack.drop(["Timestamp" , "Normal/Attack", "P201", "AIT501", "P202"] , axis = 1)
# print('labels',labels)

# Transform all columns into float64
for i in list(attack):
    attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)
x = attack.values 
x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x_scaled)

window_size=48
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]

windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))
labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]
np.save('labels_modified.npy',labels)

BATCH_SIZE = 64
N_EPOCHS = 200
DEVICE = 0

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]

train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],
    1, windows_normal_train.shape[1], windows_normal_train.shape[2]]))), 
    batch_size=2500, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],
    1, windows_normal_val.shape[1], windows_normal_val.shape[2]]))), 
    batch_size=2500, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],
    1, windows_attack.shape[1], windows_attack.shape[2]]))),
    batch_size=2500, shuffle=False, num_workers=0)

def train():
    UNet1 = Net().to(DEVICE)
    UNet1.train()

    # UNet2 = Net().to(DEVICE)
    # UNet2.train()

    criterion = torch.nn.MSELoss()
    optimizer1 = optim.SGD(UNet1.parameters(), lr = 1e-3, momentum = 0.1)
    # optimizer2 = optim.SGD(UNet2.parameters(), lr = 1e-3, momentum = 0.1)
    for epo in range(0, N_EPOCHS):
        loss_sum1 = 0
        # loss_sum2 = 0
        for i, [batch] in enumerate(train_loader):
            batch = batch.to(DEVICE)
            optimizer1.zero_grad()
            out1 = UNet1(batch)
            loss1 = criterion(out1, batch)
            weight = nn.functional.softmax((out1 -batch).abs().detach(), dim=3)
            loss_sum1 += loss1
            loss1.backward()
            optimizer1.step()
            
            # optimizer2.zero_grad()
            # out2 = UNet2(batch+batch*weight)
            # loss2 = criterion(out2, batch)
            # loss_sum2 += loss2
            # loss2.backward()
            # optimizer2.step()

        print('Unet1:This is epoch {epo}, the loss is {loss}'.format(epo=epo, loss=loss_sum1))
        # print('Unet2:This is epoch {epo}, the loss is {loss}'.format(epo=epo, loss=loss_sum2))
        torch.save(UNet1.state_dict(), 'two_models_para/u_net1_{epo}_epo.pth'.format(epo=epo))
        # torch.save(UNet2.state_dict(), 'two_models_para/u_net2_{epo}_epo.pth'.format(epo=epo))


if __name__=='__main__':
    for epo in range(1,11):
        UNet1 = bfNet().to(DEVICE)
        UNet1.load_state_dict(torch.load("two_models_para/u_net1_%d_epo.pth" % epo))
        UNet1.eval()

        UNet2 = bfNet().to(DEVICE)
        UNet2.load_state_dict(torch.load("two_models_para/u_net2_%d_epo.pth" % epo))
        UNet2.eval()
        for i, [batch] in enumerate(test_loader):
            batch = batch.to(DEVICE)
            out1 = UNet1(batch)
            weight = nn.functional.softmax((out1 -batch).abs().detach(), dim=3)
            out2 = UNet2(batch+batch*weight)
            out2 = out2.detach().cpu().numpy()
            # out1 = out1.detach().cpu().numpy()
            batch = batch.detach().cpu().numpy()
            if i==0:
                distence = out2-batch
            else:
                distence = np.concatenate((distence, out2-batch),axis=0)
        np.save('mean_distence/epo_%d_mean_distence.npy' % epo,np.array(distence))


    # for epo in range(1, 11):
    #     UNet1 = bfNet().to(DEVICE)
    #     UNet1.load_state_dict(torch.load("two_models_para/u_net1_%d_epo.pth" % epo))
    #     UNet1.eval()

    #     UNet2 = bfNet().to(DEVICE)
    #     UNet2.load_state_dict(torch.load("two_models_para/u_net2_%d_epo.pth" % epo))
    #     UNet2.eval()
        
    #     for i, [batch] in enumerate(train_loader):
    #         batch = batch.to(DEVICE)
    #         out1 = UNet1(batch)
    #         weight = nn.functional.softmax((out1 -batch).abs().detach(), dim=3)
    #         out2 = UNet2(batch+batch*weight)
    #         out2 = out2.detach().cpu().numpy()
    #         batch = batch.detach().cpu().numpy()
    #         if i==0:
    #             distence = np.max(np.abs(out2 -batch), axis=2)
    #         else:
    #             distence = np.concatenate((distence, np.max(np.abs(out2 -batch), axis=2)),axis=0)
    #     distence = np.max(distence, axis=0)
    #     np.save('threshold_sum/epo_%d_threshold.npy' % epo,np.array(distence))

    #     for i, [batch] in enumerate(test_loader):
    #         batch = batch.to(DEVICE)
    #         out1 = UNet1(batch)
    #         weight = nn.functional.softmax((out1 -batch).abs().detach(), dim=3)
    #         out2 = UNet2(batch+batch*weight)
    #         out2 = out2.detach().cpu().numpy()
    #         batch = batch.detach().cpu().numpy()
    #         if i==0:
    #             distence = out2-batch
    #         else:
    #             distence = np.concatenate((distence, out2-batch),axis=0)
    #     np.save('mean_distence/epo_%d_mean_distence.npy' % epo,np.array(distence))

        # for i, [batch] in enumerate(test_loader):
        #     batch = batch.to(DEVICE)
        #        # for index_x in 
        #     out1 = UNet1(batch)
        #     weight = nn.functional.softmax((out1 -batch).abs().detach(), dim=3)
        #     out2 = UNet2(batch+batch*weight)
        #     out2 = out2.detach().cpu().numpy()
        #     batch = batch.detach().cpu().numpy()
        #     batch = batch[0]
        #     batch1 = batch[0]
        #         #print(batch1.shape)
        #     bizhi = np.zeros([47, 48])
        #     for index in range(47):
        #         bizhi = batch1[index]/(batch1[index+1] + 1e-10)
        #             #print(bizhi)
        #     out = out2[0]
        #     out3 = out[0]
        #         #print(out3.shape)
        #     bizhi_hou = np.zeros([47, 48])
        #     for index in range(47):
        #         bizhi_hou = out3[index]/(out3[index+1] + 1e-10)
        #             #print(bizhi_hou)
        #     break
        #     if i==0:
        #         distence = out2-batch
        #     else:
        #         distence = np.concatenate((distence, out2-batch),axis=0)
        # np.save('mean_distence/epo_%d_mean_distence.npy' % epo,np.array(distence))







