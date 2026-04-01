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

#Read normal data
normal = pd.read_csv("input/machine-train1-1.csv",sep=",")#, nrows=1000)
normal = normal.drop(["col5","col8","col17","col18","col27","col29"] , axis = 1)

print(normal.shape)
# Transform all columns into float64
# for i in list(normal): 
#     normal[i]=normal[i].apply(lambda x: str(x).replace("," , "."))
normal = normal.astype(float)

# min_max_scaler = preprocessing.MinMaxScaler()
x = normal.values
# n1 = pd.DataFrame(x)
#print("max_x", max(n1), "min_x", min(n1))
# x_scaled = min_max_scaler.fit_transform(x)
normal = pd.DataFrame(x)
#print("max", max(normal), "min", min(normal))

#Read attack data
attack = pd.read_csv("input/machine-testlabel-1-1.csv",sep=",")#, nrows=1000)

labels = [ float(label!= '0' ) for label  in attack["Normal/Attack"].values]
attack = attack.drop(["col5","col8","col17","col18","col27","col29","Normal/Attack"] , axis = 1)
print(attack.shape)
# print('labels',labels)

# Transform all columns into float64
# for i in list(attack):
#     attack[i]=attack[i].apply(lambda x: str(x).replace("," , "."))
attack = attack.astype(float)
x = attack.values 
# x_scaled = min_max_scaler.transform(x)
attack = pd.DataFrame(x)
#print("max", max(attack), "min", min(attack))

window_size=32
windows_normal=normal.values[np.arange(window_size)[None, :] + np.arange(normal.shape[0]-window_size)[:, None]]
windows_attack=attack.values[np.arange(window_size)[None, :] + np.arange(attack.shape[0]-window_size)[:, None]]

windows_labels=[]
for i in range(len(labels)-window_size):
    windows_labels.append(list(np.int_(labels[i:i+window_size])))
labels = [1.0 if (np.sum(window) > 0) else 0 for window in windows_labels]


BATCH_SIZE = 128
N_EPOCHS = 100
DEVICE = 0

windows_normal_train = windows_normal[:int(np.floor(.8 *  windows_normal.shape[0]))]
windows_normal_val = windows_normal[int(np.floor(.8 *  windows_normal.shape[0])):int(np.floor(windows_normal.shape[0]))]


train_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_train).float().view(([windows_normal_train.shape[0],
    1, windows_normal_train.shape[1], windows_normal_train.shape[2]]))), 
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

val_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_normal_val).float().view(([windows_normal_val.shape[0],
    1, windows_normal_val.shape[1], windows_normal_val.shape[2]]))), 
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

test_loader = torch.utils.data.DataLoader(data_utils.TensorDataset(
    torch.from_numpy(windows_attack).float().view(([windows_attack.shape[0],
    1, windows_attack.shape[1], windows_attack.shape[2]]))),
    batch_size=5000, shuffle=False, num_workers=0)

def train():
    UNet1 = Net().to(DEVICE)
    #UNet1.load_state_dict(torch.load("two_models_para/u_net1_78_epo.pth" ))
    UNet1.train()

    UNet2 = Net().to(DEVICE)
    #UNet2.load_state_dict(torch.load("two_models_para/u_net2_78_epo.pth"))
    UNet2.train()

    criterion = torch.nn.MSELoss()
    
    optimizer1 = optim.Adam(UNet1.parameters(),lr=8e-5,betas=(0.9,0.99))
    optimizer2 = optim.Adam(UNet2.parameters(),lr=8e-5,betas=(0.9,0.99))
    # optimizer1 = optim.SGD(UNet1.parameters(), lr = 1e-2, momentum = 0.1)
    # optimizer2 = optim.SGD(UNet2.parameters(), lr = 1e-2, momentum = 0.1)
    for epo in range(0, N_EPOCHS):
        loss_sum1 = 0
        loss_sum2 = 0
        for i, [batch] in enumerate(train_loader):
            batch = batch.to(DEVICE)
            #print("max——b", batch.max().item(), "min", batch.min().item())
            optimizer1.zero_grad()
            out1 = UNet1(batch)
            loss1 = criterion(out1, batch)
            weight = nn.functional.softmax((out1 -batch).abs().detach(), dim=3)
            loss_sum1 += loss1
            loss1.backward()
            optimizer1.step()
            
            optimizer2.zero_grad()
            out2 = UNet2(batch+batch*weight)
            loss2 = criterion(out2, batch)
            loss_sum2 += loss2
            loss2.backward()
            optimizer2.step()

        print('Unet1:This is epoch {epo}, the loss is {loss}'.format(epo=epo+1, loss=loss_sum1))
        print('Unet2:This is epoch {epo}, the loss is {loss}'.format(epo=epo+1, loss=loss_sum2))
        torch.save(UNet1.state_dict(), 'two_models_para/u_net1_{epo}_epo.pth'.format(epo=epo+1))
        torch.save(UNet2.state_dict(), 'two_models_para/u_net2_{epo}_epo.pth'.format(epo=epo+1))

def test():
    pres = []
    UNet = Net().to(DEVICE)
    UNet.load_state_dict(torch.load("u_net.pth"))
    UNet.eval()
    for i, [batch] in enumerate(test_loader):
            batch = batch.to(DEVICE)
            out = UNet(batch)
            pres.append((out-batch).norm(p=2).item())
            # print((out-batch).norm(p=2).item())
    np.save('pres.npy',np.array(pres))
    print(np.array(pres).shape)


if __name__=='__main__':
    train()
