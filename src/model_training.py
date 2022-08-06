import random
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import dataloader
import torch.optim as optim
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

current = datetime.utcnow().strftime("%Y%m%d-%H:%M:%S")
USE_CUDA = True

from google.colab import drive
drive.mount('/content/drive')

driveurl = '/content/drive/MyDrive/MRP/Data/Synthetic_Financial_Accounting_Dataset.csv'
data_df = pd.read_csv(driveurl)

data_df.label.value_counts()
label = data_df.pop('label')

can = ['KTOSL', 'PRCTR', 'BSCHL', 'HKONT', 'BUKRS', 'WAERS']
data_dfc_trans = pd.get_dummies(data_df[can])
num_an = ['DMBTR', 'WRBTR']
num_a = data_df[num_an] + 1e-4
num_a = num_a.apply(np.log)
data_df_num_a = (num_a - num_a.min()) / (num_a.max() - num_a.min())
df_trans = pd.concat([data_dfc_trans, data_df_num_a], axis = 1)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.L1 = nn.Linear(input_size, hidden_size[0], bias=True) 
        nn.init.xavier_uniform_(self.L1.weight) 
        nn.init.constant_(self.L1.bias, 0.0) 
        self.R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True) 
        self.L2 = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.constant_(self.L2.bias, 0.0)
        self.R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L3 = nn.Linear(hidden_size[1], hidden_size[2], bias=True)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.constant_(self.L3.bias, 0.0)
        self.R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L4 = nn.Linear(hidden_size[2], hidden_size[3], bias=True)
        nn.init.xavier_uniform_(self.L4.weight)
        nn.init.constant_(self.L4.bias, 0.0)
        self.R4 = torch.nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L5 = nn.Linear(hidden_size[3], hidden_size[4], bias=True)
        nn.init.xavier_uniform_(self.L5.weight)
        nn.init.constant_(self.L5.bias, 0.0)
        self.R5 = torch.nn.LeakyReLU(negative_slope=0.4, inplace=True)
        
    def forward(self, layer):
        layer = self.R1(self.L1(layer))
        layer = self.R2(self.L2(layer))
        layer = self.R3(self.L3(layer))
        layer = self.R4(self.L4(layer))
        layer = self.R5(self.L5(layer))
        return layer

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size):
        super(Decoder, self).__init__()
        self.L1 = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        nn.init.xavier_uniform_(self.L1.weight)
        nn.init.constant_(self.L1.bias, 0.0)
        self.R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L2 = nn.Linear(hidden_size[1], hidden_size[2], bias=True)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.constant_(self.L2.bias, 0.0)
        self.R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L3 = nn.Linear(hidden_size[2], hidden_size[3], bias=True)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.constant_(self.L3.bias, 0.0)
        self.R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L4 = nn.Linear(hidden_size[3], hidden_size[4], bias=True)
        nn.init.xavier_uniform_(self.L4.weight)
        nn.init.constant_(self.L4.bias, 0.0)
        self.R4 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L5 = nn.Linear(hidden_size[4], output_size, bias=True)
        nn.init.xavier_uniform_(self.L5.weight)
        nn.init.constant_(self.L5.bias, 0.0)
        self.sig = torch.nn.Sigmoid()
    def forward(self, layer):
        layer = self.R1(self.L1(layer))
        layer = self.R2(self.L2(layer))
        layer = self.R3(self.L3(layer))
        layer = self.R4(self.L4(layer))
        layer = self.sig(self.L5(layer))
        return layer

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()
        self.L1 = nn.Linear(input_size, hidden_size[0], bias=True) 
        nn.init.xavier_uniform_(self.L1.weight) 
        nn.init.constant_(self.L1.bias, 0.0)
        self.R1 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L2 = nn.Linear(hidden_size[0], hidden_size[1], bias=True)
        nn.init.xavier_uniform_(self.L2.weight)
        nn.init.constant_(self.L2.bias, 0.0)
        self.R2 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L3 = nn.Linear(hidden_size[1], hidden_size[2], bias=True)
        nn.init.xavier_uniform_(self.L3.weight)
        nn.init.constant_(self.L3.bias, 0.0)
        self.R3 = nn.LeakyReLU(negative_slope=0.4, inplace=True)
        self.L4 = nn.Linear(hidden_size[2], output_size, bias=True)
        nn.init.xavier_uniform_(self.L4.weight)
        nn.init.constant_(self.L4.bias, 0.0)
        self.sig2 = torch.nn.Sigmoid()

    def forward(self, layer):
        layer = self.R1(self.L1(layer))
        layer = self.R2(self.L2(layer))
        layer = self.R3(self.L3(layer))
        layer = self.sig2(self.L4(layer))
        return layer

lr_e = 1e-3
lr_d = 1e-3
lr_z = 1e-5
en_train = Encoder(input_size=df_trans.shape[1], hidden_size=[256, 64, 16, 4, 2])
de_train = Decoder(output_size=df_trans.shape[1], hidden_size=[2, 4, 16, 64, 256])
dis_train = Discriminator(input_size=2, hidden_size=[256, 16, 4, 2], output_size=1)
en_opt = optim.Adam(en_train.parameters(), lr=lr_e)
de_opt = optim.Adam(de_train.parameters(), lr=lr_d)
dis_cr = nn.BCELoss()
dis_opt = optim.Adam(dis_train.parameters(), lr=lr_z)
rcc = nn.BCELoss(reduction='mean')
rcn = nn.MSELoss(reduction='mean')
rcc = rcc.cuda()
rcn = rcn.cuda()

numberofepochs = 100
mini_bs = 128
e_rl = []
e_dl = []
e_gl = []
mini_bvs = 1000
data = torch.from_numpy(df_trans.values).float()
d_loader = dataloader(data, batch_size=mini_bs, shuffle=True, num_workers=0)

for epoch in range(numberofepochs):
    mini_bc = 0
    b_rl = 0.0
    b_dl = 0.0
    b_gl = 0.0
    if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
        en_train.cuda()
        de_train.cuda()
        dis_train.cuda()
    en_train.train()
    de_train.train()
    dis_train.train()
    start_time = datetime.current()
    for mini_bd in d_loader:
        mini_bc += 1
        if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
            mini_bt = torch.cuda.FloatTensor(mini_bd)
        else:
             mini_bt = torch.FloatTensor(mini_bd)
        en_train.zero_grad()
        de_train.zero_grad()
        dis_train.zero_grad()
        out_sam = en_train(mini_bt)
        mini_br = de_train(out_sam)
        bca = mini_bt[:, :data_dfc_trans.shape[1]]
        bnum = mini_bt[:, data_dfc_trans.shape[1]:]
        r_bca = mini_br[:, :data_dfc_trans.shape[1]]
        r_bnum = mini_br[:, data_dfc_trans.shape[1]:]
        r_ec = rcc(input=r_bca, target=bca)
        r_en = rcn(input=r_bnum, target=bnum) 
        rl = r_ec + r_en
        rl.backward()
        b_rl += rl.item()
        de_opt.step()
        en_opt.step()
        dis_train.eval()
        ztb = zcsam[random.sample(range(0, zcsam.shape[0]), mini_bs),:]
        ztb = torch.FloatTensor(ztb)

        if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
            ztb = ztb.cuda()
        zfg = en_train(mini_bt)
        drg = dis_train(ztb)
        dfg = dis_train(zfg)
        dt = torch.FloatTensor(torch.ones(drg.shape))
        dft = torch.FloatTensor(torch.zeros(dfg.shape))
        if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
            dt = dt.cuda()
            dft = dft.cuda()
        dlr = dis_cr(target=dt, input=drg)
        dlf = dis_cr(target=dft, input=dfg)
        dl = dlf + dlr
        dl.backward()
        b_dl += dl.item()
        dis_opt.step()
        en_train.zero_grad()
        de_train.zero_grad()
        dis_train.zero_grad()
        en_train.train()
        en_train.zero_grad()
        zfg = en_train(mini_bt)
        dfg = dis_train(zfg)
        dft = torch.FloatTensor(torch.ones(dfg.shape))
        if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
            dft = dft.cuda()
        gl = dis_cr(target=dft, input=dfg)
        b_gl += gl.item()
        gl.backward()
        en_opt.step()
        en_train.zero_grad()
        de_train.zero_grad()
        dis_train.zero_grad()
    epoch_rl = b_rl / mini_bc
    e_rl.extend([epoch_rl])
    epoch_dl = b_dl / mini_bc
    e_dl.extend([epoch_dl])
    epoch_gl = b_gl / mini_bc
    e_gl.extend([epoch_gl])
    current = datetime.utccurrent().strftime("%Y%m%d-%H:%M:%S")
    print('[LOG TRAIN {}] epoch: {:04}/{:04}, reconstruction loss: {:.4f}'.format(current, epoch + 1, numberofepochs, epoch_rl))
    print('[LOG TRAIN {}] epoch: {:04}/{:04}, discriminator loss: {:.4f}'.format(current, epoch + 1, numberofepochs, epoch_dl))
    print('[LOG TRAIN {}] epoch: {:04}/{:04}, generator loss: {:.4f}'.format(current, epoch + 1, numberofepochs, epoch_gl))

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(range(1, len(e_rl)+1), e_rl,color='blue',linestyle='dashed',linewidth=2)
ax.set_facecolor('white')
ax.grid(False)
ax.set_xlabel('training epochs values')
ax.set_ylabel('reconstruction loss values')
ax.set_title('Performance of reconstruction network over 100 epoch');

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(range(0, len(e_dl)), e_dl,color='blue',linestyle='dashed',linewidth=2)
ax.set_facecolor('white')
ax.grid(False)
ax.set_xlabel('training epochs values')
ax.set_ylabel('discrimination loss values')
ax.set_title('Performance of discrimination network over 100 epoch');

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111)
ax.plot(range(0, len(e_gl)), e_gl,color='blue',linestyle='dashed',linewidth=2)
ax.set_facecolor('white')
ax.grid(False)
ax.set_xlabel('training epochs values')
ax.set_ylabel('generation loss values')
ax.set_title('Performance of generation network over 100 epoch');