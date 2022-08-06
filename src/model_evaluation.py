
from datetime import datetime
import torch
from torch import nn
from torch.utils.data import dataloader
import torch.optim as optim
import pandas as pd
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
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
mini_bs = 128
e_rl = []
e_dl = []
e_gl = []
data = torch.from_numpy(df_trans.values).float()
d_loader = data_loader(data, batch_size=mini_bs, shuffle=True, num_workers=0)

tu = 5
rad = 0.8
sigma = 0.01
dim = 2
xcenter = (rad * np.sin(np.linspace(0, 2 * np.pi, tu, endpoint=False)) + 1) / 2
ycenter = (rad * np.cos(np.linspace(0, 2 * np.pi, tu, endpoint=False)) + 1) / 2
gua = np.vstack([xcenter, ycenter]).T
sgua = 100000
for i, mu in enumerate(gua):
    if i == 0:
        zcsa = np.random.normal(mu, sigma, size=(sgua, dim))
    else:
        zcs = np.random.normal(mu, sigma, size=(sgua, dim))
        zcsa = np.vstack([zcsa, zcs])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
ax.scatter(zcsa[:, 0], zcsa[:, 1], c='C2',marker="x", cmap=None, alpha=0.5, linewidth=2.0, edgecolors=colors) 
ax.set_facecolor('white')
ax.grid(False)
ax.set_title('Prior Distribution of Journal Entries');

en_model = '/content/drive/MyDrive/MRP/Models/en_model.pth'
de_model = '/content/drive/MyDrive/MRP/Models/de_model.pth'
en_eval = Encoder(input_size=df_trans.shape[1], hidden_size=[256, 64, 16, 4, 2])
de_eval = Decoder(output_size=df_trans.shape[1], hidden_size=[2, 4, 16, 64, 256])
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
    en_eval = en_eval.cuda()
    de_eval = de_eval.cuda()
en_eval.load_state_dict(torch.load(en_model))
de_eval.load_state_dict(torch.load(de_model))

td = torch.from_numpy(df_trans.values).float()
d_loader_eval = dataloader(td, batch_size=mini_bs, shuffle=False, num_workers=0)
if (torch.backends.cudnn.version() != None) and (USE_CUDA == True):
    d_loader_eval = dataloader(td.cuda(), batch_size=mini_bs, shuffle=False)

en_eval.eval()
de_eval.eval()
bc = 0
for etb in d_loader_eval:
    z_etb = en_eval(etb) 
    if bc == 0:
      zet = z_etb
    else:
      zet = torch.cat((zet, z_etb), dim=0)
    bc += 1
zet = zet.cpu().detach().numpy()

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
rd = zet[label == 'regular']
go = zet[label == 'global']
lo = zet[label == 'local']
ax.scatter(rd[:, 0], rd[:, 1], c='C2', marker="x", label='regular', edgecolors='w', linewidth=0.5) # plot regular transactions
ax.scatter(go[:, 0], go[:, 1], c='C1', marker="D", label='global', edgecolors='w', s=60) # plot global outliers
ax.scatter(lo[:, 0], lo[:, 1], c='C3', marker="d", label='local', edgecolors='w', s=60) # plot local outliers
ax.set_facecolor('white')
ax.grid(False)
ax.set_title('Semantic Learned Partition Distribution of Journal Entries')
ax.legend(loc='best');