!pip install pysqldf
from pysqldf import SQLDF
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
mini_bs = 128
e_rl = []
e_dl = []
e_gl = []
data = torch.from_numpy(df_trans.values).float()
d_loader = dataloader(data, batch_size=mini_bs, shuffle=True, num_workers=0)

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

def distance(x, y): 
    edist = np.sqrt(np.sum((x - y) ** 2, axis=1))
    return edist

dist_values = np.apply_along_axis(func1d=distance, axis=1, arr=zet, y=mu_gauss)
mdiv = np.min(dist_values, axis=1)
c_ids = np.argmin(dist_values, axis=1)
mdas = np.asarray(mdiv)
for cid in np.unique(c_ids).tolist():
    mask = c_ids == cid
    mdas[mask] = (mdiv[mask] - mdiv[mask].min()) / (mdiv[mask].ptp())

rcce = nn.BCEWithLogitsLoss(reduction='none')
rcne = nn.MSELoss(reduction='none')
if (torch.backends.cudnn.version() != None and USE_CUDA == True):
    rcce = rcce.cuda()
    rcne = rcne.cuda()
en_eval.eval()
de_eval.eval()
bc = 0
for etb in d_loader_eval:
    z_etb = en_eval(etb)
    rb = de_eval(z_etb)
    ica = etb[:, :data_dfc_trans.shape[1]]
    ina = etb[:, data_dfc_trans.shape[1]:]
    rca = rb[:, :data_dfc_trans.shape[1]]
    rna = rb[:, data_dfc_trans.shape[1]:]
    reca = rcce(input=rca, target=ica).mean(dim=1)
    rena = rcne(input=rna, target=ina).mean(dim=1)
    reab = reca + rena
    if bc == 0:
      rea = reab
    else:
      rea = torch.cat((rea, reab), dim=0)
    bc += 1
rea = rea.cpu().detach().numpy()

reas = np.asarray(rea)
for cid in np.unique(c_ids).tolist():
    mask = c_ids == cid
    reas[mask] = (rea[mask] - rea[mask].min()) / (rea[mask].ptp())

alpha = 0.8
anomaly_score = alpha * reas + (1.0 - alpha) * mdas

ano_data = pd.concat([pd.Series(anomaly_score, name='anomaly_score'), 
                       pd.Series(label, name='label')],
                     axis=1)

ano_trans = ano_data.groupby(['label'])['anomaly_score'].mean().reset_index(name='mean')
ano_trans.rename(columns = {'label':'Labels','mean':'MeanScore'}, inplace = True)
ano_trans

plot_data = pd.concat([pd.Series(anomaly_score, name='anomaly_score'), 
                       pd.Series(label, name='label'),                        
                       pd.Series(c_ids, name='cid')],
                     axis=1)
sqldf = SQLDF(globals())
query = 'SELECT * FROM plot_data where cid IN (1,2)'
pdata=sqldf.execute(query)

fig = plt.figure(figsize=(18,10))
ax = fig.add_subplot(111)
rd = pdata[label == 'regular']
go = pdata[label == 'global']
lo = pdata[label == 'local']
ax.scatter(rd.index, rd['anomaly_score'], c='C2', marker='x', alpha=0.4, s=30, linewidth=3, label='regular', edgecolors='w')
ax.scatter(go.index, go['anomaly_score'], c='C1', marker='^', s=120, linewidth=3, label='global', edgecolors='w')
ax.scatter(lo.index, lo['anomaly_score'], c='C3', marker='D', s=120, linewidth=3, label='local', edgecolors='w')
ax.set_facecolor('white')
ax.grid(False)
ax.legend(frameon=True, loc='upper right', ncol=3)
ax.set_title('Anomaly Score with Alpha 0.0');