import torch
import torch.nn as nn
import pandas as pd
import os, gc
import sys
import numpy as np
from sklearn.model_selection import KFold
from torch.optim import AdamW,lr_scheduler
import torch.nn.functional as F
from fastai.vision.all import *
from model import *
from data import *
from torch.utils.data import DataLoader

from torch.cuda.amp import GradScaler, autocast
import pandas as pd
import pyarrow as pa
import random
from tqdm.auto import tqdm
import pyarrow.parquet as pq
from torch.nn.parallel import DataParallel
from tensorboardX import SummaryWriter
from utils import TopModelHeap
import yaml
from transformers import RoFormerModel,PreTrainedTokenizerFast
from torch import nn
import pandas as pd
def calloss(pred,target,use_weight=False):
    p = pred[target['mask'][:,:pred.shape[1]]]
    y = target['react'][target['mask']].clip(0,1)
    if use_weight:
        weight = target['sn'].unsqueeze(1).expand(*pred.shape)
        weight = weight[target['mask'][:,:pred.shape[1]]]
        loss = F.l1_loss(p, y, reduction='none') * weight
    else:
        loss = F.l1_loss(p, y, reduction='none')
    loss = loss[~torch.isnan(loss)].mean()
    return loss

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.roformer = roformer
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.roformer.config.hidden_size, 2)

    def forward(self, input_tensor):
        x = self.roformer(input_tensor)[0]
        x = self.dropout(x)
        return self.linear(x)
    
class RNA_Dataset(Dataset):
    def __init__(self, df,tokenizer, mode='train', seed=2023, fold=0, nfolds=4, mask_only:False, **kwargs):
        '''V1'''
        self.tokenizer = tokenizer
        self.Lmax = 206
        df['L'] = df.sequence.apply(len)
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        split = list(KFold(n_splits=nfolds, random_state=seed, 
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(self.Lmax, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        seq = self.tokenizer.encode(x, truncation=True, padding=True, max_length=self.Lmax, pad_to_max_length=True)
        seq = np.array(seq)
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,self.Lmax-len(seq)))
        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        # 额外添加质控过滤，参考 OpenVaccine 比赛，过滤掉误差较大的结果，与LB的高质量数据对齐
        react[react_err>10] = float('nan')
        react[react/react_err<1.5] = float('nan')
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])
        return {'seq':torch.from_numpy(seq), 'mask':mask,'origin_seq':self.seq[idx]}, \
               {'react':react, 
                'sn':sn, 'mask':mask}
def dict_to(x, device='cuda'):
    data = {}
    for k in x:
        if k != 'origin_seq':
            data[k] = x[k].to(device)
        else:
            data[k] = x[k]
    return data

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)

def train(model,dataloader,optimizer,scheduler,epochs,output_dir,loss_func):
    log_dir = f"{OUT}/logs"  # TensorBoard 日志目录
    writer = SummaryWriter(log_dir=log_dir)
    saver = TopModelHeap(output_dir=output_dir)
    model.train()
    step = 0
    train_mae = MAE()
    iters = len(dataloader)
    for epoch in range(epochs):
        qbar = tqdm(dataloader)
        for i, batch in enumerate(qbar):
            # batch.to('cuda')
            optimizer.zero_grad()
            # input = batch['input'].to('cuda')
            # target = batch['output'].to('cuda')
            input, target = batch
            pred = model(input)
            loss = loss_func(pred,target)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0)
            optimizer.step()
            step += 1
            train_mae.accumulate(pred,target)
            qbar.set_description(f'Epoch - {epoch}-th Train: MAE: {train_mae.value.item():.2f}, loss: {loss.item():.2f}')
            scheduler.step(epoch + i / iters)
        writer.add_scalar('Train/Loss',train_mae.value.item(),epoch)
        writer.add_scalar('Train/LR',scheduler.get_last_lr()[0],epoch)
        train_mae.reset()
        val_mae = val(model,dl_val,writer,epoch)
        model_name = f'epoch_{epoch}_th.pth'
        saver.push(val_mae,model_name,model.state_dict())
        

if __name__ =='__main__':
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./my_tokens")
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = RoFormerModel.from_pretrained('./outputs/pretrain_model/checkpoint-720000')
    print('Loading Pretrained Model and Token')
    df = pd.read_parquet('../datas/train_data.parquet')
    fold = 0
    nfold = 4
    OUT = f'./outputs/pretrain_720000_fold_{fold}'
    epochs = 200
    os.makedirs(OUT, exist_ok=True)
    ds_train = RNA_Dataset(df,tokenizer)
    ds_train_len = Dataset(df, mode='train', fold=fold, 
                    nfolds=nfolds, mask_only=True)
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
                    drop_last=True)
    dl_train = DeviceDataLoader(DataLoader(ds_train, 
                    num_workers=num_workers,
                    persistent_workers=True,batch_sampler=len_sampler_train), device)
    ds_val = Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
    ds_val_len = Dataset(df, mode='eval', fold=fold, nfolds=nfolds, 
            mask_only=True)
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=bs, 
            drop_last=False)
    dl_val= DeviceDataLoader(DataLoader(ds_val, 
            batch_sampler=len_sampler_val, num_workers=num_workers), device)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3,weight_decay=0.05)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs)
    train(model,dl_train,optimizer,scheduler,epochs,OUT,calloss)
    
    