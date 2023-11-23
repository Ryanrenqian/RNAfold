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

def load_config(yaml_file):
    with open(yaml_file,'r') as f:
        return yaml.load(f,yaml.FullLoader)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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

    loss = loss[~torch.isnan(loss)].mean()
    return loss

class MAE(Metric):
    def __init__(self): 
        self.reset()
        
    def reset(self): 
        self.x,self.y = [],[]
        
    def accumulate(self, x,y):
        x = x[y['mask'][:,:x.shape[1]]] # predict
        y = y['react'][y['mask']].clip(0,1) # True label
        self.x.append(x)
        self.y.append(y)

    @property
    def value(self):
        x,y = torch.cat(self.x,0),torch.cat(self.y,0)
        loss = F.l1_loss(x, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    
@torch.no_grad()
def eval(model,dataloader:DataLoader,writer:SummaryWriter,epoch:int):
    model.eval()
    mae = MAE()
    qbar = tqdm(dataloader)
    for i,batch in enumerate(qbar):
        input, target = batch
        pred = model(input)
        mae.accumulate(pred,target)
        qbar.set_description(f'Step {i}-th: avarage val MAE: {mae.value.item():.2f}')
    writer.add_scalar('Val/Loss',mae.value.item(),epoch)
    return mae.value.item()
        
def get_pl_loader(df,length):
    ds_train = PL_Dataset(df,length=length)
    ds_train_len = PL_Dataset(df, mask_only=True,length=length)
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=bs,
                drop_last=True)
    dl_pl = DeviceDataLoader(DataLoader(ds_train, 
                num_workers=num_workers,
                persistent_workers=True,batch_sampler=len_sampler_train), device)
    print('PL size',len(ds_train_len))
    return dl_pl
    

def train(model,dl_train,dl_val,dl_pl,optimizer,scheduler,epochs,output_dir):
    
    log_dir = f"{OUT}/logs"  # TensorBoard 日志目录
    writer = SummaryWriter(log_dir=log_dir)
    saver = TopModelHeap(output_dir=output_dir)
    model.train()
    step = 0
    train_mae = MAE()
    gama2 = 1/epochs
    for epoch in range(epochs):
        qbar = tqdm(zip(dl_train,dl_pl))
        for i,(sl_batch, ss_batch) in enumerate(qbar):
            # batch.to('cuda')
            optimizer.zero_grad()
            # input = batch['input'].to('cuda')
            # target = batch['output'].to('cuda')
            input, target = sl_batch
            pred = model(input)
            loss1 = calloss(pred,target)
            ss_input,ss_target = ss_batch
            ss_pred = model(ss_input)
            loss2 = calloss(ss_pred,ss_target)
            loss = loss1 + loss2
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0)
            optimizer.step()
            scheduler.step()
            step += 1
            train_mae.accumulate(pred,target)
            qbar.set_description(f'Epoch {epoch}-th Train: MAE: {train_mae.value.item():.2f}, train loss: {loss.item():.2f},ss_loss: {loss2.item():.2f}, sl_loss: {loss1.item():.2f}')
        writer.add_scalar('Train/Loss',train_mae.value.item(),epoch)
        writer.add_scalar('Train/LR',scheduler.get_last_lr()[0],epoch)
        train_mae.reset()
        val_mae = eval(model,dl_val,writer,epoch)
        model_name = f'epoch_{epoch}_th.pth'
        saver.push(val_mae,model_name,model.state_dict())
        dl_pl.dataloader.dataset.sample() # 重置数据
        
    
if __name__ == '__main__':
    if len(sys.argv)>1:
        config = load_config(sys.argv[1])
    else:
        config = load_config('./configs/baseline.yaml')
    os.environ['CUDA_VISIBLE_DEVICES']= config['CUDA_VISIBLE_DEVICES']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fname = config['exp_name']
    OUT = f'./outputs/{fname}_PL'
    PATH = config['PATH']
    bs = 256
    bs = bs//2
    num_workers = 8
    SEED = 2023
    nfolds = 4
    fold = 0

    epochs = 300
    seed_everything(SEED)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(PATH,'train_data.parquet'))
    Dataset = RNA_Dataset
    ds_train = Dataset(df, mode='train', fold=fold, nfolds=nfolds)
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
    df = pd.read_parquet('./outputs/baseline_PL.parquet')
    dl_pl = get_pl_loader(df,len(ds_train))
    
    model = load_model(config['model_name']) 
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3,weight_decay=0.05)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100,T_mult= 1000)
    dl_pl = get_pl_loader(df,len(dl_train.dataloader.dataset))
    train(model,dl_train,dl_val,dl_pl,optimizer,scheduler,epochs,OUT)