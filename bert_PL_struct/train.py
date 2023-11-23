import torch
import torch.nn as nn
import pandas as pd
import os, gc
import sys
import numpy as np
from sklearn.model_selection import KFold
# from torch.optim import AdamW,lr_scheduler
from transformers import  AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from fastai.vision.all import *
from model import *
from data import *
from torch.utils.data import DataLoader
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
def val(model:nn.Module,dataloader:DataLoader,writer:SummaryWriter,epoch:int):
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

def train_epoch(model:nn.Module,optimizer:torch.optim.Optimizer,writer:SummaryWriter,dataloader:DataLoader,scheduler:torch.optim.lr_scheduler.LRScheduler,epoch:float):
    model.train()
    step = 0
    train_mae = MAE()
    iters = len(dataloader)
    qbar = tqdm(dataloader)
    for i, batch in enumerate(qbar):
        # batch.to('cuda')
        optimizer.zero_grad()
        # input = batch['input'].to('cuda')
        # target = batch['output'].to('cuda')
        input, target = batch
        pred = model(input)
        loss = calloss(pred,target)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), clip_value=3.0)
        optimizer.step()
        step += 1
        train_mae.accumulate(pred,target)
        qbar.set_description(f'Epoch - {epoch}-th Train: MAE: {train_mae.value.item():.2f}, loss: {loss.item():.2f}')
        scheduler.step(epoch + i / iters) # type: ignore
        writer.add_scalar('Train/Loss',train_mae.value.item(),epoch) # type: ignore
        writer.add_scalar('Train/LR',scheduler.get_last_lr()[0],epoch) # type: ignore
        
@torch.no_grad()
def inference(model,dataloader,mode='label'):
    rows = []
    model.eval()
    for x,y in tqdm(dataloader):
        p = torch.nan_to_num(model(x),0).clip(0,1).cpu().numpy()
        row1 = {}
        row2 = {}
        for seq, pi, yi in zip(x['origin_seq'],  p, y['react']):
            row1['sequence'] = seq
            row1['experiment_type'] = '2A3_MaP'
            row2['sequence'] = seq
            row2['experiment_type'] = 'DMS_MaP'
            if mode=='unlabel':
                for i in range(457):
                    if i < pi.shape[0]:
                        row1[f'reactivity_{i+1:04}'] = pi[i,0]
                        row2[f'reactivity_{i+1:04}'] = pi[i,1]
                    else:
                        row1[f'reactivity_{i+1:04}'] = float('nan')
                        row2[f'reactivity_{i+1:04}'] = float('nan')
                rows.append(row1)
                rows.append(row2)
            if mode == 'label':
                '''从模型预测结果来看RNA的两端预测效果很差，可能是噪声较大导致的，因此这部分区域采用半监督进行学习'''
                for i in range(25):
                    if i < pi.shape[0]:
                        row1[f'reactivity_{i+1:04}'] = pi[i,0]
                        row2[f'reactivity_{i+1:04}'] = pi[i,1]
                    else:
                        row1[f'reactivity_{i+1:04}'] = float('nan')
                        row2[f'reactivity_{i+1:04}'] = float('nan')
                for i in range(25,126):
                    if i < yi.shape[0]:
                        row1[f'reactivity_{i+1:04}'] = yi[i,0]
                        row2[f'reactivity_{i+1:04}'] = yi[i,1]
                    else:
                        row1[f'reactivity_{i+1:04}'] = float('nan')
                        row2[f'reactivity_{i+1:04}'] = float('nan')
        
                for i in range(126,457):
                    if i < pi.shape[0]:
                        row1[f'reactivity_{i+1:04}'] = pi[i,0]

                        row2[f'reactivity_{i+1:04}'] = pi[i,1]

                    else:
                        row1[f'reactivity_{i+1:04}'] = float('nan')
                        row2[f'reactivity_{i+1:04}'] = float('nan')

    df = pd.DataFrame(rows)
    return df
@torch.no_grad()
def label_epoch(model):
    print('Start Label')
    model.eval()
    label_dl,unlabel_dl = load_data('label')
    label_df = inference(model,label_dl,'label')
    unlabel_df = inference(model,label_dl,'unlabel')
    df = pd.concat([label_df,unlabel_df])
    df['SN_filter']=1
    return load_data('train',df=df)

def sl_circle(model,train_dl,eval_dl,optimizer,scheduler,saver,writer,epoch_id,epochs=5):
    print('Supervised Learning')
    val_mae = 0
    checkpoint = {}
    for epoch in range(epochs):
        epoch_id += 1
        train_epoch(model,optimizer,writer,train_dl,scheduler,epoch_id)
        val_mae = val(model,eval_dl,writer,epoch_id)
        model_name = f'epoch_{epoch_id}_th.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        saver.push(val_mae,model_name,checkpoint)
    return val_mae,checkpoint,epoch_id
    
def pl_circle(model,train_dl,eval_dl,optimizer,scheduler,saver,writer,epoch_id,epochs=2):
    val_mae = np.inf
    checkpoint = {}
    print('Pesudo-Label Learning')
    for epoch in range(epochs):
        epoch_id += 1
        train_epoch(model,optimizer,writer,train_dl,scheduler,epoch_id)
        val_mae = val(model,eval_dl,writer,epoch_id)
        model_name = f'epoch_{epoch_id}_th.pth'
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        saver.push(val_mae,model_name,checkpoint)
    return val_mae,epoch_id

def resume_checkpoint(checkpoint,model,optimizer,scheduler):
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

def main():
    if len(sys.argv)>1:
        config = load_config(sys.argv[1])
    else:
        config = load_config('./configs/baseline.yaml')
    print('Config',str(config))
    os.environ['CUDA_VISIBLE_DEVICES']= config['CUDA_VISIBLE_DEVICES']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fname = config['exp_name']
    PATH = config['PATH']
    
    bs = 256
    num_workers = 8
    SEED = 2023
    nfolds = 4
    fold = 0
    cricles = 100 # 20轮半监督交互
    seed_everything(SEED)
    output_dir = os.path.join('./outputs',config['exp_name']+f'_fold_{fold}')
    saver = TopModelHeap(output_dir=output_dir)
    writer = SummaryWriter(log_dir=os.path.join(output_dir,'logs'))
    train_dl,_ = load_data(stage='train',bs=bs,num_workers=num_workers,device=device)
    eval_dl,_ = load_data(stage='train',bs=bs,num_workers=num_workers,device=device)
    model = load_model(config['model_name']) 
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3,weight_decay=0.05)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=617,num_training_steps=123400)
    
    # 开始训练
    epoch_id = 0
    for cricle in range(cricles):
        sf_loss,checkpoint,epoch_id = sl_circle(model,train_dl,eval_dl,optimizer,scheduler,saver,writer,epoch_id,epochs=5)
        # pl_dataloader = label_epoch(model)
        # pl_loss,epoch_id = pl_circle(model,pl_dataloader,eval_dl,optimizer,scheduler,saver,writer,epoch_id,epochs=2)
        # if pl_loss - sf_loss> 0.04:
        #     print('Load chepoint! Resume!!')
        #     resume_checkpoint(checkpoint,model,optimizer,scheduler)
        
        
if __name__ == '__main__':
    main()
        