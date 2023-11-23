from model import RNA_Model
from data import RNA_Dataset_Test,DeviceDataLoader
import sys
import pandas as pd
import os
import numpy as np
from tqdm.auto import tqdm
import math
import torch

if __name__ =='__main__':
    checkpoint = './outputs/baseline/epoch_55_th.pth'
    save = checkpoint[:-4]+'_submission.parquet'
    bs = 256
    num_workers = 16
    device = 'cuda'
    df_test = pd.read_parquet(os.path.join('../datas/','test_sequences.parquet'))    
    ds = RNA_Dataset_Test(df_test)
    dl = DeviceDataLoader(torch.utils.data.DataLoader(ds, batch_size=bs, 
                shuffle=False, drop_last=False, num_workers=num_workers), device)
    model = RNA_Model() 
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint,map_location=torch.device('cuda')))
    model.eval()
    ids,preds = [],[]
    for x,y in tqdm(dl):
        with torch.no_grad():
            p = torch.nan_to_num(model(x)).clip(0,1)
            
        for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
            ids.append(idx[mask])
            preds.append(pi[mask[:pi.shape[0]]])
    ids = torch.concat(ids)
    preds = torch.concat(preds)

    df = pd.DataFrame({'id':ids.numpy(), 'reactivity_DMS_MaP':preds[:,1].numpy(), 
                    'reactivity_2A3_MaP':preds[:,0].numpy()})
    for col in df.columns:
        if df[col].dtype == np.float16:
            df[col] = df[col].astype(np.float32)
    df.to_parquet(save)
        