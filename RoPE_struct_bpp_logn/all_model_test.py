from models import RNAConfig,RNAModel
from datas import RibonanzaDatasetTest
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from utils import collate_fn

def plot(df,out):
    font_size=6
    id1=269545321
    id2=269724007
    reshape1=391
    reshape2=457
    #get predictions
    pred_DMS=df[id1:id2+1]['reactivity_DMS_MaP'].to_numpy().reshape(reshape1,reshape2)
    pred_2A3=df[id1:id2+1]['reactivity_2A3_MaP'].to_numpy().reshape(reshape1,reshape2)
    #plot mutate and map
    fig = plt.figure()
    plt.subplot(121)
    plt.title(f'reactivity_DMS_MaP', fontsize=font_size)
    plt.imshow(pred_DMS,vmin=0,vmax=1, cmap='gray_r')
    plt.subplot(122)
    plt.title(f'reactivity_2A3_MaP', fontsize=font_size)
    plt.imshow(pred_2A3,vmin=0,vmax=1, cmap='gray_r')
    plt.tight_layout()
    plt.savefig(out,dpi=500)
    plt.clf()
    plt.close()
    

def main(model_path,dl):
    checkpoint = torch.load(model_path)
    save = '/'.join(model_path.split('/')[:-1]) + '-submission.parquet'
    config = RNAConfig()
    
    device = 'cuda'
    
    model = RNAModel(config)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    ids,preds = [],[]
    with torch.no_grad():
        for batch in tqdm(dl):
            batch = {k:v.to(device) for k,v in batch.items()}
            # for k,v in batch.items():
            #     if k != 'ids':
            #         batch[k] = v.to(device)
            output = model(**batch)
            p = torch.nan_to_num(output.logits).clip(0,1)
            for idx, pi in zip(batch['ids'],p):
                idx = idx[idx != -1] # 去掉pad部分
                ids.append(idx)
                preds.append(pi[:len(idx)])
    ids = torch.concat(ids).cpu()
    preds = torch.concat(preds).cpu()
    df = pd.DataFrame({'id':ids.numpy(), 'reactivity_DMS_MaP':preds[:,1].numpy(), 
                    'reactivity_2A3_MaP':preds[:,0].numpy()})
    # for col in df.columns:
    #     if df[col].dtype == np.float16:
    #         df[col] = df[col].astype(np.float32)
    df.to_parquet(save)
    fig_out = save[:-8]+'.png'
    plot(df,fig_out)
    
if __name__ =='__main__':
    import glob
    config = RNAConfig()
    df_test = pd.read_parquet('./datas/data_struct/test_data_eternafold.parquet')
    ds = RibonanzaDatasetTest(df_test,config)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, 
                shuffle=False, drop_last=False, num_workers=2,collate_fn=collate_fn)
    qbar = tqdm(glob.glob('./RoPE_bpp_struct_distance/outputs/*/*/pytorch_model.bin'))
    for model_path in qbar:
        qbar.set_description(f'Loading and Testing {model_path}')
        main(model_path,dl)
    
    