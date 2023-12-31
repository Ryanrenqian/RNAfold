from models import RNAConfig,RNAModel
from datas import RibonanzaDatasetTest
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import pandas as pd
import torch
import numpy as np
from utils import collate_fn
if __name__ =='__main__':
    pretrain = 'checkpoint-355320'
    model_name = 'checkpoint-154112'
    checkpoint = torch.load(f'./outputs/RoPE-{pretrain}/{model_name}/pytorch_model.bin')
    save = f'./outputs/RoPE-{pretrain}/{model_name}/submission.parquet'
    df_test = pd.read_parquet('../datas/data_struct/test_sequences.parquet')
    config = RNAConfig()
    ds = RibonanzaDatasetTest(df_test,config)
    device = 'cuda'
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, 
                shuffle=False, drop_last=False, num_workers=2,collate_fn=collate_fn)
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
                idx = idx[idx != 0] # 去掉pad部分
                ids.append(idx)
                preds.append(pi[:len(idx)])
    ids = torch.concat(ids).cpu()
    preds = torch.concat(preds).cpu()
    df = pd.DataFrame({'id':ids.numpy(), 'reactivity_DMS_MaP':preds[:,1].numpy(), 
                    'reactivity_2A3_MaP':preds[:,0].numpy()})
    for col in df.columns:
        if df[col].dtype == np.float16:
            df[col] = df[col].astype(np.float32)
    df.to_parquet(save)