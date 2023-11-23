from models import RNAConfig,RNAModel
from datas import RibonanzaDatasetTest
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
import pandas as pd
import torch
import numpy as np
def collate_fn(batch):
    new_batch = dict()
    for k in batch[0].keys():
        new_batch[k] = pad_sequence((i[k] for i in batch), batch_first=True, padding_value=0) # type: ignore
    return new_batch
class RNAConfig:
    pad_token_id = 0
    vocab_map = { 'PAD': pad_token_id, 
        ('G', '('): 1,
        ('G', '.'): 2,
        ('G', ')'): 3,
        ('A', '('): 4,
        ('A', '.'): 5,
        ('A', ')'): 6,
        ('C', '('): 7,
        ('C', '.'): 8,
        ('C', ')'): 9,
        ('U', '('): 10,
        ('U', '.'): 11,
        ('U', ')'): 12,
        'SINK_TOKEN': 13 }
    vocab_size = len(vocab_map)
    hidden_size = 128
    intermediate_size = 256
    rms_norm_eps = 1e-6
    num_attention_heads = 8
    # rope_theta = 10000.0
    rope_theta = 10000 * 8 ** (16/14)
    max_position_embeddings = 512
    num_hidden_layers = 12
    initializer_range = 0.02
    
if __name__ =='__main__':
    model_name = 'checkpoint-151704'
    checkpoint = torch.load(f'./outputs/RoPE/{model_name}/pytorch_model.bin')
    save = f'./outputs/RoPE/{model_name}/submission_ntk.parquet'
    df_test = pd.read_parquet('../datas/data_struct/test_data_eternafold.parquet')
    
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
    print(save)