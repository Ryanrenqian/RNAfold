from sklearn.model_selection import KFold
import torch
import numpy as np
import os
from scipy.sparse import save_npz, load_npz
import math
def prepare_targets(df,mode):
    label_cols = [c for c in df.columns if c.startswith('reactivity_0')]
    # error_cols = [c for c in df.columns if c.startswith('reactivity_error_0')]
    targets = df[label_cols].values.astype(np.float32)
    # if mode == 'train':
    #     mask = df[error_cols].values.astype(np.float32) > 1.0
    #     targets[mask] = np.nan
    return targets
def mask_rna_input(sequence,nmute=.15):
    nmute=int(sequence.shape[0]*nmute)
    perm = torch.randperm(sequence.shape[0])
    to_mutate = perm[:nmute]
    masked_seq = sequence.clone()
    masked_seq[to_mutate]=14
    mlm_mask = torch.zeros_like(masked_seq,dtype=torch.bool)
    mlm_mask[to_mutate] = 1
    return masked_seq,mlm_mask

def get_distance(seq,k=4):
    N = len(seq)
    dis = np.zeros((N,N,k))
    for i in range(N):
        for j in range(N):
            for k in range(1,k):
                if i != j:
                    dis[i,j,k] = 1/(i-j)**k
    return dis
    

class RibonanzaDatasetPreTrain():
    def __init__(self, df, config,
                 mask_only=False,data_path='./datas/data_struct',use_sparse=False, **kwargs):
        df = df.reset_index(drop=True)
        self.sequences = df[['sequence','structure']].apply(lambda x: [config.vocab_map[c] for c in zip(x['sequence'], x['structure'])],axis=1)
        self.data_path = data_path
        self.sequences_id = df['sequence_id']
        self.use_sparse = use_sparse
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences.iloc[idx]
        if self.use_sparse:
            bbps = load_npz(os.path.join(self.data_path,self.sequences_id[idx])+'.npz').toarray()
        else:
            bbps = np.load(os.path.join(self.data_path,self.sequences_id[idx])+'.npy.npy')
        dis = get_distance(seq)
        dis[:,:,0] = bbps
        logn = math.log(len(seq),512)
        seq = torch.tensor(seq, dtype=torch.long)
        masked_seq,mlm_mask = mask_rna_input(seq)
        outputs = {
            'input_ids': masked_seq,
            'attention_mask': torch.ones(len(seq), dtype=torch.bool),
            'bbps': torch.tensor(dis,dtype=torch.float).permute(2,0,1),
            'mlm_mask':mlm_mask, 
            'logn': logn,
            'labels': seq,
            
        }
        return outputs
    
class RibonanzaDatasetTrain():
    def __init__(self, df, config,mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False, data_path='./datas/data_struct/bpps/',use_sparse=False,**kwargs):
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        split = list(KFold(n_splits=nfolds, random_state=seed, 
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        # if mode != 'train':
            # m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
            # df_2A3 = df_2A3.loc[m].reset_index(drop=True)
            # df_DMS = df_DMS.loc[m].reset_index(drop=True)
        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        self.sequences = df_2A3[['sequence','structure']].apply(lambda x: [config.vocab_map[c] for c in zip(x['sequence'], x['structure'])],axis=1)
        self.sequences_id = df_2A3['sequence_id']
        self.data_path = data_path
        self.targets_DMS_MaP = prepare_targets(df_DMS,mode)
        self.targets_2A3_MaP = prepare_targets(df_2A3,mode)
        self.use_sparse = use_sparse

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        
        seq = self.sequences[idx]
        logn = math.log(len(seq),512)
        if self.use_sparse:
            bbps = load_npz(os.path.join(self.data_path,self.sequences_id[idx])+'.npz').toarray()
        else:
            bbps = np.load(os.path.join(self.data_path,self.sequences_id[idx])+'.npy.npy')
        dis = get_distance(seq)
        dis[:,:,0] = bbps
        outputs = {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'attention_mask': torch.ones(len(seq), dtype=torch.long),
            'bbps': torch.tensor(dis,dtype=torch.float).permute(2,0,1),
            'logn': logn,
        }
        outputs['labels'] = torch.from_numpy(np.stack(
                    (self.targets_DMS_MaP[idx][:len(seq)], 
                     self.targets_2A3_MaP[idx][:len(seq)]), axis=1
                ))
        
        return outputs
    
    def load_bbps(self,bbp_path):
        bpps = []
        for p in bbp_path:
            bpps.append(np.load(os.path.join(self.data_path,p)+'.npz'))
        return bpps

class RibonanzaDatasetTest():
    def __init__(self, df, config,mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False,data_path='./datas/data_struct', **kwargs):
        self.sequences = df[['sequence','structure']].apply(lambda x: [config.vocab_map[c] for c in zip(x['sequence'], x['structure'])],axis=1).values
        self.sequences_id = df['sequence_id']
        self.id_min = df['id_min']
        self.id_max = df['id_max']
        self.data_path = data_path

        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        logn = math.log(len(seq),512)
        if self.use_sparse:
            bbps = load_npz(os.path.join(self.data_path,self.sequences_id[idx])+'.npz').toarray()
        else:
            bbps = np.load(os.path.join(self.data_path,self.sequences_id[idx])+'.npy.npy')
        dis = get_distance(seq)
        dis[:,:,0] = bbps
        id_min = self.id_min[idx]
        id_max = self.id_max[idx]
        outputs = {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'attention_mask': torch.ones(len(seq), dtype=torch.float),
            'ids': torch.tensor(np.arange(id_min,id_max+1), dtype=torch.int),
            'bbps': torch.tensor(dis,dtype=torch.float).permute(2,0,1),
            'logn': logn,
        }
        return outputs
    