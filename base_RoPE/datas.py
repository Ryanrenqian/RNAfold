from sklearn.model_selection import KFold
import torch
import numpy as np

def prepare_targets(df):
    label_cols = [c for c in df.columns if c.startswith('reactivity_0')]
    error_cols = [c for c in df.columns if c.startswith('reactivity_error_0')]
    targets = df[label_cols].values.astype(np.float32)
    mask = df[error_cols].values.astype(np.float32) > 1.0
    targets[mask] = np.nan
    return targets

class RibonanzaDatasetTrain():
    def __init__(self, df, config,mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False, **kwargs):
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']
        split = list(KFold(n_splits=nfolds, random_state=seed, 
                shuffle=True).split(df_2A3))[fold][0 if mode=='train' else 1]
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        self.sequences = df_DMS['sequence'].apply(lambda x: [config.vocab_map[c] for c in x])
        
        self.targets_DMS_MaP = prepare_targets(df_DMS)
        self.targets_2A3_MaP = prepare_targets(df_2A3)
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        
        seq = self.sequences[idx]
        outputs = {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'attention_mask': torch.ones(len(seq), dtype=torch.float),
        }
        outputs['labels'] = torch.from_numpy(np.stack(
                    (self.targets_DMS_MaP[idx][:len(seq)], 
                     self.targets_2A3_MaP[idx][:len(seq)]), axis=1
                ))
        
        return outputs

class RibonanzaDatasetTest():
    def __init__(self, df, config,mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False, **kwargs):
        self.sequences = df['sequence'].apply(lambda x: [config.vocab_map[c] for c in x])
        self.id_min = df['id_min']
        self.id_max = df['id_max']
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        id_min = self.id_min[idx]
        id_max = self.id_max[idx]
        outputs = {
            'input_ids': torch.tensor(seq, dtype=torch.long),
            'attention_mask': torch.ones(len(seq), dtype=torch.float),
            'ids': torch.tensor(np.arange(id_min,id_max+1), dtype=torch.int)
        }
        return outputs