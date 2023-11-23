import pandas as pd
from tokenizer import RNATokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
import numpy as np

def preprocess(df,tokenizer):
    max_prompt_len = df['sequence'].apply(len).max()
    print('max_prompt_len:',max_prompt_len)
    train_ids = []
    pad_str = 'P'
    for prompt in df['sequence']:
        sequence_ids = tokenizer.encode(prompt)
        train_ids.append(sequence_ids)
    return train_ids
        
class SPL_Dataset(Dataset):
    def __init__(self, train_ids, mode='train', seed=2023, fold=0, nfolds=4, 
                 mask_only=False, **kwargs):
        '''V1'''
        self.Lmax = 405
        self.train_ids = train_ids
        self.tokenizer = RNATokenizer

    def __len__(self):
        return len(self.train_ids)  
    
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.train_ids[idx])}

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
# tqdm.pandas()

class TrainDataset(Dataset):
    def __init__(self, text,label):
        self.encodings = text
        self.labels = label

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        item = {"input_ids": torch.tensor(self.encodings.iloc[index])}
        item["labels"] = torch.tensor([self.labels.iloc[index]])
        return item

class LegalDataset(Dataset):
    def __init__(self, text):
        self.encodings = text

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        item = {"input_ids": torch.tensor(self.encodings.iloc[index])}
        return item


def process_text(filename, name, map_tokenize, encoding):
    print("Opening file...")
    file = open(filename, "r", encoding=encoding)
    text = file.readlines() # list
    file.close()
    text = pd.Series(text)
    tqdm.pandas(desc="Tokenizing")
    text = text.progress_map(map_tokenize)
    dataset = LegalDataset(text)
    text = None
    occ = filename.rfind("/") + 1
    path = filename[:occ]
    torch.save(dataset, path+name+".pt")
    return path+name+".pt"
        
        
        
