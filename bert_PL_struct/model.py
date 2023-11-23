from torch import nn
import torch
import math
from typing import List, Dict, Tuple, Union, Any, Optional
# from transformers import LlamaModel, LlamaConfig, BertConfig,BertModel,BertForTokenClassification

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim) * (-emb)).to('cuda')
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RNA_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32, **kwargs):
        super().__init__()
        self.seq_emb = nn.Embedding(5,dim//2)
        self.struct_emb = nn.Embedding(4,dim//2)
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.drop_out = nn.Dropout(0.1)
        self.proj_out = nn.Linear(dim,2)
        self.post_init()
    
    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x_seq = x0['seq'][:,:Lmax]
        x_struct =  x0['struct'][:,:Lmax]
        x_seq = self.seq_emb(x_seq)
        x_struct = self.struct_emb(x_struct)
        
        x = torch.concat([x_seq,x_struct],dim=2)
        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        # print(x_seq.shape,x_struct.shape,pos.shape,x.shape)
        x = x + pos
        
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.drop_out(x)
        x = self.proj_out(x)
        
        return x
    
    def post_init(self):
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
            

def load_model(model_name: str) -> nn.Module:
    print(model_name)
    if model_name == 'RNA_Model':
        model = RNA_Model()
        return model
    else: 
        raise ValueError('please check your DL model name')