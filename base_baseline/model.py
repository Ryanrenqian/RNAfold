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

class BaseModel(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32,dropout=0.1, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(5,dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
    
    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]
        
        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos
        x = self.transformer(x, src_key_padding_mask=~mask)        
        return x
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPosEmb):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            
class RNA_Model(nn.Module):
    def __init__(self, dim=192, depth=12, head_size=32,dropout=0.1, **kwargs):
        super().__init__()
        self.base = BaseModel(dim=192, depth=12, head_size=32,dropout=0.1, **kwargs)
        self.proj_out = nn.Linear(dim,2)
    
    def forward(self, x0):
        x = self.base(x0)
        x = self.proj_out(x)
        
        return x
            

def load_model(model_name: str) -> nn.Module:
    print(model_name)
    if model_name == 'RNA_Model':
        model = RNA_Model()
        return model
    else: 
        raise ValueError('please check your DL model name')