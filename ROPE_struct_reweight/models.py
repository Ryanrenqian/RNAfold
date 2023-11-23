from transformers.modeling_outputs import TokenClassifierOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from typing import List, Dict, Tuple, Union, Any, Optional
import math
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
    rope_theta = 10000.0
    max_position_embeddings = 512
    num_hidden_layers = 12
    initializer_range = 0.02

class RNARotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int, base: float,device='cuda'):
        super().__init__()
        
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)

        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype) # type: ignore

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len and seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cache[:, :, :seq_len, ...].to(x.dtype), # type: ignore
            self.sin_cache[:, :, :seq_len, ...].to(x.dtype), # type: ignore
        )
    
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
def apply_rotary_pos_emb(q, k, cos, sin, position_idx):
    cos = cos.squeeze(1).squeeze(0) # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0) # [seq_len, dim]
    cos = cos[position_idx].unsqueeze(1) # [bs, 1, seq_len, dim]
    sin = sin[position_idx].unsqueeze(1) # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RNAAttention(nn.Module):
    def __init__(self, config: RNAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        self.rotary_emb = RNARotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, seq_len=q_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states) # [bs, num_heads, q_len, head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)

        return attn_output

class RNAMLP(nn.Module):
    def __init__(self, config: RNAConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.act_fn(self.up_proj(hidden_states))
        down = self.down_proj(gate * up)

        return down

class RNALayerNorm(nn.Module):
    def __init__(self, config: RNAConfig, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(config.hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class RNADecoderLayer(nn.Module):
    def __init__(self, config: RNAConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = RNAAttention(config)
        self.mlp = RNAMLP(config)
        self.input_layernorm = RNALayerNorm(config, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RNALayerNorm(config, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.self_attn(
            hidden_states,
            attention_mask,
            position_ids,
            padding_mask
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        outputs = residual + hidden_states

        return outputs


class RNATransformer(nn.Module):
    def __init__(self, config: RNAConfig):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([RNADecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RNALayerNorm(config, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
    ): 
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device # type: ignore
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device) # type: ignore
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length) # type: ignore
        else:
            position_ids = position_ids.view(-1, seq_length).long() # type: ignore

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).log() # type: ignore

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,) # type: ignore

            hidden_states = layer(
                hidden_states,
                attention_mask,
                position_ids
            )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,) # type: ignore
            return all_hidden_states
        else:
            return hidden_states
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def loss_fn(outputs: torch.Tensor, targets: torch.Tensor,reweight: torch.Tensor) -> torch.Tensor:
    # print(outputs.shape,targets.shape,reweight.shape)
    outputs = outputs[~targets.isnan()]
    reweight = reweight[~targets.isnan()]
    targets = targets[~targets.isnan()].clip(0,1)
    loss = F.l1_loss(outputs, targets, reduction='none') * reweight
    return loss.mean()

class RNAModel(nn.Module):
    def __init__(self, config: RNAConfig):
        super().__init__()
        self._config =  config
        self.backbone = RNATransformer(config)
        self.head = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        reweight: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> TokenClassifierOutput:
        
        outputs = self.backbone(
            input_ids,
            attention_mask,
            position_ids,
            inputs_embeds,
            output_hidden_states
        )
        predictions = self.head(outputs)

        loss = None
        if labels is not None:
            loss = loss_fn(predictions, labels, reweight)

        return TokenClassifierOutput(
            loss=loss, # type: ignore
            logits=predictions,
        )
        
    def post_init(self):
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self._config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()