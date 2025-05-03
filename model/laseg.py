# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from torch_radon import RadonFanbeam
import numpy as np
import torch
import torch.nn.functional as F
from .vit_seg_modeling import VisionTransformer as ViT_seg
from .vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torch import nn
from einops import rearrange
from time import time
import numbers

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
        )
        self.wo = nn.Linear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
        )


    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        
        xk = xk.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_heads, self.head_dim) 
        
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        keys = xk
        values = xv
        xq = xq.transpose(1, 2) 
        keys = keys.transpose(1, 2) 
        values = values.transpose(
            1, 2
        )
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values) 
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class Positional_Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))
        self.wq = nn.Conv2d(
            in_channels=256, 
            out_channels=256,
            kernel_size=8,   
            stride=8,        
            padding=0,
            groups=256
        )
        self.pos_wk = nn.Conv2d(
            in_channels=1,   
            out_channels=256,
            kernel_size=8,   
            stride=8,        
            padding=0,       
        )
        
        self.pos_wv = nn.Conv2d(
            in_channels=1,   
            out_channels=256,
            kernel_size=8,   
            stride=8,        
            padding=0,       
        )
        
        self.upsample_layer = nn.Upsample(size=(720, 768), mode='bilinear', align_corners=False)
        self.project_out = nn.Conv2d(
            in_channels=256,  
            out_channels=1,   
            kernel_size=(3, 3), 
            stride=(1, 1),      
            padding=(1, 1)      
        )

    def forward(
        self,
        x: torch.Tensor,
        pos_kv: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape
        
        x = x.unsqueeze(0)
        
        b, c, h, w = pos_kv.shape
        h = h // 8
        w = w // 8
        
        xq = self.wq(pos_kv)
        
        xk = self.pos_wk(x)
        xv = self.pos_wv(x)
        
        xq = rearrange(xq, 'b c h w -> b c (h w)')
        xk = rearrange(xk, 'b c h w -> b c (h w)')
        xv = rearrange(xv, 'b c h w -> b c (h w)')


        xq = torch.nn.functional.normalize(xq, dim=-1)
        xk = torch.nn.functional.normalize(xk, dim=-1)

        attn = (xq @ xk.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ xv)
        
        out = rearrange(out, 'b c (h w) -> b c h w', h=h, w=w)
        
        out = self.project_out(self.upsample_layer(out)).squeeze(0)

        return out


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)

        self.w1 = nn.Linear(
            dim, hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            hidden_dim, dim, bias=False
        )
        self.w3 = nn.Linear(
            dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.pos_attn = Positional_Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
    ):
        # print(torch.isnan(x).any())
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class posTransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.pos_attn = Positional_Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_kv: torch.Tensor,
    ):
        
        h = x + self.pos_attn(self.attention_norm(x), pos_kv)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs, configs=None):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        self.tok_embeddings1 = nn.Linear(in_features=768, out_features=768, bias=True)
        self.pos_embeddings = nn.Linear(in_features=768, out_features=768, bias=True)
        self.configs = configs

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
            self.layers.append(posTransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )
        self.output1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=3,
            padding=1,
        )  
        self.output2 = nn.Conv2d(
            in_channels=64,
            out_channels=1,
            kernel_size=3,
            padding=1,
        )  
        
        self.seg_net = ViT_seg(self.configs, img_size=self.configs.img_size, num_classes=self.configs.n_classes).cuda()
        
    def radon_fanbeam(self, sinogram, num_view=720):
        image_size = 512
        detector_count = 768
        source_distance = 600
        det_distance = 290
        
        angles = np.linspace(0, 2*np.pi, num_view, endpoint=False)
        radon = RadonFanbeam(
            image_size,
            det_count=detector_count,
            angles=angles,
            source_distance=source_distance,
            det_distance=det_distance,
        )
        filtered_sinogram = radon.filter_sinogram(sinogram, "ram-lak")
        reconstructed_image = radon.backward(filtered_sinogram)
        
        return reconstructed_image 
    
    def forward(self, tokens: torch.Tensor, start_pos: int, pos_kv: torch.Tensor, min, max):

        _bsz, seqlen, _ = tokens.shape
        
        h = self.tok_embeddings1(tokens)
        pos_kv = self.pos_embeddings(pos_kv)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        for layer in self.layers:
            h = layer(h, pos_kv, start_pos, freqs_cis)
            
        h_r = self.norm(h)
        output = self.output2(self.output1(h_r.unsqueeze(0))).squeeze(0)
        output1 = self.radon_fanbeam((output*(max-min)+min))

        seg_output = self.seg_net(h.unsqueeze(0))  
        return output1, output, seg_output, h
