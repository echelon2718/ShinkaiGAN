#### This module was adopted from Zheng, et. al. Hybrid Perception Block in ITTR (https://github.com/lucidrains/ITTR-pytorch?tab=readme-ov-file)
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

class ChanLayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b

class DPSA(nn.Module):
    """ Dual-pruned Self-attention Block """
    def __init__(self, dim, height_top_k=16, width_top_k=16, dim_head=32, heads=8, dropout=0.2):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.norm = ChanLayerNorm(dim)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias=False)

        self.height_top_k = height_top_k
        self.width_top_k = width_top_k

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Conv2d(inner_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Fold out heads
        q = q.view(b, self.heads, self.dim_head, h, w)
        k = k.view(b, self.heads, self.dim_head, h, w)
        v = v.view(b, self.heads, self.dim_head, h, w)

        # L2 normalization on queries and keys
        q = l2norm(q)
        k = l2norm(k)

        # Determine need for height and width selection
        need_height_select_and_rank = self.height_top_k < h
        need_width_select_and_rank = self.width_top_k < w

        if need_width_select_and_rank or need_height_select_and_rank:
            q_probe = q.sum(dim=[3, 4])

        # Gather along height
        if need_height_select_and_rank:
            k_height = k.sum(dim=3)
            top_h_indices = torch.topk(torch.einsum('bnd,bnhd->bnh', q_probe, k_height), k=self.height_top_k, dim=-1).indices
            k = torch.gather(k, 3, top_h_indices.expand(-1, -1, -1, -1, w))
            v = torch.gather(v, 3, top_h_indices.expand(-1, -1, -1, -1, w))

        # Gather along width
        if need_width_select_and_rank:
            k_width = k.sum(dim=4)
            top_w_indices = torch.topk(torch.einsum('bnd,bnwd->bnw', q_probe, k_width), k=self.width_top_k, dim=-1).indices
            k = torch.gather(k, 4, top_w_indices.expand(-1, -1, -1, h, -1))
            v = torch.gather(v, 4, top_w_indices.expand(-1, -1, -1, h, -1))

        # Reshape for matrix multiplication
        q = q.reshape(b * self.heads, h * w, self.dim_head)
        k = k.reshape(b * self.heads, h * w, self.dim_head)
        v = v.reshape(b * self.heads, h * w, self.dim_head)
        
        # Cosine similarity and attention
        sim = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(sim, dim=-1)
        attn = self.dropout(attn)

        # Aggregate output
        out_bmm = torch.bmm(attn, v)
        out = out_bmm.view(b, self.heads, self.dim_head, h, w)
        out = out.permute(0, 1, 3, 4, 2).reshape(b, self.heads * self.dim_head, h, w)
        return self.to_out(out)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class HPB(nn.Module):
    """ Hybrid Perception Block """
    def __init__(
        self,
        in_channels,
        dim_head = 32,
        heads = 8,
        ff_mult = 4,
        attn_height_top_k = 16,
        attn_width_top_k = 16,
        ff_dropout = 0.
    ):
        super().__init__()

        self.attn = DPSA(
            dim = in_channels,         # dimension
            dim_head = dim_head,     # dimension per attention head
            heads = heads,         # number of attention heads
            height_top_k = attn_height_top_k, # number of top indices to select along height, for the attention pruning
            width_top_k = attn_width_top_k,  # number of top indices to select along width, for the attention pruning
            dropout = 0.       # attn dropout
        )

        self.dwconv = nn.Conv2d(in_channels, in_channels, 3, padding = 1, groups = in_channels)
        self.attn_parallel_combine_out = nn.Conv2d(in_channels * 2, in_channels, 1)

        ff_inner_dim = in_channels * ff_mult

        self.ff = nn.Sequential(
            nn.Conv2d(in_channels, ff_inner_dim, 1),
            nn.InstanceNorm2d(ff_inner_dim),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            Residual(nn.Sequential(
                nn.Conv2d(ff_inner_dim, ff_inner_dim, 3, padding = 1, groups = ff_inner_dim),
                nn.InstanceNorm2d(ff_inner_dim),
                nn.GELU(),
                nn.Dropout(ff_dropout)
            )),
            nn.Conv2d(ff_inner_dim, in_channels, 1),
            nn.InstanceNorm2d(ff_inner_dim)
        )

        self.attn_map_extractor = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        attn_branch_out = self.attn(x)
        conv_branch_out = self.dwconv(x)

        concatted_branches = torch.cat((attn_branch_out, conv_branch_out), dim = 1)
        attn_out = self.attn_parallel_combine_out(concatted_branches) + x

        return self.ff(attn_out), self.attn_map_extractor(attn_out)
