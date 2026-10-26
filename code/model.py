"""Llama-style decoder-only transformer.

Pre-norm RMSNorm, rotary position embeddings, SwiGLU MLP (hidden = 8/3 * d_model), no biases,
no dropout, untied input and output embeddings. Attention uses n_heads heads of size head_dim,
so the attention width n_heads * head_dim can differ from d_model (needed for the head-dim sweep).
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    n_layer: int
    d_model: int
    n_heads: int
    head_dim: int
    max_len: int = 4096
    rope_base: float = 10000.0

    @property
    def d_attn(self):
        return self.n_heads * self.head_dim

    @property
    def d_ff(self):
        return 32 * math.ceil(8 * self.d_model / 3 / 32)


# main-sweep sizes; depth and width scale together, head_dim fixed at 64 (32 for xs)
SIZES = {
    "xs": dict(n_layer=2, d_model=128, n_heads=4, head_dim=32),
    "s": dict(n_layer=4, d_model=256, n_heads=4, head_dim=64),
    "m": dict(n_layer=6, d_model=384, n_heads=6, head_dim=64),
    "l": dict(n_layer=8, d_model=512, n_heads=8, head_dim=64),
    "xl": dict(n_layer=12, d_model=640, n_heads=10, head_dim=64),
}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).type_as(x) * self.weight


def rope_tables(head_dim, max_len, base):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    ang = torch.outer(torch.arange(max_len).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd); rotate pairs (x1, x2) = first and second halves
    T = x.shape[2]
    cos, sin = cos[:T].to(x.dtype), sin[:T].to(x.dtype)
    x1, x2 = x.chunk(2, -1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.h, self.hd = cfg.n_heads, cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_attn, bias=False)
        self.out = nn.Linear(cfg.d_attn, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, probe=None):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        mass = None
        if probe is not None:
            # attention from each query position onto its ground-truth source, computed only for
            # the query rows so memory stays O(n * T) instead of O(T^2)
            qpos, src = probe
            qs = q.gather(2, qpos[:, None, :, None].expand(B, self.h, -1, self.hd))
            s = (qs @ k.transpose(-1, -2)).float() / math.sqrt(self.hd)
            future = torch.arange(T, device=x.device)[None, None, None, :] > qpos[:, None, :, None]
            a = s.masked_fill(future, float("-inf")).softmax(-1)
            mass = a.gather(3, src[:, None, :, None].expand(B, self.h, -1, 1)).squeeze(-1)  # (B, H, n)
        return self.out(y.transpose(1, 2).reshape(B, T, -1)), mass


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_up = nn.Linear(cfg.d_model, 2 * cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        g, u = self.gate_up(x).chunk(2, -1)
        return self.down(F.silu(g) * u)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x, cos, sin, probe=None):
        h, a = self.attn(self.n1(x), cos, sin, probe)
        x = x + h
        return x + self.mlp(self.n2(x)), a


class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg.head_dim, cfg.max_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for b in self.blocks:  # scale residual-branch outputs by depth (GPT-2 style)
            for w in (b.attn.out.weight, b.mlp.down.weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x, qpos=None, src=None):
        """Returns logits at qpos (B, n, V), or at every position if qpos is None. If src is given,
        also returns per-layer attention mass (B, H, n) from each query onto its source position."""
        h = self.emb(x)
        probe = (qpos, src) if src is not None else None
        attns = []
        for b in self.blocks:
            h, a = b(h, self.cos, self.sin, probe)
            attns.append(a)
        if qpos is not None:  # only project the positions we score
            h = h.gather(1, qpos.unsqueeze(-1).expand(-1, -1, h.shape[-1]))
        return self.head(self.norm(h)), attns


def n_params(model):
    """Non-embedding parameters (excludes the input embedding and the output head)."""
    return sum(p.numel() for n, p in model.named_parameters() if not n.startswith(("emb.", "head.")))


def train_flops_per_seq(cfg, T, n_scored, n_nonemb):
    """Training FLOPs for one sequence (forward + backward = 3x forward), following Kaplan et al.
    (2020): forward cost per token is 2N + 2 * n_layer * T * d_attn. The output head is only
    applied at the n_scored query positions."""
    per_token = 6 * n_nonemb + 6 * cfg.n_layer * T * cfg.d_attn
    return T * per_token + 6 * cfg.d_model * cfg.vocab_size * n_scored
