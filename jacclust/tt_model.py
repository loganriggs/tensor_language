import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from dataclasses import dataclass
try:
    from einops import einsum
except Exception: pass
class Rotary(torch.nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4 # multihead attention
    d = x.shape[3]//2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype))

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.squared_attn = config.squared_attn
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977

    def forward(self, x, v1=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        if(self.squared_attn):
            y = self.naive_squared_attention(q, k, v)
        else:
            y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, v1

    def naive_squared_attention(self, q, k, v):
        B, T, H, D = q.shape # B: batch size, T: sequence length, H: number of heads, D: dimension of each head
        # 64, 1024, 6, 64
        scores = einsum(q, k, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")

        pattern = (scores / D).square()
        causal_mask = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
        pattern = pattern.masked_fill(causal_mask.logical_not(), 0.0)
        pattern = pattern / pattern.sum(-1, keepdim=True).clamp_min(1e-9)  # FIX: row-normalize (checkpoint was trained normalized; repo naive_ version omits this -> CE 7.5 vs 3.5)

        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... n_head seq_q d_head")
        return z

class CausalBilinearSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.squared_attn = config.squared_attn
        assert self.n_embd % self.n_head == 0
        self.c_q = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_q2 = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_k2 = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_v = CastedLinear(self.n_embd, self.n_embd, bias=False)
        # output projection
        self.c_proj = CastedLinear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.rotary = Rotary(self.head_dim)
        self.lamb = nn.Parameter(torch.tensor(0.5)) # @Grad62304977
        if(config.bilinear_attn and not config.squared_attn):
            self.bilinear_lamb = nn.Parameter(torch.tensor(0.8))

    def forward(self, x, v1=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        q2 = self.c_q2(x).view(B, T, self.n_head, self.head_dim)
        k2 = self.c_k2(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        if v1 is None:
            v1 = v # This happens if we are in the first block. v needs to be accessed by subsequent blocks
        v = (1 - self.lamb) * v + self.lamb * v1.view_as(v) # @Grad62304977
        cos, sin = self.rotary(q)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),)) # QK norm suggested by @Grad62304977
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q2, k2 = F.rms_norm(q2, (q2.size(-1),)), F.rms_norm(k2, (k2.size(-1),)) # QK norm suggested by @Grad62304977
        q2, k2 = apply_rotary_emb(q2, cos, sin), apply_rotary_emb(k2, cos, sin)
        if(self.squared_attn):
            y = self.squared_attention(q, k, v, q2, k2)
        else:
            y = self.differential_attention(q, k, v, q2, k2)
        y = y.transpose(1, 2).contiguous().view_as(x) # re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y, v1

    def squared_attention(self, q, k, v, q2, k2):
        B, T, H, D = q.shape # B: batch size, T: sequence length, H: number of heads, D: dimension of each head
        # 64, 1024, 6, 64
        scores = einsum(q, k, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        scores2 = einsum(q2, k2, "... seq_q2 n_head d_head2, ... seq_k2 n_head d_head2 -> ... n_head seq_q2 seq_k2")
        pattern = (scores / D) * (scores2 / D)
        causal_mask = torch.tril(torch.ones(T, T, device=pattern.device, dtype=torch.bool))
        pattern.masked_fill_(causal_mask.logical_not(), 0.0)

        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... n_head seq_q d_head")
        return z
    
    def differential_attention(self, q, k, v, q2, k2):
        B, T, H, D = q.shape # B: batch size, T: sequence length, H: number of heads, D: dimension of each head
        # 64, 1024, 6, 64
        scores = einsum(q, k, "... seq_q n_head d_head, ... seq_k n_head d_head -> ... n_head seq_q seq_k")
        scores2 = einsum(q2, k2, "... seq_q2 n_head d_head2, ... seq_k2 n_head d_head2 -> ... n_head seq_q2 seq_k2")
        causal_mask = torch.tril(torch.ones(T, T, device=scores.device, dtype=torch.bool))
        scores.masked_fill_(causal_mask.logical_not(), float('-inf'))
        scores2.masked_fill_(causal_mask.logical_not(), float('-inf'))
        pattern = F.softmax(scores / D, dim=-1) - self.bilinear_lamb*F.softmax(scores2 / D, dim=-1)
        z = einsum(pattern, v, "... n_head seq_q seq_k, ... seq_k n_head d_head -> ... n_head seq_q d_head")
        return z

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = CastedLinear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj  = CastedLinear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_() # zero init suggested by @Grad62304977
        self.bias = nn.Parameter(torch.zeros(config.n_embd))
        self.config = config
    def forward(self, x):
        x = self.c_fc(x)
        if self.config.squared_mlp:
            x = x.square()
        else:
            x = F.relu(x).square() # https://arxiv.org/abs/2109.08668v2; ~1-2% better than GELU; suggested by @SKYLINEZ007 and @Grad62304977
        x = self.c_proj(x)
        return x + self.bias


class Bilinear(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        expansion_factor = config.expansion_factor
        self.Left  = CastedLinear(config.n_embd, expansion_factor* config.n_embd, bias=False)
        self.Right  = CastedLinear(config.n_embd, expansion_factor* config.n_embd, bias=False)
        self.Down  = CastedLinear(expansion_factor* config.n_embd, config.n_embd, bias=False)
        self.Down_bias = nn.Parameter(torch.zeros(config.n_embd))
        self.Down.weight.data.zero_() # zero init suggested by @Grad62304977

    def forward(self, x):
        if(self.config.gated):
            x = F.silu(self.Left(x))*self.Right(x)
        else:
            x = self.Left(x)*self.Right(x)
        x = self.Down(x) + self.Down_bias
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.bilinear_attn:
            self.attn = CausalBilinearSelfAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        if config.bilinear:
            self.mlp = Bilinear(config)
        else:
            self.mlp = MLP(config)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))
        self.config = config
    def forward(self, x, v1, x0):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x1, v1 = self.attn(F.rms_norm(x, (x.size(-1),)), v1)
        x = x + x1
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x, v1

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
class GPTConfig:
    vocab_size : int = 50304
    n_layer : int = 12
    n_head : int = 6  # head dim 128 suggested by @Grad62304977
    n_embd : int = 768
    squared_mlp : bool = False
    bilinear : bool = False
    expansion_factor : int = 4
    gated : bool = False
    squared_attn : bool = False
    bilinear_attn : bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.zero_() # @Grad62304977

    def forward(self, idx, target):

        # forward the GPT model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = F.rms_norm(x, (x.size(-1),)) # @Grad62304977
        x0 = x
        v1 = None
        for block in self.transformer.h:
            x, v1 = block(x, v1, x0)
        x = F.rms_norm(x, (x.size(-1),))

        logits = self.lm_head(x)
        logits = 30 * torch.tanh(logits / 30) # @Grad62304977
        logits = logits.float()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss.float()

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader
