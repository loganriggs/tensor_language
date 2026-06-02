#!/usr/bin/env python3
"""Bilinear architecture sweep — verify that adding components lowers loss.

Every architecture here is a *tensor network* once trained:
  - Embed / Unembed         : linear
  - RoPE                    : fixed per-position rotation (orthogonal linear)
  - Bilinear attention      : (Q1x·K1x)(Q2x·K2x)/d_h^2 — degree-4 polynomial in x
  - BatchNorm on Q1/K1/Q2/K2: per-channel affine at inference -> folds into Q/K weights
  - Bilinear MLP            : D(Lx ⊙ Rx) — degree-2 polynomial in x
  - final norm              : SWEPT — see `final_norm` below

The only non-foldable choice is `layernorm` (its 1/sqrt(var) is per-sample, data
dependent). `static-rms` divides by a *running* scalar RMS (like BatchNorm running
stats), so it folds into the unembed and stays a tensor network. `none` is the
purest network. We sweep all three to *measure* what the non-TN LayerNorm buys.

Variants (components added left to right -> loss should fall monotonically):
    embed_unembed  : 0 layers (unigram floor)        Embed -> norm -> Unembed
    attn1 / attn2  : 1 / 2 bilinear-attention layers
    xf1   / xf2    : 1 / 2 transformer layers (bilinear attn + bilinear MLP)

Usage:
    python train_sweep.py --smoke              # tiny wiring check on cached data, ~seconds
    python train_sweep.py --steps 2000         # short Pile run, all 5 variants @ d=128
    python train_sweep.py --steps 50000 --widths 128,256,512 --norms layernorm,static-rms,none
"""
import math
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, einsum


# =============================================================================
# Config
# =============================================================================
VOCAB_SIZE = 5000
D_HEAD = 32                       # fixed; n_head = d_model // D_HEAD
CACHE_DIR = Path(__file__).parent / "cached_tokens"
PILE_VAL = CACHE_DIR / "dsir_pile_val_ctx512.pt"

# component recipe per variant — adding components should only lower loss.
# attn4/xf4 are opt-in (not in the default ladder) for a stronger depth stress test:
# the deepest stacks are the ones most likely to destabilize a foldable norm.
VARIANTS = {
    "embed_unembed": dict(n_layers=0, use_mlp=False),
    "attn1":         dict(n_layers=1, use_mlp=False),
    "attn2":         dict(n_layers=2, use_mlp=False),
    "attn4":         dict(n_layers=4, use_mlp=False),
    "xf1":           dict(n_layers=1, use_mlp=True),
    "xf2":           dict(n_layers=2, use_mlp=True),
    "xf4":           dict(n_layers=4, use_mlp=True),
}
DEFAULT_VARIANTS = ["embed_unembed", "attn1", "attn2", "xf1", "xf2"]


# =============================================================================
# Model components (self-contained — no lib dependency)
# =============================================================================

class Rotary(nn.Module):
    """Rotary Position Embedding (RoPE). A fixed orthogonal linear map per position."""
    def __init__(self, dim, n_ctx, base=10000):
        super().__init__()
        freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        ctx = torch.arange(n_ctx).float()
        freqs = torch.einsum("i,j->ij", ctx, freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos_cached", torch.cat([cos, cos], dim=-1)[None, :, None, :], persistent=False)
        self.register_buffer("sin_cached", torch.cat([sin, sin], dim=-1)[None, :, None, :], persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        a, b = x.chunk(2, dim=-1)
        y = torch.cat((-b, a), dim=-1)
        return (x * self.cos_cached[:, :seq_len]) + (y * self.sin_cached[:, :seq_len])


class BilinearBatchNormAttention(nn.Module):
    """Bilinear attention with BatchNorm on Q1,K1,Q2,K2 (no biases), causal mask.

    pattern = (q1·k1)*(q2·k2) / d_head^2 * causal_mask, then z = pattern @ v.
    BatchNorm folds into the Q/K weights at inference -> tensor-network compatible.
    Residual is added internally (out = x + scale * o(z)).

    `rezero_init` (if not None) makes `scale` a learnable scalar initialized small
    (ReZero / SkipInit). It folds into `o` at inference, so TN-purity is preserved,
    and it tames the slow residual-stream drift that destabilizes deep static-rms
    stacks (see README). `None` keeps the fixed scale (legacy behavior).
    """
    def __init__(self, d_model, n_head, n_ctx, scale=1.0, rope_base=10000, rezero_init=None,
                 layer_norm="none"):
        super().__init__()
        self.d_head = d_model // n_head
        self.n_head = n_head
        self.scale = nn.Parameter(torch.tensor(float(rezero_init))) if rezero_init is not None else scale
        self.prenorm = make_norm(layer_norm, d_model)   # per-layer pre-norm (depth stabilizer)
        self.rotary = Rotary(self.d_head, n_ctx, base=rope_base)
        self.register_buffer("causal_mask", torch.tril(torch.ones(n_ctx, n_ctx)), persistent=False)

        self.q1 = nn.Linear(d_model, d_model, bias=False)
        self.k1 = nn.Linear(d_model, d_model, bias=False)
        self.q2 = nn.Linear(d_model, d_model, bias=False)
        self.k2 = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

        self.bn_q1 = nn.BatchNorm1d(d_model)
        self.bn_k1 = nn.BatchNorm1d(d_model)
        self.bn_q2 = nn.BatchNorm1d(d_model)
        self.bn_k2 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        B, T, _ = x.shape
        xn = self.prenorm(x)              # normalize block input; residual stays on raw x
        q1 = self.bn_q1(self.q1(xn).reshape(B * T, -1)).reshape(B, T, -1)
        k1 = self.bn_k1(self.k1(xn).reshape(B * T, -1)).reshape(B, T, -1)
        q2 = self.bn_q2(self.q2(xn).reshape(B * T, -1)).reshape(B, T, -1)
        k2 = self.bn_k2(self.k2(xn).reshape(B * T, -1)).reshape(B, T, -1)

        q1 = self.rotary(rearrange(q1, "b t (nh dh) -> b t nh dh", nh=self.n_head))
        k1 = self.rotary(rearrange(k1, "b t (nh dh) -> b t nh dh", nh=self.n_head))
        q2 = self.rotary(rearrange(q2, "b t (nh dh) -> b t nh dh", nh=self.n_head))
        k2 = self.rotary(rearrange(k2, "b t (nh dh) -> b t nh dh", nh=self.n_head))
        v = rearrange(self.v(xn), "b t (nh dh) -> b t nh dh", nh=self.n_head)

        scores1 = einsum(q1, k1, "b sq nh dh, b sk nh dh -> b nh sq sk")
        scores2 = einsum(q2, k2, "b sq nh dh, b sk nh dh -> b nh sq sk")
        pattern = (scores1 * scores2) / (self.d_head ** 2)
        pattern = pattern * self.causal_mask[None, None, :T, :T]

        z = einsum(pattern, v, "b nh sq sk, b sk nh dh -> b sq nh dh")
        z = rearrange(z, "b seq nh dh -> b seq (nh dh)")
        return x + self.scale * self.o(z)


class BilinearMLP(nn.Module):
    """Bilinear MLP: y = D(Lx ⊙ Rx), no biases. Degree-2 polynomial -> TN compatible.

    `rezero_init` (if not None) makes `scale` a learnable scalar (folds into `D`).
    """
    def __init__(self, d_model, d_hidden=None, scale=1.0, rezero_init=None, layer_norm="none"):
        super().__init__()
        d_hidden = d_hidden or 2 * d_model
        self.L = nn.Linear(d_model, d_hidden, bias=False)
        self.R = nn.Linear(d_model, d_hidden, bias=False)
        self.D = nn.Linear(d_hidden, d_model, bias=False)
        self.scale = nn.Parameter(torch.tensor(float(rezero_init))) if rezero_init is not None else scale
        self.prenorm = make_norm(layer_norm, d_model)

    def forward(self, x):
        xn = self.prenorm(x)
        return self.scale * self.D(self.L(xn) * self.R(xn))


class StaticRMSNorm(nn.Module):
    """RMS norm with a STATIC scale -> tensor-network foldable.

    Tracks an EMA of the global activation RMS (like BatchNorm running stats).
    At inference: x / running_rms * weight, a diagonal linear map that folds
    into the unembed (Unembed @ diag(weight / running_rms)). Contrast with
    nn.LayerNorm, whose per-sample 1/sqrt(var) cannot be folded.
    """
    def __init__(self, d_model, momentum=0.1, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.register_buffer("running_rms", torch.ones(1))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x):
        if self.training:
            rms = x.detach().pow(2).mean().sqrt()
            self.running_rms.mul_(1 - self.momentum).add_(self.momentum * rms)
            scale = rms
        else:
            scale = self.running_rms
        return x / (scale + self.eps) * self.weight


class RMSNorm(nn.Module):
    """Per-sample RMS norm: x / sqrt(mean(x^2, dim=-1)) * weight.

    Normalizes each token independently (like LayerNorm without mean-subtraction),
    so it is the effective inter-layer stabilizer for deep bilinear stacks — a global
    scalar (static-rms) cannot tame the per-token blow-up that degree-4 attention
    creates. NOT foldable on its own (per-sample); folds to a running scalar only if
    later frozen to StaticRMSNorm. Used as the per-layer pre-norm; the *final* norm is
    the swept ablation.
    """
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(kind, d_model):
    """Norm factory shared by the final norm and the per-layer pre-norm."""
    if kind == "layernorm":
        return nn.LayerNorm(d_model)      # NOT foldable (per-sample variance)
    if kind == "rmsnorm":
        return RMSNorm(d_model)           # per-sample; inter-layer depth stabilizer
    if kind == "static-rms":
        return StaticRMSNorm(d_model)     # foldable (running scalar)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")


make_final_norm = make_norm  # backwards-compatible alias


class SweepLM(nn.Module):
    """Config-driven bilinear LM: Embed -> [attn (+ mlp)] x n_layers -> norm -> Unembed."""
    def __init__(self, vocab_size, n_ctx, d_model, n_layers, use_mlp,
                 final_norm="layernorm", rope_base=10000,
                 std_embed=0.02, std_qkv=0.02, std_o=0.01, rezero_init=None,
                 layer_norm="none"):
        super().__init__()
        n_head = max(1, d_model // D_HEAD)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn_layers = nn.ModuleList([
            BilinearBatchNormAttention(d_model, n_head, n_ctx, rope_base=rope_base,
                                       rezero_init=rezero_init, layer_norm=layer_norm)
            for _ in range(n_layers)
        ])
        self.mlp_layers = nn.ModuleList([
            BilinearMLP(d_model, rezero_init=rezero_init, layer_norm=layer_norm) if use_mlp else None
            for _ in range(n_layers)
        ])
        self.final_norm = make_final_norm(final_norm, d_model)
        self.unembed = nn.Linear(d_model, vocab_size, bias=False)
        self._init_weights(std_embed, std_qkv, std_o)

    def _init_weights(self, std_embed, std_qkv, std_o):
        nn.init.normal_(self.embed.weight, std=std_embed)
        nn.init.normal_(self.unembed.weight, std=std_embed)
        for attn in self.attn_layers:
            for name in ("q1", "k1", "q2", "k2", "v"):
                nn.init.normal_(getattr(attn, name).weight, std=std_qkv)
            nn.init.normal_(attn.o.weight, std=std_o)
        for mlp in self.mlp_layers:
            if mlp is not None:
                nn.init.normal_(mlp.L.weight, std=std_qkv)
                nn.init.normal_(mlp.R.weight, std=std_qkv)
                nn.init.normal_(mlp.D.weight, std=std_o)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for attn, mlp in zip(self.attn_layers, self.mlp_layers):
            x = attn(x)
            if mlp is not None:
                x = x + mlp(x)
        return self.unembed(self.final_norm(x))


# =============================================================================
# Optimizer / scheduler
# =============================================================================

def create_optimizer(model, lr=3e-4, muon_lr=0.02, weight_decay=0.1,
                     betas=(0.9, 0.95), use_muon=False):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (nodecay if (p.ndim < 2 or "bn_" in n or "norm" in n) else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    if not use_muon:
        return torch.optim.AdamW(groups, lr=lr, betas=betas)
    # opt-in Muon for attention matrices (requires `muon` package)
    from muon import SingleDeviceMuonWithAuxAdam
    muon_params = [p for n, p in model.named_parameters()
                   if p.ndim >= 2 and "attn_layers" in n and ".bn_" not in n]
    muon_ids = {id(p) for p in muon_params}
    adam_decay = [p for p in decay if id(p) not in muon_ids]
    return SingleDeviceMuonWithAuxAdam([
        dict(params=muon_params, use_muon=True, lr=muon_lr, weight_decay=weight_decay),
        dict(params=adam_decay, use_muon=False, lr=lr, betas=betas, weight_decay=weight_decay),
        dict(params=nodecay, use_muon=False, lr=lr, betas=betas, weight_decay=0.0),
    ])


def create_scheduler(optimizer, warmup_steps, max_steps, lr_decay_frac=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = min(float(step - warmup_steps) / float(max(1, max_steps - warmup_steps)), 1.0)
        return lr_decay_frac + 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - lr_decay_frac)
    return LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Loss / eval
# =============================================================================

def compute_loss(logits, input_ids):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return F.cross_entropy(shift_logits.view(B * T, V), shift_labels.view(B * T))


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Eval-mode val loss (BatchNorm/StaticRMSNorm use running stats -> no leak)."""
    model.eval()
    total, n = 0.0, 0
    for batch in dataloader:
        ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            total += compute_loss(model(ids), ids).item()
        n += 1
    model.train()
    return total / max(1, n)


@torch.no_grad()
def most_learned_tokens(model, val_ids, device, top_k=20):
    """Datapoints (seq, pos) the model predicts best — lowest next-token CE."""
    model.eval()
    losses = []
    for start in range(0, val_ids.shape[0], 16):
        batch = val_ids[start:start + 16].to(device)
        logits = model(batch).float()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]),
                             batch[:, 1:].reshape(-1), reduction="none")
        losses.append(ce.reshape(batch.shape[0], -1).cpu())
    model.train()
    losses = torch.cat(losses, 0)
    flat = losses.flatten()
    vals, idx = flat.topk(top_k, largest=False)
    T = losses.shape[1]
    return [(int(i // T), int(i % T) + 1, float(v)) for i, v in zip(idx, vals)]


# =============================================================================
# Data
# =============================================================================

class CachedDataset(Dataset):
    def __init__(self, path, n_ctx, max_samples=None):
        data = torch.load(path, weights_only=True).to(torch.long)[:, :n_ctx]
        self.data = data[:max_samples] if max_samples else data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {"input_ids": self.data[idx]}


class DSIRPileStreaming(IterableDataset):
    """Real training data: streaming DSIR-filtered Pile, GPT-2 tokens mod vocab."""
    def __init__(self, n_ctx, vocab_size=VOCAB_SIZE):
        self.n_ctx = n_ctx
        self.vocab_size = vocab_size

    def __iter__(self):
        from datasets import load_dataset
        from transformers import GPT2Tokenizer
        tok = GPT2Tokenizer.from_pretrained("gpt2")
        ds = load_dataset("stanford-crfm/DSIR-filtered-pile-50M", split="train", streaming=True)
        buf = []
        for ex in ds:
            buf.extend(t % self.vocab_size for t in tok.encode(ex["contents"]))
            while len(buf) >= self.n_ctx:
                chunk, buf = buf[:self.n_ctx], buf[self.n_ctx:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}


def cycle(dataloader):
    """Yield batches forever — re-iterates finite (cached) loaders without caching."""
    while True:
        for batch in dataloader:
            yield batch


def make_loaders(data, n_ctx, batch_size, smoke_val_ids=None):
    """Returns (train_dl, val_dl, val_ids). `data` is 'cached' (smoke) or 'pile'."""
    val_ids = torch.load(PILE_VAL, weights_only=True).to(torch.long)[:, :n_ctx]
    val_dl = DataLoader(CachedDataset(PILE_VAL, n_ctx, max_samples=500),
                        batch_size=batch_size, drop_last=True)
    if data == "cached":
        # Smoke: train on the cached val tensor itself (wiring check, not a real run).
        train_dl = DataLoader(TensorDataset(val_ids), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    else:
        train_dl = DataLoader(DSIRPileStreaming(n_ctx), batch_size=batch_size, drop_last=True)
    return train_dl, val_dl, val_ids


# =============================================================================
# Diagnostics — localize where a (finite) failure originates: activations or weights
# =============================================================================

@torch.no_grad()
def collect_diagnostics(net, ids):
    """Per-layer activation per-token RMS (max = the blow-up signal) and per-matrix
    weight norms (Frobenius + top singular value). Eval-mode forward, no stat update."""
    was_training = net.training
    net.eval()
    rec = {}

    def ptrms_max(t):  # max per-token RMS across the batch — where activations explode
        return round(t.float().pow(2).mean(-1).sqrt().max().item(), 3)

    x = net.embed(ids)
    rec["act_embed"] = ptrms_max(x)
    for i, (attn, mlp) in enumerate(zip(net.attn_layers, net.mlp_layers)):
        x = attn(x)
        rec[f"act_L{i}_attn"] = ptrms_max(x)
        if mlp is not None:
            x = x + mlp(x)
            rec[f"act_L{i}_mlp"] = ptrms_max(x)

    def wnorms(W):
        return round(W.norm().item(), 3), round(torch.linalg.svdvals(W.float())[0].item(), 3)

    for i, attn in enumerate(net.attn_layers):
        for nm in ("q1", "k1", "q2", "k2", "v", "o"):
            fro, sv = wnorms(getattr(attn, nm).weight)
            rec[f"w_L{i}_{nm}_fro"], rec[f"w_L{i}_{nm}_sv"] = fro, sv
    for i, mlp in enumerate(net.mlp_layers):
        if mlp is not None:
            for nm in ("L", "R", "D"):
                fro, sv = wnorms(getattr(mlp, nm).weight)
                rec[f"w_L{i}_{nm}_fro"], rec[f"w_L{i}_{nm}_sv"] = fro, sv
    rec["w_unembed_fro"] = round(net.unembed.weight.norm().item(), 3)
    if was_training:
        net.train()
    return rec


def diag_summary(net, ids, tag):
    """One-line human summary: which layer's activations / which weight is largest."""
    rec = collect_diagnostics(net, ids)
    acts = {k: v for k, v in rec.items() if k.startswith("act_")}
    svs = {k: v for k, v in rec.items() if k.endswith("_sv")}
    hot_act = max(acts, key=acts.get)
    hot_w = max(svs, key=svs.get) if svs else ("n/a", 0)
    print(f"    diag {tag}: peak act-rms {hot_act}={acts[hot_act]} | "
          f"largest weight σ {hot_w}={svs.get(hot_w, 0)}")


# =============================================================================
# Train one config
# =============================================================================

def train_one(variant, d_model, final_norm, *, steps, batch_size, n_ctx,
              data, device, use_compile, use_muon, lr, rezero_init=None,
              layer_norm="none", diagnostics=False, run_dir=None, tag=None):
    cfg = VARIANTS[variant]
    torch.manual_seed(42)
    model = SweepLM(VOCAB_SIZE, n_ctx, d_model, cfg["n_layers"], cfg["use_mlp"],
                    final_norm=final_norm, rezero_init=rezero_init,
                    layer_norm=layer_norm).to(device)
    if use_compile:
        model = torch.compile(model)
    n_params = sum(p.numel() for p in model.parameters())

    # Pure AdamW (no Muon) needs a higher lr than the original scripts' 3e-4 aux lr,
    # which only applied to embeddings/norms while Muon drove the attn matrices at 0.02.
    optimizer = create_optimizer(model, lr=lr, use_muon=use_muon)
    scheduler = create_scheduler(optimizer, warmup_steps=min(100, steps // 5), max_steps=steps)
    train_dl, val_dl, val_ids = make_loaders(data, n_ctx, batch_size)

    net = getattr(model, "_orig_mod", model)   # unwrap torch.compile for diagnostics
    diag_path = diag_batch = None
    if diagnostics and run_dir is not None:
        (run_dir / "diag").mkdir(exist_ok=True)
        diag_path = run_dir / "diag" / f"{tag}.jsonl"
        diag_batch = val_ids[:32].to(device)        # fixed batch -> comparable across steps
    diag_every = max(1, steps // 25)

    init_val = evaluate(model, val_dl, device)
    model.train()
    step, t0, last_grad = 0, time.time(), float("nan")
    for batch in cycle(train_dl):
        if step >= steps:
            break
        ids = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch["input_ids"].to(device)
        if diag_path is not None and (step % diag_every == 0 or step == steps - 1):
            rec = {"step": step, "loss": round(loss.item(), 4) if step else None,
                   "grad_norm": round(float(last_grad), 4), **collect_diagnostics(net, diag_batch)}
            with open(diag_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        optimizer.zero_grad()
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = compute_loss(model(ids), ids)
        loss.backward()
        last_grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1

    final_val = evaluate(model, val_dl, device)
    if diag_path is not None:
        diag_summary(net, diag_batch, tag)
    return {
        "variant": variant, "d_model": d_model, "final_norm": final_norm,
        "n_params": n_params, "init_val": init_val, "final_val": final_val,
        "steps": step, "secs": round(time.time() - t0, 1),
        "val_ids": val_ids, "model": model,
    }


# =============================================================================
# Sweep
# =============================================================================

def run_sweep(*, variants, widths, norms, steps, batch_size, n_ctx,
              data, use_compile, use_muon, lr, save_checkpoints=False, top_tokens=0,
              rezero_init=None, layer_norm="none", diagnostics=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(__file__).parent / "runs" / f"{timestamp}_sweep"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "sweep.jsonl"
    if save_checkpoints:
        (run_dir / "checkpoints").mkdir(exist_ok=True)

    print(f"=== Bilinear architecture sweep ===")
    print(f"device={device} data={data} steps={steps} bs={batch_size} n_ctx={n_ctx} "
          f"compile={use_compile} muon={use_muon} rezero_init={rezero_init} "
          f"layer_norm={layer_norm} diagnostics={diagnostics}")
    print(f"variants={variants}  widths={widths}  norms={norms}  (final-norm is the ablation)")
    print(f"run_dir={run_dir}\n")

    results = []
    for norm in norms:
        for d_model in widths:
            for variant in variants:
                tag = f"{variant}_d{d_model}_{norm}"
                r = train_one(variant, d_model, norm, steps=steps, batch_size=batch_size,
                              n_ctx=n_ctx, data=data, device=device,
                              use_compile=use_compile, use_muon=use_muon, lr=lr,
                              rezero_init=rezero_init, layer_norm=layer_norm,
                              diagnostics=diagnostics, run_dir=run_dir, tag=tag)
                row = {k: r[k] for k in ("variant", "d_model", "final_norm", "n_params",
                                         "init_val", "final_val", "steps", "secs")}
                row["layer_norm"] = layer_norm
                # Step 3: log the datapoints each setting learns best (lowest next-token CE).
                if top_tokens:
                    learned = most_learned_tokens(r["model"], r["val_ids"], device, top_k=top_tokens)
                    row["top_learned"] = [{"seq": s, "pos": p, "ce": round(c, 4)}
                                          for s, p, c in learned]
                    best = ", ".join(f"(seq{s},pos{p}:{c:.3f})" for s, p, c in learned[:3])
                    print(f"    best-learned {tag}: {best}")
                # Step 4: checkpoint (unwrap torch.compile) for later mech-interp.
                if save_checkpoints:
                    net = getattr(r["model"], "_orig_mod", r["model"])
                    torch.save({"state_dict": net.state_dict(), "config": row},
                               run_dir / "checkpoints" / f"{tag}.pt")
                results.append(row)
                with open(log_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"  {variant:<14} d={d_model:<4} {norm:<11} "
                      f"params={r['n_params']:>10,}  "
                      f"val {r['init_val']:.3f} -> {r['final_val']:.3f}  ({r['secs']}s)")

    print_ordering(results)
    return results, run_dir


def print_ordering(results):
    """Check the monotonicity contract: more components -> lower loss."""
    order = list(VARIANTS)
    print("\n=== Monotonicity check (final val, lower is better) ===")
    by_group = {}
    for r in results:
        by_group.setdefault((r["d_model"], r["final_norm"]), {})[r["variant"]] = r["final_val"]
    # adding a component (fewer -> more) should not meaningfully RAISE loss.
    # tolerance is floor-aware: near the overfit floor, ~0.009 vs ~0.010 ties are fine.
    pairs = [("embed_unembed", "attn1"), ("attn1", "attn2"), ("attn2", "attn4"),
             ("attn1", "xf1"), ("attn2", "xf2"), ("xf2", "xf4")]
    all_ok = True
    for (d_model, norm), vals in sorted(by_group.items()):
        seq = [(v, vals[v]) for v in order if v in vals]
        line = "  ".join(f"{v}={l:.3f}" for v, l in seq)
        diverged = [v for v, l in vals.items() if not math.isfinite(l)]   # nan/inf = failed run
        violations = [f"{more}>{fewer}"
                      for fewer, more in pairs
                      if fewer in vals and more in vals
                      and math.isfinite(vals[fewer]) and math.isfinite(vals[more])
                      and vals[more] > vals[fewer] + 0.02 + 0.05 * abs(vals[fewer])]
        all_ok &= not violations and not diverged
        flags = (["XX " + ",".join(violations)] if violations else []) + \
                ([f"DIVERGED:{','.join(diverged)}"] if diverged else [])
        flag = "  ".join(flags) if flags else "OK "
        print(f"  d={d_model} {norm:<11} {flag:<28} {line}")
    print(f"\n{'✓ all monotonic' if all_ok else '✗ NON-MONOTONIC / DIVERGED — investigate (bug, instability, or undertrained)'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true",
                   help="tiny wiring check on cached data (no network, ~seconds)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-ctx", type=int, default=512)
    p.add_argument("--widths", type=str, default="128")
    p.add_argument("--norms", type=str, default="layernorm")
    p.add_argument("--variants", type=str, default=",".join(DEFAULT_VARIANTS),
                   help=f"comma-separated; available: {','.join(VARIANTS)} "
                        "(attn4/xf4 are opt-in depth stress tests)")
    p.add_argument("--data", choices=["cached", "pile"], default="pile")
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--muon", action="store_true", help="use Muon (requires `muon` pkg)")
    p.add_argument("--save-checkpoints", action="store_true",
                   help="save per-config state_dicts (torch.compile-unwrapped) for mech-interp")
    p.add_argument("--top-tokens", type=int, default=0,
                   help="log the N (seq,pos) datapoints each config predicts best (0=off)")
    p.add_argument("--rezero-init", type=float, default=None,
                   help="ReZero: learnable residual scale init (foldable). ~0.25 stabilizes "
                        "deep static-rms; None=fixed scale 1.0 (default)")
    p.add_argument("--layer-norm", type=str, default="none",
                   choices=["none", "rmsnorm", "static-rms", "layernorm"],
                   help="per-layer pre-norm on every block (depth stabilizer). "
                        "rmsnorm needed for deep stacks; the *final* norm (--norms) is the ablation")
    p.add_argument("--diagnostics", action="store_true",
                   help="log per-layer activation RMS + per-matrix weight norms over training "
                        "to runs/<ts>/diag/<tag>.jsonl (localizes where a finite failure originates)")
    args = p.parse_args()

    if args.smoke:
        args.data = "cached"
        args.steps = min(args.steps, 60)
        args.n_ctx = min(args.n_ctx, 128)
        args.batch_size = min(args.batch_size, 16)
        args.no_compile = True
        args.norms = "layernorm,static-rms,none"

    run_sweep(
        variants=args.variants.split(","),
        widths=[int(w) for w in args.widths.split(",")],
        norms=args.norms.split(","),
        steps=args.steps, batch_size=args.batch_size, n_ctx=args.n_ctx,
        data=args.data, use_compile=not args.no_compile, use_muon=args.muon, lr=args.lr,
        save_checkpoints=args.save_checkpoints, top_tokens=args.top_tokens,
        rezero_init=args.rezero_init, layer_norm=args.layer_norm, diagnostics=args.diagnostics,
    )


if __name__ == "__main__":
    main()
