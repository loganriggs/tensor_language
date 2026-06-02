#!/usr/bin/env python3
"""Poly-softmax experiments on a small nanoGPT-style char model (see
`poly_softmax_experiments.md`). Two one-variable studies on tiny-Shakespeare:

  Exp 1 — swap the ATTENTION softmax for a poly-softmax (taylor/spherical), RMSNorm kept.
  Exp 2 — swap RMSNorm for SoftmaxNorm (poly-softmax over features + rescale), softmax attn kept.

Change exactly one component per run; same seed/data/init/schedule. Logs train+val CE vs
step and plots each experiment's runs on shared axes. Standard softmax-CE is the metric for
ALL runs (only internal components change, never the loss).

Usage: python poly_softmax_gpt.py --experiment both --max-iters 3000
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
DATA = HERE / "data" / "shakespeare_input.txt"


# ---------------------------------------------------------------- poly-softmax
def poly_softmax(z, dim=-1, kind="taylor", keep=None, eps=1e-6):
    """Spherical-family softmax. `keep`: optional 0/1 mask applied multiplicatively
    (NOT via -inf — these numerators are non-negative)."""
    if kind == "taylor":
        num = 1.0 + z + 0.5 * z * z          # 2nd-order Taylor of exp; > 0 everywhere
    elif kind == "spherical":
        num = z * z                          # >= 0, loses sign (negative control)
    else:
        raise ValueError(kind)
    if keep is not None:
        num = num * keep
    denom = num.sum(dim=dim, keepdim=True).clamp_min(eps)
    return num / denom


# ---------------------------------------------------------------- norms
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SoftmaxNorm(nn.Module):
    """Drop-in for RMSNorm: poly-softmax across features, then rescale (softmax output
    ~1/dim, so multiply by ~dim to restore unit RMS). Scale is learnable, init at dim."""
    def __init__(self, dim, kind="taylor", scale=None):
        super().__init__()
        self.kind = kind
        self.weight = nn.Parameter(torch.ones(dim))
        self.log_scale = nn.Parameter(torch.log(torch.tensor(float(dim if scale is None else scale))))

    def forward(self, x):
        p = poly_softmax(x, dim=-1, kind=self.kind)
        return self.log_scale.exp() * p * self.weight


def make_norm(dim, norm_kind):
    return RMSNorm(dim) if norm_kind == "rmsnorm" else SoftmaxNorm(dim, kind=norm_kind)


# ---------------------------------------------------------------- model
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg["n_head"]
        self.attn_kind = cfg["attn_kind"]
        self.c_attn = nn.Linear(cfg["n_embd"], 3 * cfg["n_embd"], bias=False)
        self.c_proj = nn.Linear(cfg["n_embd"], cfg["n_embd"], bias=False)
        self.attn_dropout = nn.Dropout(cfg["dropout"])
        self.resid_dropout = nn.Dropout(cfg["dropout"])
        self.register_buffer("bias", torch.tril(torch.ones(cfg["block_size"], cfg["block_size"]))
                             .view(1, 1, cfg["block_size"], cfg["block_size"]))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        keep = self.bias[:, :, :T, :T]
        if self.attn_kind == "softmax":
            att = att.masked_fill(keep == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
        elif self.attn_kind == "none":
            att = att * keep                 # raw masked scores, NO weight normalization
        else:                                # bilinear's default behavior, 1 qk pair
            att = poly_softmax(att, dim=-1, kind=self.attn_kind, keep=keep)
        att = self.attn_dropout(att)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.c_fc = nn.Linear(cfg["n_embd"], 4 * cfg["n_embd"], bias=False)
        self.c_proj = nn.Linear(4 * cfg["n_embd"], cfg["n_embd"], bias=False)
        self.dropout = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = make_norm(cfg["n_embd"], cfg["norm_kind"])
        self.attn = CausalSelfAttention(cfg)
        self.n2 = make_norm(cfg["n_embd"], cfg["norm_kind"])
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        x = x + self.mlp(self.n2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg["vocab_size"], cfg["n_embd"])
        self.wpe = nn.Embedding(cfg["block_size"], cfg["n_embd"])
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg["n_layer"])])
        self.norm_f = make_norm(cfg["n_embd"], cfg["norm_kind"])
        self.lm_head = nn.Linear(cfg["n_embd"], cfg["vocab_size"], bias=False)
        self.wte.weight = self.lm_head.weight          # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("c_proj.weight"):            # scaled residual init (nanoGPT)
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg["n_layer"]))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for b in self.blocks:
            x = b(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


# ---------------------------------------------------------------- data
def load_data():
    if not DATA.exists():
        import urllib.request
        DATA.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"downloading tiny-shakespeare -> {DATA}")
        urllib.request.urlretrieve(url, DATA)
    text = DATA.read_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(split_data, block_size, batch_size, device):
    ix = torch.randint(len(split_data) - block_size, (batch_size,))
    x = torch.stack([split_data[i:i + block_size] for i in ix])
    y = torch.stack([split_data[i + 1:i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_ce(model, train_d, val_d, cfg, device, iters=20):
    model.eval()
    out = {}
    for name, d in (("train", train_d), ("val", val_d)):
        tot = 0.0
        for _ in range(iters):
            xb, yb = get_batch(d, cfg["block_size"], cfg["batch_size"], device)
            with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
                _, loss = model(xb, yb)
            tot += loss.item()
        out[name] = tot / iters
    model.train()
    return out


# ---------------------------------------------------------------- train
def train_run(label, attn_kind, norm_kind, base_cfg, *, max_iters, eval_interval, lr, device, seed=1337):
    torch.manual_seed(seed)
    cfg = {**base_cfg, "attn_kind": attn_kind, "norm_kind": norm_kind}
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(it):  # warmup + cosine
        warm = 100
        if it < warm:
            return lr * (it + 1) / warm
        prog = (it - warm) / max(1, max_iters - warm)
        return 0.1 * lr + 0.5 * (1 + math.cos(math.pi * prog)) * 0.9 * lr

    train_d, val_d, _ = load_data()
    hist = {"step": [], "train": [], "val": []}
    t0 = time.time()
    for it in range(max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        if it % eval_interval == 0 or it == max_iters:
            ce = eval_ce(model, train_d, val_d, cfg, device)
            hist["step"].append(it); hist["train"].append(ce["train"]); hist["val"].append(ce["val"])
            print(f"  [{label}] iter {it:5d}  train {ce['train']:.4f}  val {ce['val']:.4f}", flush=True)
        if it == max_iters:
            break
        xb, yb = get_batch(train_d, cfg["block_size"], cfg["batch_size"], device)
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    hist["secs"] = round(time.time() - t0, 1)
    hist["final_val"] = round(hist["val"][-1], 4)
    hist["best_val"] = round(min(hist["val"]), 4)
    print(f"  [{label}] done: final val {hist['final_val']}  best val {hist['best_val']}  ({hist['secs']}s)", flush=True)
    return hist


def plot_ce(history, title, out_png):
    plt.figure(figsize=(8, 5))
    for label, h in history.items():
        line, = plt.plot(h["step"], h["val"], label=f"{label}  best={h['best_val']} final={h['final_val']}", lw=1.8)
        plt.plot(h["step"], h["train"], ls="--", alpha=0.45, color=line.get_color())
    plt.xlabel("step"); plt.ylabel("cross-entropy loss (solid=val, dashed=train)")
    plt.title(title); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, dpi=140)
    print(f"plot saved: {out_png}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", choices=["1", "2", "both"], default="both")
    p.add_argument("--max-iters", type=int, default=3000)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-layer", type=int, default=6)
    p.add_argument("--n-head", type=int, default=6)
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, _, vocab = load_data()
    base = dict(vocab_size=vocab, n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                block_size=args.block_size, batch_size=args.batch_size, dropout=args.dropout)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "runs" / f"{ts}_polysoftmax"; out.mkdir(parents=True, exist_ok=True)
    common = dict(max_iters=args.max_iters, eval_interval=args.eval_interval, lr=args.lr, device=device)
    print(f"device={device} vocab={vocab} cfg={base}\nout={out}\n")

    all_hist = {}
    if args.experiment in ("1", "both"):
        print("=== Experiment 1: attention softmax variants (RMSNorm kept) ===")
        h1 = {}
        for label, ak in [("no-softmax / raw (bilinear default)", "none"),
                          ("softmax (baseline)", "softmax"),
                          ("taylor", "taylor"), ("spherical", "spherical")]:
            h1[label] = train_run(label, ak, "rmsnorm", base, **common)
        plot_ce(h1, "Exp 1: attention softmax variants (RMSNorm kept)", out / "exp1_attention.png")
        all_hist["exp1"] = h1
    if args.experiment in ("2", "both"):
        print("\n=== Experiment 2: norm variants (softmax attention kept) ===")
        h2 = {}
        for label, nk in [("rmsnorm (baseline)", "rmsnorm"), ("taylor", "taylor"), ("spherical", "spherical")]:
            h2[label] = train_run(label, "softmax", nk, base, **common)
        plot_ce(h2, "Exp 2: norm variants (softmax attention kept)", out / "exp2_norm.png")
        all_hist["exp2"] = h2

    with open(out / "history.json", "w") as f:
        json.dump({"cfg": base, "args": vars(args), "history": all_hist}, f)
    print(f"\nhistory: {out/'history.json'}")


if __name__ == "__main__":
    main()
