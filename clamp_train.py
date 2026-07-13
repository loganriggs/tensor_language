"""Clamping-through-training (Singh et al.'s optogenetics), bilinear version.

Trains tiny-corpus attn2 with layer-0 head-0's attention PATTERN clamped to perfect
previous-token attention (one-hot at offset −1) for the whole run — the paper's
"PT-attend" clamp (their Fig 5b orange). Gradients still flow through V/O of the
clamped head and everything else. Comparison of the gated-token formation curve vs the
unclamped dense replay isolates how much of formation time is spent DISCOVERING the
previous-token substrate vs building the match/copy on top.

Usage: TL_CORPUS=tiny python clamp_train.py [seed=0]
Writes runs_lm/attn2-clampL0-dense-seed<k>/ (dense early ckpts, same recipe/data order).
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deep_model import DeepModel
from lm_train import quick_val_ce, BATCH, LR, EVAL_EVERY, CKPT_EVERY, D_MODEL, N_HEAD, NORM
from text_data import VOCAB, N_CTX, RUNS, tokens, get_batch

STEPS = 40_000


def clamp_layer0_head0(layer, n_ctx):
    """Replace head 0's pattern rows with one-hot previous-token attention."""
    prev = torch.zeros(n_ctx, n_ctx)
    idx = torch.arange(1, n_ctx)
    prev[idx, idx - 1] = 1.0
    prev = prev.cuda()
    orig_pattern = layer.pattern

    def pattern(x):
        p = orig_pattern(x)
        seq = x.size(-2)
        p = p.clone()
        p[:, 0] = prev[:seq, :seq]
        return p
    layer.pattern = pattern


def main(seed=0):
    tag = f"attn2-clampL0-dense-seed{seed}"
    out = RUNS / tag
    if (out / "model.pt").exists():
        print(f"skip {tag}")
        return
    (out / "ckpt").mkdir(parents=True, exist_ok=True)
    train, val = tokens("train"), tokens("val")
    torch.manual_seed(seed)
    data_gen = torch.Generator().manual_seed(10_000 + seed)
    model = DeepModel(VOCAB, D_MODEL, N_HEAD, ["attn", "attn"], N_CTX, norm=NORM,
                      attention="bilinear", residual="lerp").to("cuda")
    clamp_layer0_head0(model.layers[0], N_CTX)
    (out / "config.json").write_text(json.dumps({
        "spec": ["attn", "attn"], "d_model": D_MODEL, "n_head": N_HEAD, "vocab": VOCAB,
        "n_ctx": N_CTX, "norm": NORM, "residual": "lerp", "attention": "bilinear",
        "steps": STEPS, "batch": BATCH, "lr": LR, "seed": seed,
        "clamp": "L0H0=prev-token pattern"}))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.5 * (1 + math.cos(math.pi * s / STEPS))))
    hist = open(out / "history.jsonl", "a")
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        b = get_batch(train, BATCH, data_gen, "cuda")
        loss = F.cross_entropy(model(b[:, :-1]).reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % EVAL_EVERY == 0:
            vce = quick_val_ce(model, val)
            hist.write(json.dumps({"step": step, "val_ce": round(vce, 4)}) + "\n")
            hist.flush()
            print(f"{tag} step {step} val {vce:.3f} ({time.time()-t0:.0f}s)", flush=True)
        if step and (step % CKPT_EVERY == 0 or (step <= 5000 and step % 250 == 0)):
            torch.save(model.state_dict(), out / "ckpt" / f"step{step}.pt")
    torch.save(model.state_dict(), out / "model.pt")
    hist.write(json.dumps({"step": STEPS, "val_ce": round(quick_val_ce(model, val), 4)}) + "\n")
    hist.close()
    print(f"== {tag} done ({(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if sys.argv[1:] else 0)
