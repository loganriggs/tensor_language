"""Isolate the bilinear match-and-copy (induction) circuit vs vocab size.

Every bilinear config on OWT (V=5120; d128 40k/120k, d256) fails to form induction,
while TinyStories (V=1024) formed it easily. Competing explanations:
  (a) the bilinear product-score match DEGRADES with vocab size (embedding geometry /
      score resolution) — an architecture limitation;
  (b) optimization competition — the statistics basin on OWT outcompetes the circuit.

Test (a) in isolation: pure deterministic copy. Each sequence = [u ; u] with u random
(length N_CTX/2, iid uniform over V). Loss on the second half only, where every target
is determined by matching the current token's earlier occurrence (exact induction; CE
floor = 0; no statistics available — u is iid uniform). Grid over V and d_model,
bilinear attn2, same recipe.

Usage: python copy_isolation.py           -> runs_markov/copy_isolation.json
"""

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from deep_model import DeepModel

N_CTX = 128
HALF = N_CTX // 2
STEPS = 8_000
BATCH = 64
OUT = Path("runs_markov")


def batch(V, n, g, device="cuda"):
    u = torch.randint(V, (n, HALF), generator=g)
    toks = torch.cat([u, u], 1)
    return toks.to(device)


def train_one(V, d_model, seed=0, device="cuda"):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    model = DeepModel(V, d_model, 4, ["attn", "attn"], N_CTX, norm="rms",
                      attention="bilinear", residual="lerp").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.5 * (1 + math.cos(math.pi * s / STEPS))))
    hist = []
    model.train()
    for step in range(STEPS):
        toks = batch(V, BATCH, g, device)
        logits = model(toks[:, :-1])
        loss = F.cross_entropy(logits[:, HALF - 1:].reshape(-1, V),
                               toks[:, HALF:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 500 == 0:
            hist.append(round(loss.item(), 4))
    ge = torch.Generator().manual_seed(99)
    toks = batch(V, 256, ge, device)
    model.eval()
    with torch.no_grad():
        ce = F.cross_entropy(model(toks[:, :-1])[:, HALF - 1:].reshape(-1, V),
                             toks[:, HALF:].reshape(-1)).item()
    tag = f"V{V}-d{d_model}-seed{seed}"
    print(f"copy {tag}: final CE {ce:.4f} (floor 0, chance {math.log(V):.2f}) "
          f"curve {hist[::4]}", flush=True)
    return tag, ce, hist


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    res = {}
    for V in (1024, 5120):
        for d in (128, 256):
            tag, ce, hist = train_one(V, d)
            res[tag] = {"ce": ce, "curve": hist}
            (OUT / "copy_isolation.json").write_text(json.dumps(res, indent=1))
