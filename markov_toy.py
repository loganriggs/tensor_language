"""Toy isolation of the n-gram-order circuit family (results_deeper.md): which bilinear
depth learns order-k Markov statistics?

Data: ONE global order-k Markov chain per (k, data_seed) — a fixed transition table
mapping each k-token context to a sparse next-token distribution (Dirichlet(0.05) over
V=64). Purely in-weights knowledge: documents are independent chains from the same
table, so there is nothing to copy from context. The exact entropy floor is the mean
conditional entropy over visited contexts.

Models: the same bilinear recipe as the text ladder (RMSNorm, lerp attn residual, add
MLP residual), depths attn1-4 + block1/block2 (attn+bilinear-MLP blocks).

Prediction (from the OWT/tiny gate analysis): attn-d reaches the order-k floor iff
d >= k+1? — the text result says 2 attn layers do NOT get order-3 (trigram) statistics
while 3 layers do; this pins the mapping exactly and per-architecture.

Usage: python markov_toy.py [orders=1,2,3] [specs=attn1,attn2,attn3,attn4,block1,block2]
Writes runs_markov/results.json + prints the gap-to-floor table.
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deep_model import DeepModel, SPECS

V = 64
N_CTX = 128
D_MODEL = 128
N_HEAD = 4
STEPS = 12_000
BATCH = 128
LR = 1e-3
OUT = Path("runs_markov")


def make_table(order, gen):
    """Transition table: (V^order, V) probabilities, Dirichlet(0.05)."""
    g = torch.Generator().manual_seed(gen)
    alpha = torch.full((V,), 0.05)
    probs = torch._sample_dirichlet(alpha.expand(V ** order, V).contiguous(),
                                    generator=g)
    return probs


def sample_chains(table, order, batch, length, g, device="cpu"):
    """Sample (batch, length) chains from the global table; also return the exact
    per-position conditional entropy (for the floor)."""
    toks = torch.zeros(batch, length, dtype=torch.long)
    toks[:, :order] = torch.randint(V, (batch, order), generator=g)
    powers = V ** torch.arange(order - 1, -1, -1)
    ctx = (toks[:, :order] * powers).sum(1)
    ent = torch.zeros(batch, length)
    for i in range(order, length):
        p = table[ctx]
        toks[:, i] = torch.multinomial(p, 1, generator=g).squeeze(1)
        ent[:, i] = -(p * (p + 1e-12).log()).sum(1)
        ctx = (ctx % (V ** (order - 1))) * V + toks[:, i] if order > 1 else toks[:, i]
    return toks.to(device), ent


POOL = 16_384        # pregenerated chains per dataset; resampled rows per step. The
                     # table (in-weights target) is fixed, so chain reuse is harmless.


def train_one(spec_name, order, seed=0, data_seed=7, device="cuda"):
    tag = f"{spec_name}-k{order}-seed{seed}"
    table = make_table(order, data_seed)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(1000 + seed)
    pool, _ = sample_chains(table, order, POOL, N_CTX + 1, g, device)
    model = DeepModel(V, D_MODEL, N_HEAD, SPECS[spec_name], N_CTX, norm="rms",
                      attention="bilinear", residual="lerp", mlp_residual="add").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.5 * (1 + math.cos(math.pi * s / STEPS))))
    model.train()
    for step in range(STEPS):
        toks = pool[torch.randint(POOL, (BATCH,), generator=g)]
        logits = model(toks[:, :-1])
        # score only positions with a full true context
        loss = F.cross_entropy(logits[:, order - 1:].reshape(-1, V),
                               toks[:, order:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if not torch.isfinite(loss):
            return tag, float("nan"), float("nan")
    ge = torch.Generator().manual_seed(99)
    toks, ent = sample_chains(table, order, 512, N_CTX + 1, ge, device)
    model.eval()
    with torch.no_grad():
        logits = model(toks[:, :-1])
        ce = F.cross_entropy(logits[:, order - 1:].reshape(-1, V),
                             toks[:, order:].reshape(-1)).item()
    floor = ent[:, order:].mean().item()
    print(f"{tag}: CE {ce:.3f} floor {floor:.3f} gap {ce - floor:.3f}", flush=True)
    return tag, ce, floor


if __name__ == "__main__":
    a = sys.argv[1:]
    orders = [int(x) for x in (a[0].split(",") if a else "1,2,3".split(","))]
    specs = (a[1] if len(a) > 1 else "attn1,attn2,attn3,attn4,block1,block2").split(",")
    OUT.mkdir(exist_ok=True)
    resfile = OUT / "results.json"
    results = json.loads(resfile.read_text()) if resfile.exists() else {}
    for order in orders:
        for spec in specs:
            tag = f"{spec}-k{order}-seed0"
            if tag in results:
                continue
            _, ce, floor = train_one(spec, order)
            results[tag] = {"ce": ce, "floor": floor, "gap": ce - floor}
            resfile.write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
