"""Train the depth ladder on the k-hop retrieval task and measure per-hop accuracy.

Architectures (bilinear attn + bilinear MLP, polynomial, norm off):
    attn2          = ["attn","attn"]           baseline (induction depth)
    attn-mlp-attn  = ["attn","mlp","attn"]      2 attn + 1 middle bilinear MLP
    attn3          = ["attn","attn","attn"]     3 attn

Loss is cross-entropy at the answer positions only (the k-hop targets). Per-hop top-1
accuracy is measured on a fresh held-out batch. Multiple seeds.

Usage: python hop_train.py 0 1 2      (seeds; default 0)
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from deep_model import DeepModel, SPECS
from hop_data import sample_docs, score_by_hop, ANS_POS, V, N_CTX, K_MAX

D_MODEL = 128
N_HEAD = 4
STEPS = 30000
BATCH = 128


def evaluate(model, seed=12345, n=512):
    g = torch.Generator().manual_seed(seed)
    tok, qa, qk = sample_docs(n, g)
    model.eval()
    with torch.no_grad():
        logits = model(tok[:, :-1].to(next(model.parameters()).device)).cpu()
    return score_by_hop(logits, qa, qk)


if __name__ == "__main__":
    device = "cuda"
    seeds = [int(s) for s in sys.argv[1:]] or [0]
    ans_pos = ANS_POS.to(device)
    results = {}
    outdir = Path("runs_hop")
    outdir.mkdir(exist_ok=True)
    for seed in seeds:
        for spec_name, spec in SPECS.items():
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed + 1000)
            name = f"{spec_name}-seed{seed}"
            if (outdir / name / "model.pt").exists():
                print(f"skip {name} (exists)", flush=True)
                results[name] = json.loads((outdir / name / "acc.json").read_text())
                continue
            # add residual (x + o(z)) for attention too: lerp halves the stream each layer
            # since bilinear attn ≈ 0 at init, which collapses 3-deep stacks at lr 1e-3
            # (established in earlier sessions). add keeps the model polynomial.
            model = DeepModel(V, D_MODEL, N_HEAD, spec, N_CTX,
                              attention="bilinear", residual="add", mlp_residual="add").to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min((s + 1) / 200, 0.5 * (1 + math.cos(math.pi * s / STEPS))))
            model.train()
            for step in range(STEPS):
                tok, qa, qk = sample_docs(BATCH, gen)
                tok = tok.to(device)
                logits = model(tok[:, :-1])
                # CE at answer positions only
                lg = logits[:, ans_pos]                       # (B, N_Q, V)
                tgt = tok[:, ans_pos + 1]                      # (B, N_Q)
                loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
                if step % 6000 == 0:
                    acc = evaluate(model)
                    model.train()
                    print(f"{name} step {step} loss {loss.item():.3f} "
                          f"acc {[round(acc[k],2) for k in range(K_MAX+1)]}", flush=True)
            acc = evaluate(model)
            (outdir / name).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), outdir / name / "model.pt")
            (outdir / name / "config.json").write_text(json.dumps({
                "spec": spec, "d_model": D_MODEL, "n_head": N_HEAD, "vocab": V}))
            (outdir / name / "acc.json").write_text(json.dumps(acc))
            print(f"== {name}: acc by hop {[round(acc[k],3) for k in range(K_MAX+1)]}", flush=True)
            results[name] = acc
            (outdir / "results.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
