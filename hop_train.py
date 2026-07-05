"""Train the depth ladder on the k-hop retrieval task and measure per-hop accuracy.

Architectures (bilinear attn + bilinear MLP, polynomial, norm off):
    attn2          = ["attn","attn"]           baseline (induction depth)
    attn-mlp-attn  = ["attn","mlp","attn"]      2 attn + 1 middle bilinear MLP
    attn3          = ["attn","attn","attn"]     3 attn

Loss is cross-entropy at the answer positions only (the k-hop targets). Per-hop top-1
accuracy is measured on a fresh held-out batch. Multiple seeds.

Usage: python hop_train.py 0 1 2                 (seeds; all specs)
       python hop_train.py attn2 0 1 2           (one spec, listed seeds -> parallelizable)
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
STEPS = 40000
BATCH = 128
NORM = "rms"          # RMSNorm (pre-norm, affine-free) — renormalizes each layer so deep
                      # bilinear stacks don't explode; runs tagged "-rms". None = no norm.


def evaluate(model, seed=12345, n=512):
    g = torch.Generator().manual_seed(seed)
    tok, qa, qk = sample_docs(n, g)
    model.eval()
    with torch.no_grad():
        logits = model(tok[:, :-1].to(next(model.parameters()).device)).cpu()
    return score_by_hop(logits, qa, qk)


if __name__ == "__main__":
    device = "cuda"
    args = sys.argv[1:]
    spec_filter = [a for a in args if a in SPECS]
    seeds = [int(a) for a in args if a not in SPECS] or [0]
    run_specs = {k: v for k, v in SPECS.items() if (not spec_filter or k in spec_filter)}
    ans_pos = ANS_POS.to(device)
    results = {}
    outdir = Path("runs_hop")
    outdir.mkdir(exist_ok=True)
    for seed in seeds:
        for spec_name, spec in run_specs.items():
            torch.manual_seed(seed)
            gen = torch.Generator().manual_seed(seed + 1000)
            name = f"{spec_name}{'-rms' if NORM else ''}-seed{seed}"
            if (outdir / name / "model.pt").exists():
                print(f"skip {name} (exists)", flush=True)
                results[name] = json.loads((outdir / name / "acc.json").read_text())
                continue
            # lerp attention residual (0.5x + 0.5 o(z)) learns induction reliably here;
            # add stalled on the copy-only plateau. mlp uses add. RMSNorm (NORM) renormalizes
            # each layer so deep bilinear stacks stay finite (replaces the grad-clip stopgap).
            model = DeepModel(V, D_MODEL, N_HEAD, spec, N_CTX, norm=NORM,
                              attention="bilinear", residual="lerp", mlp_residual="add").to(device)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda s: min((s + 1) / 200, 0.5 * (1 + math.cos(math.pi * s / STEPS))))
            model.train()
            for step in range(STEPS):
                tok, qa, qk = sample_docs(BATCH, gen)
                tok = tok.to(device)
                logits = model(tok[:, :-1])
                # CE at answer positions only: binding values are unpredictable, so dense
                # loss floods the gradient with noise and stalls induction formation.
                lg = logits[:, ans_pos]                       # (B, N_Q, V)
                tgt = tok[:, ans_pos + 1]                      # (B, N_Q)
                loss = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1))
                opt.zero_grad(); loss.backward()
                # without normalization, deep bilinear stacks explode (degree ~2^depth);
                # grad-clip only as a fallback when NORM is off. RMSNorm handles it directly.
                if not NORM:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
                if step % 6000 == 0:
                    acc = evaluate(model)
                    model.train()
                    print(f"{name} step {step} loss {loss.item():.3f} "
                          f"acc {[round(acc[k],2) for k in range(K_MAX+1)]}", flush=True)
            acc = evaluate(model)
            (outdir / name).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), outdir / name / "model.pt")
            (outdir / name / "config.json").write_text(json.dumps({
                "spec": spec, "d_model": D_MODEL, "n_head": N_HEAD, "vocab": V, "norm": NORM}))
            (outdir / name / "acc.json").write_text(json.dumps(acc))
            print(f"== {name}: acc by hop {[round(acc[k],3) for k in range(K_MAX+1)]}", flush=True)
            results[name] = acc
            # per-model acc.json is the source of truth; aggregate written per-process
            # under a unique name so parallel spec-runs don't clobber each other.
            tag = "-".join(sorted(run_specs)) if spec_filter else "all"
            (outdir / f"results-{tag}.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(results, indent=1))
