"""Reliability lever test: does a HOP CURRICULUM make the chained-retrieval circuit reliable?
The higher-hop training is a seed lottery (attn3 1/3, attn4 also fails on seeds) — optimization
sometimes lands on the copy-only/induction plateau instead of the pointer-advance circuit. A
curriculum (train hop<=1 first, then add hop-2, then hop-3) may scaffold the pointer advance so it
forms reliably. Train attn3 with the curriculum across several seeds; report per-hop acc and how
many seeds solve hop-3 (vs the ~1/3 no-curriculum baseline).

Run: python hop_curriculum.py 0 1 2 3
"""

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from deep_model import DeepModel, SPECS
from hop_data import sample_docs, score_by_hop, ANS_POS, V, N_CTX, K_MAX

D_MODEL, N_HEAD, STEPS, BATCH, NORM = 128, 4, 30000, 128, "rms"
SPEC = ["attn", "attn", "attn"]


def evaluate(model, seed=12345, n=512):
    g = torch.Generator().manual_seed(seed)
    tok, qa, qk = sample_docs(n, g)
    model.eval()
    with torch.no_grad():
        logits = model(tok[:, :-1].to(next(model.parameters()).device)).cpu()
    return score_by_hop(logits, qa, qk)


def run(seeds):
    device = "cuda"
    ans_pos = ANS_POS.to(device)
    solved = 0
    out = {}
    for seed in seeds:
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed + 1000)
        model = DeepModel(V, D_MODEL, N_HEAD, SPEC, N_CTX, norm=NORM,
                          attention="bilinear", residual="lerp", mlp_residual="add").to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min((s + 1) / 200, 0.5 * (1 + math.cos(math.pi * s / STEPS))))
        model.train()
        for step in range(STEPS):
            cur_max = min(K_MAX, 1 + step * 3 // STEPS)        # 1 -> 2 -> 3 over thirds
            tok, qa, qk = sample_docs(BATCH, gen)
            tok = tok.to(device); qk = qk.to(device)
            logits = model(tok[:, :-1])
            lg = logits[:, ans_pos]                            # (B, N_Q, V)
            tgt = tok[:, ans_pos + 1]                          # (B, N_Q)
            mask = (qk <= cur_max).reshape(-1)                 # curriculum: only hops <= cur_max
            ce = F.cross_entropy(lg.reshape(-1, V)[mask], tgt.reshape(-1)[mask])
            opt.zero_grad(); ce.backward(); opt.step(); sched.step()
        acc = evaluate(model)
        out[f"seed{seed}"] = {str(k): round(acc[k], 3) for k in range(K_MAX + 1)}
        ok = acc[3] > 0.6
        solved += ok
        print(f"curriculum attn3 seed{seed}: acc {[round(acc[k],2) for k in range(K_MAX+1)]} "
              f"{'SOLVED hop-3' if ok else 'failed'}", flush=True)
    print(f"\ncurriculum reliability: {solved}/{len(seeds)} seeds solve hop-3 "
          f"(no-curriculum baseline ~1/3). Higher => curriculum is a reliability lever.", flush=True)
    Path("runs_hop").mkdir(exist_ok=True)
    Path("runs_hop/curriculum_attn3.json").write_text(json.dumps({"solved": solved, "n": len(seeds), **out}, indent=1))


if __name__ == "__main__":
    seeds = [int(a) for a in sys.argv[1:]] or [0, 1, 2, 3]
    run(seeds)
