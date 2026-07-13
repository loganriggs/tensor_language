"""Train the depth ladder on natural text (TinyStories, BPE-1024) — PLAN.md step 1.

Bilinear attention-only models (RMSNorm on, lerp residual — the recipe from the hop
work), full next-token CE. All depths at a given seed index see IDENTICAL data batches
(data generator seeded independently of init), so per-token CE differences across the
ladder are attributable to depth, not data order.

Per run, saved under runs_lm/<spec>-seed<k>/:
    config.json      architecture + recipe
    history.jsonl    step, train CE, quick-val CE (every EVAL_EVERY steps)
    ckpt/step*.pt    checkpoints through training (for dynamics analyses)
    model.pt         final weights

Usage: python lm_train.py attn2 0 1 2        (one spec, seeds -> parallelizable)
       python lm_train.py --heads 8 attn2 0  (head-count ladder variant, tagged -h8)
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from deep_model import DeepModel, SPECS
from text_data import VOCAB, N_CTX, RUNS, tokens, get_batch

D_MODEL = 128
N_HEAD = 4
STEPS = 40_000
BATCH = 64
LR = 1e-3
NORM = "rms"
EVAL_EVERY = 1_000
CKPT_EVERY = 2_500
QUICK_VAL_WINDOWS = 200          # quick-val = first 200 frozen windows (~51k tokens)


def quick_val_ce(model, val, device="cuda"):
    model.eval()
    ces = []
    with torch.no_grad():
        for i in range(0, QUICK_VAL_WINDOWS, 50):
            buf = np.stack([val[w * N_CTX:w * N_CTX + N_CTX + 1]
                            for w in range(i, i + 50)]).astype(np.int64)
            b = torch.from_numpy(buf).to(device)
            logits = model(b[:, :-1])
            ces.append(F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1)).item())
    model.train()
    return sum(ces) / len(ces)


def train_one(spec_name, seed, n_head=N_HEAD, device="cuda", dense=False, attention="bilinear",
              residual="lerp", steps=STEPS, d_model=D_MODEL, copymix=0.0, mixuntil=0,
              randperiod=False):
    tag = (f"{spec_name}{'' if n_head == N_HEAD else f'-h{n_head}'}"
           f"{'' if d_model == D_MODEL else f'-d{d_model}'}"
           f"{'' if attention == 'bilinear' else '-' + attention}"
           f"{'' if residual == 'lerp' else '-' + residual}"
           f"{'' if steps == STEPS else f'-s{steps // 1000}k'}"
           f"{'' if not copymix else f'-mix{int(copymix * 100)}'}"
           f"{'-rp' if randperiod else ''}"
           f"{'' if not mixuntil else f'u{mixuntil // 1000}k'}{'-dense' if dense else ''}-seed{seed}")
    out = RUNS / tag
    if (out / "model.pt").exists():
        print(f"skip {tag} (exists)", flush=True)
        return
    (out / "ckpt").mkdir(parents=True, exist_ok=True)

    train, val = tokens("train"), tokens("val")
    torch.manual_seed(seed)
    data_gen = torch.Generator().manual_seed(10_000 + seed)   # same data across depths
    model = DeepModel(VOCAB, d_model, n_head, SPECS[spec_name], N_CTX, norm=NORM,
                      attention=attention, residual=residual, mlp_residual="add").to(device)
    (out / "config.json").write_text(json.dumps({
        "spec": SPECS[spec_name], "d_model": d_model, "n_head": n_head, "vocab": VOCAB,
        "n_ctx": N_CTX, "norm": NORM, "residual": residual, "attention": attention,
        "steps": steps, "batch": BATCH, "lr": LR, "seed": seed, "n_params": model.n_params}))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 200, 0.5 * (1 + math.cos(math.pi * s / steps))))

    hist = open(out / "history.jsonl", "a")
    model.train()
    t0 = time.time()
    for step in range(steps):
        b = get_batch(train, BATCH, data_gen, device)
        if copymix and (not mixuntil or step < mixuntil):
            # replace a fraction of rows with [u;u] copy-burst sequences (iid uniform):
            # pure in-context structure, the basin-escape lever for induction formation
            rows = torch.rand(BATCH, generator=data_gen) < copymix
            n = int(rows.sum())
            if n:
                if randperiod:
                    # FIX (2026-07-09): a CONSTANT period makes content-matching and a fixed
                    # positional offset EXACTLY equivalent, so [u;u] with period N_CTX//2 teaches a
                    # positional copier (mix10: P(copy) 0.90 at period 128, chance at 150/100/64).
                    # Randomising the period per row makes the offset uninformative.
                    # TILE the block: b = [w w w ...] with |w| = P. Every position q >= P then has
                    # an earlier occurrence at q-P, so the WHOLE suffix is content-predictable.
                    # ([w ; u] truncated would leave positions q > 2P-1 as fresh unseen tokens —
                    # pure noise — which is why the first attempt at this learned nothing.)
                    P = torch.randint(N_CTX // 6, N_CTX // 2 + 1, (n,), generator=data_gen)
                    for k, r in enumerate(torch.where(rows)[0].tolist()):
                        p = int(P[k])
                        w = torch.randint(VOCAB, (p,), generator=data_gen)
                        reps = (N_CTX + 1 + p - 1) // p
                        b[r] = w.repeat(reps)[:N_CTX + 1].to(device)
                else:
                    # original path; RNG draw unchanged so existing runs stay reproducible
                    u = torch.randint(VOCAB, (n, N_CTX // 2 + 1), generator=data_gen)
                    b[rows] = torch.cat([u[:, :N_CTX // 2], u], 1)[:, :N_CTX + 1].to(device)
        logits = model(b[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if not torch.isfinite(loss):
            print(f"{tag} DIVERGED at step {step}", flush=True)
            hist.write(json.dumps({"step": step, "diverged": True}) + "\n")
            break
        if step % EVAL_EVERY == 0:
            vce = quick_val_ce(model, val, device)
            hist.write(json.dumps({"step": step, "train_ce": round(loss.item(), 4),
                                   "val_ce": round(vce, 4)}) + "\n")
            hist.flush()
            print(f"{tag} step {step} train {loss.item():.3f} val {vce:.3f} "
                  f"({(time.time()-t0):.0f}s)", flush=True)
        if step and (step % CKPT_EVERY == 0 or (dense and step <= 5000 and step % 250 == 0)):
            torch.save(model.state_dict(), out / "ckpt" / f"step{step}.pt")
    vce = quick_val_ce(model, val, device)
    hist.write(json.dumps({"step": steps, "val_ce": round(vce, 4)}) + "\n")
    hist.close()
    torch.save(model.state_dict(), out / "model.pt")
    print(f"== {tag} final val CE {vce:.4f} ({(time.time()-t0)/60:.0f} min)", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    n_head = N_HEAD
    attention = "bilinear"
    residual = "lerp"
    dense = "--dense" in args
    if dense:
        args.remove("--dense")
    if "--softmax" in args:
        attention = "softmax"; args.remove("--softmax")
    if "--add" in args:
        residual = "add"; args.remove("--add")
    if "--heads" in args:
        i = args.index("--heads"); n_head = int(args[i + 1]); del args[i:i + 2]
    steps = STEPS
    if "--steps" in args:
        i = args.index("--steps"); steps = int(args[i + 1]); del args[i:i + 2]
    d_model = D_MODEL
    if "--dmodel" in args:
        i = args.index("--dmodel"); d_model = int(args[i + 1]); del args[i:i + 2]
    copymix = 0.0
    if "--copymix" in args:
        i = args.index("--copymix"); copymix = float(args[i + 1]); del args[i:i + 2]
    randperiod = "--randperiod" in args
    if randperiod:
        args.remove("--randperiod")
    mixuntil = 0
    if "--mixuntil" in args:
        i = args.index("--mixuntil"); mixuntil = int(args[i + 1]); del args[i:i + 2]
    specs = [a for a in args if a in SPECS]
    seeds = [int(a) for a in args if a not in SPECS] or [0]
    for seed in seeds:
        for s in specs:
            train_one(s, seed, n_head, dense=dense, attention=attention, residual=residual,
                      steps=steps, d_model=d_model, copymix=copymix, mixuntil=mixuntil,
                      randperiod=randperiod)
