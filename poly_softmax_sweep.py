#!/usr/bin/env python3
"""Overnight sweep: does a polynomial-leaning transformer (taylor attention + spherical
SoftmaxNorm) match the standard softmax+RMSNorm baseline as we scale layers and steps?

Trained on a large pre-tokenized Pile corpus (data/pile_tokens.pt, vocab 5000, no overfitting
so "more steps" is meaningful). Four model recipes:
  baseline  = softmax  + rmsnorm     (standard)
  poly      = taylor   + spherical   (best polynomial-leaning of Exp 1 & 2)
  attn-only = taylor   + rmsnorm     (isolate the attention change)
  norm-only = softmax  + spherical   (isolate the norm change)

Robust for unattended runs: NaN-guarded, results saved + plots regenerated after EVERY run,
and a wall-clock budget stops launching new runs near the limit. Essential runs first.

Usage: python poly_softmax_sweep.py [--budget-hours 8.5] [--no-compile]
"""
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from poly_softmax_gpt import train_run, load_data, HERE

RECIPES = {
    "baseline":  ("softmax", "rmsnorm"),
    "poly":      ("taylor",  "spherical"),
    "attn-only": ("taylor",  "rmsnorm"),
    "norm-only": ("softmax", "spherical"),
}


def build_runs():
    """Ordered list of run specs. Essential science first; long/extra runs last so a budget
    cutoff still leaves a complete core result. Each: (group, recipe, n_layer, iters, seed)."""
    R = []
    # Phase 1 — 2x2 component ablation at two depths (isolate each change)
    for L in (6, 12):
        for rec in ("baseline", "attn-only", "norm-only", "poly"):
            R.append((f"ablation_L{L}", rec, L, 12000, 1))
    # Phase 2 — depth scaling, baseline vs poly (does the gap grow/shrink with depth?)
    for L in (4, 8, 16):
        for rec in ("baseline", "poly"):
            R.append(("depth_scaling", rec, L, 12000, 1))
    # Phase 3 — seeds at L8 (are the ~0.02 gaps real or noise?)
    for seed in (2, 3):
        for rec in ("baseline", "poly"):
            R.append(("seeds_L8", rec, 8, 12000, seed))
    # Phase 4 — long training (does the gap hold with many more steps?) — expensive
    for rec in ("baseline", "poly"):
        R.append(("long_L12", rec, 12, 30000, 1))
    # ---- extras (run only if budget remains) ----
    # extra depth points for a smoother scaling curve
    for L in (10, 14):
        for rec in ("baseline", "poly"):
            R.append(("depth_scaling", rec, L, 12000, 1))
    # a 3rd seed
    for rec in ("baseline", "poly"):
        R.append(("seeds_L8", rec, 8, 12000, 4))
    # a second long comparison at a different depth
    for rec in ("baseline", "poly"):
        R.append(("long_L8", rec, 8, 30000, 1))
    return R


def plot_group(results, group, out_png):
    rows = {lbl: h for lbl, h in results.items() if h["group"] == group}
    if not rows:
        return
    plt.figure(figsize=(9, 5.5))
    for lbl, h in sorted(rows.items()):
        d = " DIVERGED" if h.get("diverged") else ""
        line, = plt.plot(h["step"], h["val"], lw=1.8, label=f"{lbl}  best={h['best_val']}{d}")
        plt.plot(h["step"], h["train"], ls="--", alpha=0.4, color=line.get_color())
    plt.xlabel("step"); plt.ylabel("CE (solid=val, dashed=train)")
    plt.title(f"poly-softmax sweep on Pile — {group}")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget-hours", type=float, default=8.5)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--n-embd", type=int, default=384)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_d, val_d, vocab = load_data("pile")
    base = dict(vocab_size=vocab, n_head=6, n_embd=args.n_embd, block_size=args.block_size,
                batch_size=args.batch_size, dropout=0.0)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "runs" / f"{ts}_polysweep"; out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} vocab={vocab} tokens={len(train_d)/1e6:.0f}M train  out={out}", flush=True)

    runs = build_runs()
    budget = args.budget_hours * 3600
    start = time.time()
    results, summary = {}, []
    for i, (group, recipe, n_layer, iters, seed) in enumerate(runs):
        elapsed = time.time() - start
        if elapsed > budget:
            print(f"\n[budget] {elapsed/3600:.1f}h elapsed >= {args.budget_hours}h — skipping remaining "
                  f"{len(runs)-i} runs", flush=True)
            break
        attn, norm = RECIPES[recipe]
        label = f"{recipe} L{n_layer}" + (f" s{seed}" if seed != 1 else "") + \
                (f" {iters//1000}k" if iters != 12000 else "")
        print(f"\n=== run {i+1}/{len(runs)}  [{group}] {label}  ({attn}+{norm}, {n_layer}L, {iters} it, "
              f"seed {seed})  elapsed {elapsed/3600:.1f}h ===", flush=True)
        cfg = {**base, "n_layer": n_layer}
        try:
            h = train_run(label, attn, norm, cfg, max_iters=iters, eval_interval=500, lr=args.lr,
                          device=device, seed=seed, train_d=train_d, val_d=val_d,
                          use_compile=not args.no_compile)
        except Exception as e:
            print(f"  [{label}] ERROR: {type(e).__name__}: {e}", flush=True)
            continue
        h.update(group=group, recipe=recipe, n_layer=n_layer, iters=iters, seed=seed,
                 attn=attn, norm=norm)
        results[label] = h
        summary.append(dict(label=label, group=group, recipe=recipe, n_layer=n_layer, iters=iters,
                            seed=seed, best_val=h["best_val"], final_val=h["final_val"],
                            diverged=h.get("diverged", False), secs=h["secs"]))
        # incremental save + replot after every run
        with open(out / "results.json", "w") as f:
            json.dump({"base": base, "lr": args.lr, "summary": summary, "results": results}, f)
        for g in sorted(set(r["group"] for r in results.values())):
            plot_group(results, g, out / f"{g}.png")

    print(f"\n=== SWEEP DONE ({(time.time()-start)/3600:.1f}h) — {len(summary)} runs ===")
    for s in summary:
        d = " DIVERGED" if s["diverged"] else ""
        print(f"  {s['group']:16} {s['label']:22} best={s['best_val']:.4f} final={s['final_val']:.4f}{d}")
    print(f"\nresults: {out}")


if __name__ == "__main__":
    main()
