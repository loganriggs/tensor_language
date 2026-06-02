#!/usr/bin/env python3
"""Deep (8-layer) loss-curve comparison of the top norm configs, on identical data.

Trains several `xf` (bilinear attn + bilinear MLP) models of a fixed depth on the *same*
pre-collected stream of Pile batches, logs the **training** loss every step, and saves a
log-scale plot. No val set: streaming Pile batches are fresh (seen once), and every model
sees the identical batch sequence, so the curves are directly comparable relative to each
other (per the experiment design — relative comparison, not absolute smoothness).

Default configs (label, final_norm, per-layer norm, is-polynomial):
  - layernorm + rmsnorm-layers   : best overall, NOT polynomial (both norms per-sample)
  - none      + rmsnorm-layers   : best "none", NOT polynomial (per-layer rmsnorm is per-sample)
  - none      + static-rms-layers: FULLY polynomial / foldable (TN-pure)

Usage:
  python train_depth_curves.py --layers 8 --steps 5000
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

import train_sweep as ts
from train_sweep import DSIRPileStreaming
from torch.utils.data import DataLoader

CONFIGS = [
    dict(label="non-foldable best (none + rmsnorm layers)",
         final_norm="none", layer_norm="rmsnorm", spectral=False, rezero=None, poly=False),
    dict(label="foldable WINNER (none + static-rms layers + spectral + rezero0.1)",
         final_norm="none", layer_norm="static-rms", spectral=True, rezero=0.1, poly=True),
    dict(label="foldable baseline (none + static-rms layers) — diverges",
         final_norm="none", layer_norm="static-rms", spectral=False, rezero=None, poly=True),
]


def collect_batches(n_batches, n_ctx, batch_size):
    """Pre-collect a fixed list of batches so every model trains on identical data."""
    dl = DataLoader(DSIRPileStreaming(n_ctx), batch_size=batch_size)
    out, t0 = [], time.time()
    for b in dl:
        out.append(b["input_ids"])
        if len(out) >= n_batches:
            break
    print(f"collected {len(out)} batches in {time.time() - t0:.0f}s", flush=True)
    return out


def train_curve(cfg, batches, *, d_model, n_ctx, n_layers, lr, device, use_compile):
    torch.manual_seed(42)
    model = ts.SweepLM(ts.VOCAB_SIZE, n_ctx, d_model, n_layers, use_mlp=True,
                       final_norm=cfg["final_norm"], layer_norm=cfg["layer_norm"],
                       spectral=cfg["spectral"], rezero_init=cfg["rezero"]).to(device)
    if use_compile and not cfg["spectral"]:   # spectral_norm + compile -> NaN
        model = torch.compile(model)
    opt = ts.create_optimizer(model, lr=lr)
    sched = ts.create_scheduler(opt, warmup_steps=min(100, len(batches) // 5), max_steps=len(batches))
    losses, t0 = [], time.time()
    model.train()
    for ids in batches:
        ids = ids.to(device)
        opt.zero_grad()
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            loss = ts.compute_loss(model(ids), ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())     # logged every batch
    return losses, round(time.time() - t0, 1)


def moving_avg(xs, w):
    if len(xs) < w:
        return xs
    import numpy as np
    return np.convolve(xs, np.ones(w) / w, mode="valid").tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-ctx", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--smooth", type=int, default=50, help="moving-avg window for the plot")
    p.add_argument("--no-compile", action="store_true")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ts_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = Path(__file__).parent / "runs" / f"{ts_str}_curves_xf{args.layers}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== xf{args.layers} loss-curve comparison ===")
    print(f"device={device} layers={args.layers} steps={args.steps} d={args.d_model} "
          f"ctx={args.n_ctx} bs={args.batch_size} lr={args.lr}")
    print(f"out_dir={out_dir}\n")

    batches = collect_batches(args.steps, args.n_ctx, args.batch_size)

    results = {}
    for cfg in CONFIGS:
        losses, secs = train_curve(cfg, batches,
                                   d_model=args.d_model, n_ctx=args.n_ctx, n_layers=args.layers,
                                   lr=args.lr, device=device, use_compile=not args.no_compile)
        tail = sum(losses[-100:]) / len(losses[-100:])
        results[cfg["label"]] = {"final_norm": cfg["final_norm"], "layer_norm": cfg["layer_norm"],
                                 "spectral": cfg["spectral"], "rezero": cfg["rezero"],
                                 "polynomial": cfg["poly"], "losses": losses,
                                 "final_loss_avg100": round(tail, 4), "secs": secs}
        print(f"  {cfg['label']:60} final-100-avg={tail:.4f}  ({secs}s)", flush=True)

    with open(out_dir / "curves.json", "w") as f:
        json.dump({"config": vars(args), "results": results}, f)

    # --- 2-panel plot: (left) all configs log-scale (shows any divergence),
    #                   (right) only the stable (finite-tail) configs, zoomed. ---
    def smoothed(r):
        sm = moving_avg(r["losses"], args.smooth)
        return range(len(r["losses"]) - len(sm), len(r["losses"])), sm

    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5.5))
    for label, r in results.items():
        xs, sm = smoothed(r)
        tag = " [polynomial]" if r["polynomial"] else ""
        ax_all.plot(xs, sm, lw=1.6, label=f"{label}{tag} →{r['final_loss_avg100']:.3f}")
    ax_all.set_yscale("log")
    ax_all.set_title(f"xf{args.layers}: all configs (log scale)")
    ax_all.set_xlabel("training step (batch)"); ax_all.set_ylabel("train CE (log)")
    ax_all.legend(fontsize=8); ax_all.grid(True, which="both", alpha=0.3)

    stable = {l: r for l, r in results.items() if r["final_loss_avg100"] < 50}
    for label, r in stable.items():
        xs, sm = smoothed(r)
        ax_zoom.plot(xs, sm, lw=1.8, label=f"{label} →{r['final_loss_avg100']:.3f}")
    if stable:
        lo = min(r["final_loss_avg100"] for r in stable.values())
        ax_zoom.set_ylim(lo - 0.2, lo + 1.2)
    ax_zoom.set_title(f"xf{args.layers}: stable configs (zoomed)")
    ax_zoom.set_xlabel("training step (batch)"); ax_zoom.set_ylabel("train CE (linear)")
    ax_zoom.legend(fontsize=9); ax_zoom.grid(True, alpha=0.3)

    plt.tight_layout()
    png = out_dir / f"loss_curves_xf{args.layers}.png"
    plt.savefig(png, dpi=130)
    print(f"\nplot saved: {png}")


if __name__ == "__main__":
    main()
