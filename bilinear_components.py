#!/usr/bin/env python3
"""Bilinear component ladder: train embed_unembed / attn1 / attn2 / xf1 / xf2 to convergence
(RMSNorm per-layer + final) and attribute *which datapoints* each added component improves.

- Architecture: the bilinear model from train_sweep.py (bilinear attention (+ bilinear MLP)),
  per-layer pre-norm = RMSNorm, final norm = RMSNorm.
- Train: random windows over the 150M-token Pile corpus (data/pile_tokens.pt, no overfitting).
- Eval: per-(seq,pos) cross-entropy on the fixed cached Pile val (dsir_pile_val_ctx512.pt).
- Attribution: for each component step (embed->attn1, attn1->attn2, attn1->xf1, attn2->xf2),
  delta = CE_fewer - CE_more per datapoint; the datapoints with the largest positive delta are
  the ones that component "explains". Full per-datapoint CE arrays are saved for later analysis
  (e.g. grouping / similarity of what each component learns).

Usage: python bilinear_components.py --steps 25000
"""
import json
import time
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import train_sweep as ts

HERE = Path(__file__).parent
LADDER = ["embed_unembed", "attn1", "attn2", "xf1", "xf2"]
# component-addition steps: (fewer, more, what-was-added)
STEPS = [("embed_unembed", "attn1", "+1st attention"),
         ("attn1", "attn2", "+2nd attention"),
         ("attn1", "xf1", "+MLP (1 layer)"),
         ("attn2", "xf2", "+MLP (2 layer)")]


def get_batch(corpus, n_ctx, bs, device):
    ix = torch.randint(len(corpus) - n_ctx - 1, (bs,))
    x = torch.stack([corpus[i:i + n_ctx + 1] for i in ix]).long().to(device)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


@torch.no_grad()
def per_datapoint_ce(model, val, n_ctx, device, bs=25):
    """[n_seq, n_ctx-1] next-token CE on the fixed val set."""
    model.eval()
    out = []
    for s in range(0, val.shape[0], bs):
        b = val[s:s + bs, :n_ctx].long().to(device)
        logits = model(b).float()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             b[:, 1:].reshape(-1), reduction="none")
        out.append(ce.reshape(b.shape[0], -1).cpu())
    model.train()
    return torch.cat(out, 0)


def train_variant(variant, corpus, val, *, d_model, n_ctx, steps, lr, batch_size, device, use_compile):
    cfg = ts.VARIANTS[variant]
    torch.manual_seed(42)
    model = ts.SweepLM(ts.VOCAB_SIZE, n_ctx, d_model, cfg["n_layers"], cfg["use_mlp"],
                       final_norm="rmsnorm", layer_norm="rmsnorm").to(device)
    if use_compile:
        model = torch.compile(model)
    opt = ts.create_optimizer(model, lr=lr)
    sched = ts.create_scheduler(opt, warmup_steps=min(200, steps // 10), max_steps=steps)
    curve = {"step": [], "loss": []}
    t0 = time.time()
    model.train()
    for it in range(steps):
        xb, yb = get_batch(corpus, n_ctx, batch_size, device)
        with torch.amp.autocast(device, dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model(xb)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if it % 100 == 0:
            curve["step"].append(it); curve["loss"].append(loss.item())
    net = getattr(model, "_orig_mod", model)
    ce = per_datapoint_ce(model, val, n_ctx, device)
    secs = round(time.time() - t0, 1)
    print(f"  [{variant:14}] final train {curve['loss'][-1]:.4f}  val-mean {ce.mean():.4f}  ({secs}s)", flush=True)
    return curve, ce, net.state_dict(), secs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25000)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-ctx", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--top-k", type=int, default=25)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    corpus = torch.load(HERE / "data" / "pile_tokens.pt", weights_only=True)
    val = torch.load(ts.PILE_VAL, weights_only=True)
    ts_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "runs" / f"{ts_str}_bilinear_components"; out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(exist_ok=True)
    print(f"device={device} steps={args.steps} d={args.d_model} ctx={args.n_ctx} "
          f"rmsnorm (per-layer+final)  out={out}\n", flush=True)

    curves, ces, secs = {}, {}, {}
    for v in LADDER:
        curve, ce, sd, sc = train_variant(v, corpus, val, d_model=args.d_model, n_ctx=args.n_ctx,
                                          steps=args.steps, lr=args.lr, batch_size=args.batch_size,
                                          device=device, use_compile=not args.no_compile)
        curves[v] = curve; ces[v] = ce; secs[v] = sc
        torch.save(sd, out / "checkpoints" / f"{v}.pt")

    # ---- training-loss curves (log y) ----
    def smooth(xs, w=10):
        return np.convolve(xs, np.ones(w) / w, mode="valid") if len(xs) >= w else np.array(xs)
    plt.figure(figsize=(9, 5.5))
    for v in LADDER:
        sm = smooth(curves[v]["loss"]); xs = curves[v]["step"][len(curves[v]["step"]) - len(sm):]
        plt.plot(xs, sm, lw=1.8, label=f"{v}  val-mean={ces[v].mean():.3f}")
    plt.yscale("log"); plt.xlabel("step"); plt.ylabel("train CE (log, smoothed)")
    plt.title(f"Bilinear component ladder (RMSNorm) — train loss to convergence")
    plt.legend(fontsize=9); plt.grid(True, which="both", alpha=0.3); plt.tight_layout()
    plt.savefig(out / "train_curves.png", dpi=140); plt.close()

    # ---- monotonicity (val-mean CE should fall as components are added) ----
    print("\n=== val-mean CE (lower = better) ===")
    for v in LADDER:
        print(f"  {v:14} {ces[v].mean():.4f}")

    # ---- per-datapoint attribution ----
    attribution = {}
    print("\n=== component attribution: datapoints each added component improves most ===")
    for fewer, more, what in STEPS:
        delta = (ces[fewer] - ces[more])          # >0 : 'more' lowered the loss there
        T = delta.shape[1]
        flat = delta.flatten()
        vals, idx = flat.topk(args.top_k)
        top = [{"seq": int(i // T), "pos": int(i % T) + 1, "delta": round(float(v), 4),
                "ce_fewer": round(float(ces[fewer].flatten()[i]), 4),
                "ce_more": round(float(ces[more].flatten()[i]), 4)} for i, v in zip(idx, vals)]
        frac_improved = float((delta > 0).float().mean())
        attribution[f"{fewer}->{more}"] = {"what": what, "mean_delta": round(float(delta.mean()), 4),
                                           "frac_improved": round(frac_improved, 3), "top": top}
        print(f"\n  {what}  ({fewer} -> {more}):  mean Δ={delta.mean():+.4f}  "
              f"improved {frac_improved*100:.0f}% of datapoints")
        for t in top[:8]:
            print(f"     seq{t['seq']:3} pos{t['pos']:3}  Δ={t['delta']:+.3f}  "
                  f"({t['ce_fewer']:.2f} -> {t['ce_more']:.2f})")

    # ---- save everything for later similarity/grouping analysis ----
    torch.save({"ces": ces, "val": val[:, :args.n_ctx], "ladder": LADDER, "steps": STEPS},
               out / "per_datapoint_ce.pt")
    with open(out / "attribution.json", "w") as f:
        json.dump({"val_mean": {v: float(ces[v].mean()) for v in LADDER},
                   "secs": secs, "attribution": attribution}, f, indent=2)
    print(f"\nsaved: {out}/  (train_curves.png, per_datapoint_ce.pt, attribution.json, checkpoints/)")


if __name__ == "__main__":
    main()
