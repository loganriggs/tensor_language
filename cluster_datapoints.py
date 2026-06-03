#!/usr/bin/env python3
"""Cluster val datapoints by their *loss vector across model settings* and render an
interactive HTML (hover a point to see the predicted token + surrounding context).

Idea: each datapoint (seq, pos) gets a vector [CE under embed_unembed, attn1, attn2, ...].
Datapoints that improve only under a subset of components form distinct clusters — "subset
behavior". We z-score the loss columns, KMeans-cluster, and PCA-project to 2D for the scatter.

Uses the checkpoints saved by bilinear_components.py and a small raw-GPT2-token val (so the
context is human-readable; the model itself sees tokens mod 5000). Tiny models don't memorize
(train≈val), so a fresh-stream val is effectively held out.

Usage: python cluster_datapoints.py --models embed_unembed,attn1,attn2 --k 6
"""
import html
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import train_sweep as ts

HERE = Path(__file__).parent
RAW_VAL = HERE / "data" / "val_raw.pt"


def build_raw_val(n_seq, n_ctx):
    """Fresh stream of raw GPT2 tokens (NOT mod-5000) for readable context; cache to disk."""
    if RAW_VAL.exists():
        v = torch.load(RAW_VAL, weights_only=True)
        if v.shape[0] >= n_seq and v.shape[1] >= n_ctx:
            return v[:n_seq, :n_ctx]
    from datasets import load_dataset
    from transformers import GPT2Tokenizer
    print("building raw val (streaming Pile, raw GPT2 tokens)...", flush=True)
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    ds = load_dataset("stanford-crfm/DSIR-filtered-pile-50M", split="train", streaming=True)
    buf, seqs = [], []
    for ex in ds:
        buf.extend(tok.encode(ex["contents"]))
        while len(buf) >= n_ctx:
            seqs.append(buf[:n_ctx]); buf = buf[n_ctx:]
            if len(seqs) >= n_seq:
                break
        if len(seqs) >= n_seq:
            break
    v = torch.tensor(seqs, dtype=torch.long)
    torch.save(v, RAW_VAL)
    print(f"  saved {RAW_VAL} {tuple(v.shape)}", flush=True)
    return v


def load_model(variant, ckpt_dir, d, n_ctx, device):
    cfg = ts.VARIANTS[variant]
    m = ts.SweepLM(ts.VOCAB_SIZE, n_ctx, d, cfg["n_layers"], cfg["use_mlp"],
                   final_norm="rmsnorm", layer_norm="rmsnorm")
    sd = torch.load(ckpt_dir / f"{variant}.pt", weights_only=True)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()


@torch.no_grad()
def per_datapoint_ce(model, val_mod, n_ctx, device, bs=25):
    out = []
    for s in range(0, val_mod.shape[0], bs):
        b = val_mod[s:s + bs, :n_ctx].to(device)
        logits = model(b).float()
        ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                             b[:, 1:].reshape(-1), reduction="none")
        out.append(ce.reshape(b.shape[0], -1).cpu())
    return torch.cat(out, 0)              # [n_seq, n_ctx-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="embed_unembed,attn1,attn2")
    p.add_argument("--k", type=int, default=6, help="number of clusters")
    p.add_argument("--n-seq", type=int, default=400)
    p.add_argument("--n-ctx", type=int, default=256)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--sample", type=int, default=5000, help="datapoints to display (top cross-model std)")
    p.add_argument("--ckpt-dir", type=str, default="")
    args = p.parse_args()
    models = args.models.split(",")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else \
        sorted(HERE.glob("runs/*_bilinear_components/checkpoints"))[-1]
    print(f"ckpt_dir={ckpt_dir}  models={models}", flush=True)

    raw = build_raw_val(args.n_seq, args.n_ctx)        # [n_seq, n_ctx] raw GPT2
    val_mod = (raw % ts.VOCAB_SIZE)                     # what the model sees

    # per-datapoint CE for each model -> [n_seq, n_ctx-1] each
    ces = {}
    for v in models:
        ces[v] = per_datapoint_ce(load_model(v, ckpt_dir, args.d_model, args.n_ctx, device),
                                  val_mod, args.n_ctx, device)
        print(f"  {v:14} val-mean CE {ces[v].mean():.4f}", flush=True)

    n_seq, T = ces[models[0]].shape
    L = np.stack([ces[v].numpy().reshape(-1) for v in models], axis=1)   # [N, M]
    seq_idx = np.repeat(np.arange(n_seq), T)
    pos_idx = np.tile(np.arange(T), n_seq)                               # predicts token pos+1

    # pick the most "discriminating" datapoints (models disagree) to display
    std = L.std(axis=1)
    keep = np.argsort(-std)[:args.sample]
    Lk, seqk, posk = L[keep], seq_idx[keep], pos_idx[keep]

    # cluster on z-scored loss columns; project to 2D with PCA
    Z = (Lk - Lk.mean(0)) / (Lk.std(0) + 1e-6)
    km = KMeans(n_clusters=args.k, n_init=10, random_state=0).fit(Z)
    xy = PCA(n_components=2, random_state=0).fit_transform(Z)

    # decode context for hover
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    def ctx_text(seq, pos, span=12):
        start = max(0, pos + 1 - span)
        prev = tok.decode(raw[seq, start:pos + 1].tolist())
        nxt = tok.decode([int(raw[seq, pos + 1])])
        s = (prev + "⟦" + nxt + "⟧").replace("\n", "⏎")
        return html.escape(s[-160:])

    hover = []
    for i in range(len(keep)):
        losses = "  ".join(f"{m}:{Lk[i, j]:.2f}" for j, m in enumerate(models))
        hover.append(f"seq{seqk[i]} pos{posk[i]+1}<br>{ctx_text(seqk[i], posk[i])}<br>{losses}")

    # cluster mean RAW-loss profiles (interpret each cluster)
    print("\n=== cluster mean loss profiles (raw CE) ===")
    prof = []
    for c in range(args.k):
        m_mean = Lk[km.labels_ == c].mean(0)
        prof.append(m_mean)
        print(f"  cluster {c} (n={int((km.labels_==c).sum())}): " +
              "  ".join(f"{mm}={m_mean[j]:.2f}" for j, mm in enumerate(models)))

    # ---- HTML: scatter (PCA, colored clusters, hover) + cluster-profile lines ----
    fig = make_subplots(1, 2, column_widths=[0.62, 0.38],
                        subplot_titles=("datapoints (PCA of loss-vector), colored by cluster — hover for context",
                                        "cluster mean loss profile across models"))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf"]
    for c in range(args.k):
        msk = km.labels_ == c
        fig.add_trace(go.Scattergl(x=xy[msk, 0], y=xy[msk, 1], mode="markers",
                                   marker=dict(size=4, color=palette[c % len(palette)], opacity=0.6),
                                   name=f"cluster {c} (n={int(msk.sum())})",
                                   text=[hover[i] for i in np.where(msk)[0]], hoverinfo="text"), 1, 1)
    for c in range(args.k):
        fig.add_trace(go.Scatter(x=models, y=prof[c], mode="lines+markers",
                                 line=dict(color=palette[c % len(palette)]),
                                 name=f"c{c}", showlegend=False), 1, 2)
    fig.update_yaxes(title_text="mean CE", row=1, col=2)
    fig.update_layout(title=f"Datapoint clustering by loss-vector across {', '.join(models)} "
                            f"(top {args.sample} most-discriminating of {L.shape[0]})",
                      height=640, template="plotly_white")

    ts_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out = HERE / "runs" / f"{ts_str}_cluster"; out.mkdir(parents=True, exist_ok=True)
    html_path = out / f"clusters_{len(models)}models.html"
    fig.write_html(str(html_path))
    # also drop a copy at repo root for easy opening
    root_copy = HERE / f"clusters_{len(models)}models.html"
    root_copy.write_text(html_path.read_text())
    print(f"\nHTML: {html_path}\n  copy: {root_copy}")


if __name__ == "__main__":
    main()
