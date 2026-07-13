"""Cluster gated datapoints by FORMATION TRAJECTORY — clustering v2 (PLAN.md step 4).

Premise (Singh et al. repurposed): datapoints computed by the same circuit are learned
together — their CE-vs-training-step curves share timing and shape, and this replicates
across seeds. Feature vector per token = its CE at each saved checkpoint, concatenated
over reference runs (different seeds), each trajectory normalized to [0,1] per token
(shape, not level). K-means over trajectories; report per-cluster formation curves,
induction-pattern fraction, and decoded examples.

Advantages over knock-out fingerprints (circuit_fingerprints.py): no head-identity
alignment problem across seeds, no head-dominance artifact, and it is dynamics-native —
clusters ARE candidate subcircuit formation events.

Usage: python trajectory_clusters.py gated_depth3 attn3-seed0,attn3-seed1 [--sample 6000] [--k 6]
"""

import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans

from lm_eval import load_model
from text_data import N_CTX, RUNS, load_tokenizer, val_windows

DEVICE = "cuda"
BATCH = 128
CTX_SHOW = 40


def token_trajectories(run: Path, wins, rows, pos, b):
    ckpts = sorted(run.glob("ckpt/step*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    steps, traj = [], []
    with torch.no_grad():
        for ck in ckpts + [None]:
            model = load_model(run, ck.stem if ck else None, DEVICE)
            steps.append(int(re.findall(r"\d+", ck.name)[0]) if ck else
                         json.loads((run / "config.json").read_text())["steps"])
            ces = []
            for i in range(0, len(b), BATCH):
                logits = model(b[i:i + BATCH, :-1])
                ces.append(F.cross_entropy(logits.transpose(1, 2), b[i:i + BATCH, 1:],
                                           reduction="none"))
            traj.append(torch.cat(ces)[rows, pos].cpu().numpy())
            print(f"  {run.name} step {steps[-1]}", flush=True)
    return np.array(steps), np.stack(traj, 1)          # (n_tokens, n_ckpts)


def main(gateset="gated_depth3", runs="attn3-seed0,attn3-seed1", sample=6000, k=6, seed=0):
    data, _ = val_windows()
    gated = np.load(RUNS / f"{gateset}.npy")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(gated, min(sample, len(gated)), replace=False))
    wins, pos = np.divmod(idx, N_CTX)
    uw = np.unique(wins)
    w2row = {w: i for i, w in enumerate(uw)}
    rows = np.array([w2row[w] for w in wins])
    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX + 1] for w in uw]).astype(np.int64)
    b = torch.from_numpy(buf).to(DEVICE)
    print(f"{gateset}: {len(idx)} tokens in {len(uw)} windows", flush=True)

    all_steps, feats, raw = [], [], []
    for r in runs.split(","):
        steps, tr = token_trajectories(RUNS / r, wins, rows, pos, b)
        all_steps.append(steps); raw.append(tr)
        lo, hi = tr.min(1, keepdims=True), tr.max(1, keepdims=True)
        feats.append((tr - lo) / (hi - lo + 1e-6))       # shape-normalized per token
    X = np.concatenate(feats, 1)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    lab = km.labels_
    np.savez(RUNS / f"traj_clusters_{gateset}.npz", idx=idx, labels=lab,
             steps=np.array(all_steps[0]), traj=raw[0], traj2=raw[1] if len(raw) > 1 else raw[0])

    tok = load_tokenizer()
    is_ind = np.load(RUNS / "is_induction.npy")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    lines = [f"# Trajectory clusters: {gateset} over {runs}\n",
             f"{len(idx)} tokens, k={k}; features = shape-normalized CE trajectories.\n"]
    for c in range(k):
        m = lab == c
        mean_tr = raw[0][m].mean(0)
        half = all_steps[0][np.argmax(mean_tr <= (mean_tr[0] + mean_tr[-1]) / 2)]
        ax.plot(all_steps[0], mean_tr, label=f"C{c} (n={m.sum()}, ind {is_ind[idx[m]].mean():.0%})")
        lines.append(f"## Cluster {c} — n={m.sum()} ({m.mean():.0%}), "
                     f"induction-pattern {is_ind[idx[m]].mean():.0%}, "
                     f"half-formation ≈ step {half}, final CE {mean_tr[-1]:.2f}\n")
        for i in rng.choice(np.where(m)[0], min(8, m.sum()), replace=False):
            tpos = int(idx[i]) + int(idx[i]) // N_CTX * 0 + 1   # flat stream position of target
            tpos = (int(idx[i]) // N_CTX) * N_CTX + int(idx[i]) % N_CTX + 1
            ctx = tok.decode(list(data[max(0, tpos - CTX_SHOW):tpos]))
            lines.append(f"- `...{ctx}` ⟶ **`{tok.decode([int(data[tpos])])}`**")
        lines.append("")
    ax.set(xlabel="step", ylabel=f"mean CE of cluster ({runs.split(',')[0]})",
           title=f"formation trajectories by cluster — {gateset}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"figures/traj_clusters_{gateset}.png", dpi=150)
    Path(f"trajectory_report_{gateset}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote trajectory_report_{gateset}.md + figures/traj_clusters_{gateset}.png", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if "--sample" in a:
        i = a.index("--sample"); kw["sample"] = int(a[i + 1]); del a[i:i + 2]
    if "--k" in a:
        i = a.index("--k"); kw["k"] = int(a[i + 1]); del a[i:i + 2]
    main(*(a or ["gated_depth3", "attn3-seed0,attn3-seed1"]), **kw)
