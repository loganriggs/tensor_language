"""Cluster depth-gated datapoints by causal circuit — PLAN.md step 4.

For a sample of gated tokens, the FINGERPRINT of a token is the vector of per-head
knock-out effects: ΔCE(token) when each (layer, head) of the reference model is zeroed
(head slice zeroed pre-W_O, as in induction_dynamics.ce_split). Tokens computed by the
same algorithm should share load-bearing heads → cluster fingerprints (k-means over
L2-normalized ΔCE vectors), then decode examples per cluster for naming.

Paper §3.1 caveat: knock-outs understate redundant heads. Here that is acceptable —
fingerprints are for GROUPING datapoints (same-vs-different circuit), not for claiming a
head is unnecessary; bilinear heads also showed no redundancy in the mold study.

Outputs: runs_lm/fingerprints_<gateset>_<tag>.npz  (indices, fingerprints, cluster ids)
         fingerprint_report_<gateset>.md           (cluster sizes, decoded examples)

Usage: python circuit_fingerprints.py gated_depth3 attn3-seed0 [--sample 2000] [--k 8]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from sklearn.cluster import KMeans

from lm_eval import load_model
from text_data import N_CTX, RUNS, load_tokenizer, val_windows

DEVICE = "cuda"
BATCH = 128
CTX_SHOW = 40


def forward_with_knockout(model, x, knockout=None):
    """knockout = (layer_idx, head_idx) or None."""
    h = model.embed(x)
    for li, layer in enumerate(model.layers):
        if knockout is None or knockout[0] != li:
            h = layer(h)
        else:
            v = rearrange(layer.v(layer.norm(h)), "b t (n d) -> b t n d", n=layer.n_head)
            z = einsum(layer.pattern(h), v, "b n q k, b k n d -> b q n d")
            z[:, :, knockout[1], :] = 0
            z = rearrange(z, "b q n d -> b q (n d)")
            h = torch.lerp(h, layer.o(z), layer.scale) if layer.residual == "lerp" else h + layer.o(z)
    return model.head(h)


def main(gateset="gated_depth3", tag="attn3-seed0", sample=2000, k=8, seed=0):
    run = RUNS / tag
    model = load_model(run, None, DEVICE)
    heads = [(li, h) for li, layer in enumerate(model.layers)
             if hasattr(layer, "n_head") for h in range(layer.n_head)]
    data, n_win = val_windows()
    gated = np.load(RUNS / f"{gateset}.npy")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(gated, min(sample, len(gated)), replace=False))
    wins, pos = np.divmod(idx, N_CTX)
    uw = np.unique(wins)
    w2row = {w: i for i, w in enumerate(uw)}
    rows = np.array([w2row[w] for w in wins])
    print(f"{gateset} on {tag}: {len(idx)} tokens in {len(uw)} windows; "
          f"{len(heads)} heads -> {len(heads)+1} passes", flush=True)

    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX + 1] for w in uw]).astype(np.int64)
    b = torch.from_numpy(buf).to(DEVICE)

    def ce_at_tokens(knockout):
        ces = []
        with torch.no_grad():
            for i in range(0, len(b), BATCH):
                logits = forward_with_knockout(model, b[i:i + BATCH, :-1], knockout)
                ce = F.cross_entropy(logits.transpose(1, 2), b[i:i + BATCH, 1:], reduction="none")
                ces.append(ce)
        return torch.cat(ces)[rows, pos].cpu().numpy()          # (n_tokens,)

    base = ce_at_tokens(None)
    fp = np.stack([ce_at_tokens(ko) - base for ko in heads], 1)  # (n_tokens, n_heads)
    print(f"mean |ΔCE| per head: {np.abs(fp).mean(0).round(2)}", flush=True)

    norm = fp / (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(norm)
    lab = km.labels_
    np.savez(str(RUNS) + f"/fingerprints_{gateset}_{tag}.npz",
             idx=idx, fp=fp, labels=lab, heads=np.array(heads), base_ce=base)

    tok = load_tokenizer()
    is_ind = np.load(RUNS / "is_induction.npy")
    lines = [f"# Fingerprint clusters: {gateset} on {tag}\n",
             f"{len(idx)} sampled tokens, k={k}. Fingerprint = per-head knock-out ΔCE "
             f"(heads: {heads}).\n"]
    for c in range(k):
        m = lab == c
        mean_fp = fp[m].mean(0)
        top = np.argsort(-mean_fp)[:3]
        lines.append(f"## Cluster {c} — n={m.sum()} ({m.mean():.0%}), "
                     f"induction-pattern {is_ind[idx[m]].mean():.0%}")
        lines.append("top heads by mean ΔCE: " +
                     ", ".join(f"L{heads[t][0]}H{heads[t][1]} (+{mean_fp[t]:.2f})" for t in top) + "\n")
        for i in rng.choice(np.where(m)[0], min(8, m.sum()), replace=False):
            tpos = int(idx[i]) // N_CTX * N_CTX + int(idx[i]) % N_CTX + 1
            ctx = tok.decode(list(data[max(0, tpos - CTX_SHOW):tpos]))
            lines.append(f"- `...{ctx}` ⟶ **`{tok.decode([int(data[tpos])])}`**")
        lines.append("")
    Path(f"fingerprint_report_{gateset}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote fingerprint_report_{gateset}.md", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if "--sample" in a:
        i = a.index("--sample"); kw["sample"] = int(a[i + 1]); del a[i:i + 2]
    if "--k" in a:
        i = a.index("--k"); kw["k"] = int(a[i + 1]); del a[i:i + 2]
    main(*(a or ["gated_depth3", "attn3-seed0"]), **kw)
