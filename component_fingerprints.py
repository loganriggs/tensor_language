"""Component-level attribution clustering for deep bilinear transformers (v3).

Components: every attention HEAD and every bilinear MLP layer. For sampled gated
tokens, fingerprint = ΔCE per component knockout (attention heads zeroed pre-W_O;
MLP layers skipped — add-residual passes through). Then:
  - k-means over L2-normalized fingerprints (circuit-similarity clusters)
  - per-token LOAD-BEARING COUNT: #components with ΔCE > 0.5 nats (the causal
    "how many components does this task need" measure, Logan's protocol)
  - per-cluster report with decoded examples, mean fingerprint, component count —
    formatted as JSON for a labeling subagent.

Usage: TL_CORPUS=tiny python component_fingerprints.py <gateset> <run-tag> [--sample 3000] [--k 8]
Writes <RUNS>/cfp_<gateset>_<tag>.npz + cfp_report_<gateset>_<tag>.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, einsum
from sklearn.cluster import KMeans

from deep_model import BilinearMLP
from lm_eval import load_model
from model import Attention
from text_data import N_CTX, RUNS, load_tokenizer, val_windows

DEVICE = "cuda"
BATCH = 96
CTX_SHOW = 45
LOAD_BEARING = 0.5


def components_of(model):
    comps = []
    for li, layer in enumerate(model.layers):
        if isinstance(layer, Attention):
            comps += [("head", li, hi) for hi in range(layer.n_head)]
        elif isinstance(layer, BilinearMLP):
            comps.append(("mlp", li, 0))
    return comps


def forward_ko(model, x, ko=None):
    """ko = ('head', layer, head) zeroes that head pre-W_O; ('mlp', layer, 0) skips
    the MLP (add residual → identity)."""
    h = model.embed(x)
    for li, layer in enumerate(model.layers):
        if isinstance(layer, Attention):
            pat = layer.pattern(h)
            v = rearrange(layer.v(layer.norm(h)), "b t (n d) -> b t n d", n=layer.n_head)
            z = einsum(pat, v, "b n q k, b k n d -> b q n d")
            if ko is not None and ko[0] == "head" and ko[1] == li:
                z = z.clone()
                z[:, :, ko[2], :] = 0
            h = torch.lerp(h, layer.o(rearrange(z, "b q n d -> b q (n d)")), layer.scale)
        else:
            if ko is not None and ko[0] == "mlp" and ko[1] == li:
                continue
            h = layer(h)
    return model.head(h)


def main(gateset, tag, sample=3000, k=8, seed=0):
    run = RUNS / tag
    model = load_model(run, None, DEVICE)
    comps = components_of(model)
    names = [f"L{li}{'H' + str(hi) if kind == 'head' else 'MLP'}" for kind, li, hi in comps]
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
    print(f"{gateset} on {tag}: {len(idx)} tokens, {len(uw)} windows, "
          f"{len(comps)} components", flush=True)

    def ce_at(ko):
        ces = []
        with torch.no_grad():
            for i in range(0, len(b), BATCH):
                lg = forward_ko(model, b[i:i + BATCH, :-1], ko)
                ces.append(F.cross_entropy(lg.transpose(1, 2), b[i:i + BATCH, 1:],
                                           reduction="none"))
        return torch.cat(ces)[rows, pos].cpu().numpy()

    base = ce_at(None)
    fp = np.stack([ce_at(ko) - base for ko in comps], 1)     # (n_tok, n_comp)
    print("fingerprints done; mean |ΔCE|:", dict(zip(names, np.abs(fp).mean(0).round(2))), flush=True)

    lb_count = (fp > LOAD_BEARING).sum(1)                     # causal component count
    norm = fp / (np.linalg.norm(fp, axis=1, keepdims=True) + 1e-8)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(norm)
    lab = km.labels_
    np.savez(RUNS / f"cfp_{gateset}_{tag}.npz", idx=idx, fp=fp, labels=lab,
             base_ce=base, lb_count=lb_count, names=np.array(names))

    tok = load_tokenizer()
    is_ind = np.load(RUNS / "is_induction.npy")
    clusters = []
    for c in range(k):
        m = lab == c
        mean_fp = fp[m].mean(0)
        order = np.argsort(-mean_fp)
        exs = []
        for i in rng.choice(np.where(m)[0], min(20, int(m.sum())), replace=False):
            tpos = (int(idx[i]) // N_CTX) * N_CTX + int(idx[i]) % N_CTX + 1
            exs.append({"context": tok.decode(list(data[max(0, tpos - CTX_SHOW):tpos])),
                        "target": tok.decode([int(data[tpos])]),
                        "load_bearing_components": int(lb_count[i])})
        clusters.append({
            "cluster": c, "n": int(m.sum()),
            "induction_pattern_frac": round(float(is_ind[idx[m]].mean()), 3),
            "mean_load_bearing_count": round(float(lb_count[m].mean()), 2),
            "top_components_by_mean_dCE": [
                {"component": names[o], "mean_dCE": round(float(mean_fp[o]), 3)}
                for o in order[:5]],
            "examples": exs})
    report = {"gateset": gateset, "model": tag, "components": names,
              "n_sampled": len(idx), "load_bearing_threshold_nats": LOAD_BEARING,
              "clusters": clusters}
    out = RUNS / f"cfp_report_{gateset}_{tag}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    kw = {}
    if "--sample" in a:
        i = a.index("--sample"); kw["sample"] = int(a[i + 1]); del a[i:i + 2]
    if "--k" in a:
        i = a.index("--k"); kw["k"] = int(a[i + 1]); del a[i:i + 2]
    main(a[0], a[1], **kw)
