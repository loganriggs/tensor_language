"""Induction-circuit formation on natural text — the paper-mold analysis (PLAN.md step 5,
following Singh et al. 2024) applied to the text ladder's checkpoints.

For one run (default attn2-seed0), across its training checkpoints:
  1. val CE split into INDUCTION-PREDICTABLE tokens (their bigram completion already
     appeared earlier in the window: exists j<q with tok[j]==tok[q], tok[j+1]==tok[q+1])
     vs all other tokens — the loss decomposition that shows the phase change lives in
     the induction datapoints.
  2. per-head induction attention score (progress measure): mean attention weight from
     position q to j_last+1 (the token after the most recent earlier occurrence of
     tok[q]), at induction-predictable positions. Bilinear attention has no softmax, so
     this is the raw masked score — comparable across steps, not a probability.
  3. at the final checkpoint: per-head knock-out AND knock-all-but-one ablations
     (paper §3.1 — knock-out alone understates redundant heads), scored as induction-token
     CE. Bilinear heads are ablated by zeroing their slice of the concatenated head
     output (pre-W_O), like hop_ablate.

Outputs: figures/induction_dynamics_<tag>.png + runs_lm/<tag>/induction_dynamics.json

Usage: python induction_dynamics.py [attn2-seed0] [--windows 100]
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
from einops import rearrange, einsum

from lm_eval import load_model
from text_data import VOCAB, N_CTX, RUNS, val_windows
from palette import INK, SECONDARY

DEVICE = "cuda"


def val_batch(n_windows):
    data, n_win = val_windows()
    buf = np.stack([data[w * N_CTX:w * N_CTX + N_CTX + 1]
                    for w in range(min(n_windows, n_win))]).astype(np.int64)
    return torch.from_numpy(buf).to(DEVICE)


def induction_structure(b):
    """For tokens x = b[:, :-1]: j_last (B, T) = latest j<q with x[j]==x[q] (-1 if none),
    and is_ind (B, T) = position q's TARGET b[:, q+1] equals x[j_last+1] (bigram repeat)."""
    x = b[:, :-1]
    B, T = x.shape
    idx = torch.arange(T, device=x.device)
    match = (x[:, None, :] == x[:, :, None]) & (idx[None, :] < idx[:, None])  # (B, q, j)
    j_last = (match * (idx + 1)).amax(-1) - 1                                  # (B, T)
    has = j_last >= 0
    nxt = torch.where(has, j_last + 1, 0)
    completion = x.gather(1, nxt)                    # x[j_last+1] (j_last+1 <= q always)
    is_ind = has & (b[:, 1:] == completion) & (nxt < idx[None, :])  # exclude j+1 == q self
    return j_last, is_ind


def head_patterns(model, x):
    """Stack of per-layer patterns (n_layers list of (B, H, T, T)) + head outputs kept
    for ablations: returns list of (pattern, v) and the residual stream inputs."""
    pats = []
    h = model.embed(x)
    for layer in model.layers:
        pats.append(layer.pattern(h))
        h = layer(h)
    return pats


def ce_split(model, b, is_ind, head_mask=None):
    """Mean CE on induction vs other tokens. head_mask: {layer: bool tensor (H,)} keeps
    only masked heads' contributions (True = keep) by zeroing head slices pre-W_O."""
    x = b[:, :-1]
    h = model.embed(x)
    for li, layer in enumerate(model.layers):
        if head_mask is None or li not in head_mask:
            h = layer(h)
        else:
            v = rearrange(layer.v(layer.norm(h)), "b t (n d) -> b t n d", n=layer.n_head)
            z = einsum(layer.pattern(h), v, "b n q k, b k n d -> b q n d")
            z = z * head_mask[li].view(1, 1, -1, 1).to(z)
            z = rearrange(z, "b q n d -> b q (n d)")
            h = torch.lerp(h, layer.o(z), layer.scale) if layer.residual == "lerp" else h + layer.o(z)
    logits = model.head(h)
    ce = F.cross_entropy(logits.transpose(1, 2), b[:, 1:], reduction="none")
    return ce[is_ind].mean().item(), ce[~is_ind].mean().item()


def main(tag="attn2-seed0", n_windows=100):
    run = RUNS / tag
    b = val_batch(n_windows)
    j_last, is_ind = induction_structure(b)
    print(f"{tag}: {is_ind.float().mean():.3%} of val tokens are induction-predictable", flush=True)

    ckpts = sorted(run.glob("ckpt/step*.pt"), key=lambda p: int(re.findall(r"\d+", p.name)[0]))
    steps, results = [], {"ind_ce": [], "other_ce": [], "head_scores": []}
    x = b[:, :-1]
    q_idx = torch.arange(x.shape[1], device=DEVICE)
    tgt_pos = torch.where(j_last >= 0, j_last + 1, 0)
    with torch.no_grad():
        for ck in ckpts + [None]:
            model = load_model(run, ck.stem if ck else None, DEVICE)
            step = int(re.findall(r"\d+", ck.name)[0]) if ck else \
                json.loads((run / "config.json").read_text())["steps"]
            ice, oce = ce_split(model, b, is_ind)
            pats = head_patterns(model, x)
            scores = []
            for pat in pats:                          # (B, H, T, T)
                w = pat.gather(3, tgt_pos[:, None, :, None].expand(-1, pat.shape[1], -1, 1))
                scores.append(w.squeeze(-1).permute(1, 0, 2)[:, is_ind].mean(-1).tolist())
            steps.append(step)
            results["ind_ce"].append(ice); results["other_ce"].append(oce)
            results["head_scores"].append(scores)
            print(f"  step {step}: ind CE {ice:.3f} other CE {oce:.3f}", flush=True)

        # final-checkpoint ablations on the LAST attention layer's heads
        model = load_model(run, None, DEVICE)
        L = len(model.layers) - 1
        H = model.layers[L].n_head
        abl = {"knockout": [], "solo": []}
        for h in range(H):
            keep = torch.ones(H, dtype=torch.bool); keep[h] = False
            abl["knockout"].append(ce_split(model, b, is_ind, {L: keep})[0])
            solo = torch.zeros(H, dtype=torch.bool); solo[h] = True
            abl["solo"].append(ce_split(model, b, is_ind, {L: solo})[0])

    out = {"steps": steps, **results, "ablation_last_layer": abl,
           "ind_frac": is_ind.float().mean().item(), "n_windows": n_windows}
    (run / "induction_dynamics.json").write_text(json.dumps(out))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(steps, results["ind_ce"], color="#3987e5", label="induction tokens")
    axes[0].plot(steps, results["other_ce"], color=SECONDARY, label="other tokens")
    axes[0].set(xlabel="step", ylabel="val CE (nats)", title=f"{tag}: loss split")
    axes[0].legend()
    hs = np.array(results["head_scores"])              # (ckpt, layer, head)
    for li in range(hs.shape[1]):
        for h in range(hs.shape[2]):
            axes[1].plot(steps, hs[:, li, h], label=f"L{li}H{h}",
                         ls="-" if li == hs.shape[1] - 1 else ":")
    axes[1].set(xlabel="step", ylabel="attn score to induction target",
                title="per-head induction score (dotted = earlier layers)")
    axes[1].legend(fontsize=6, ncol=2)
    xs = np.arange(len(abl["knockout"]))
    axes[2].bar(xs - 0.2, abl["knockout"], 0.4, label="knock-out", color=SECONDARY)
    axes[2].bar(xs + 0.2, abl["solo"], 0.4, label="all-but-this", color="#3987e5")
    axes[2].axhline(results["ind_ce"][-1], color=INK, lw=0.8, ls="--", label="full model")
    axes[2].set(xlabel=f"layer-{len(model.layers)-1} head", ylabel="induction-token CE",
                title="final ablations (paper §3.1)")
    axes[2].legend()
    fig.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    fig.savefig(f"figures/induction_dynamics_{tag}.png", dpi=150)
    print(f"wrote figures/induction_dynamics_{tag}.png", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    n_windows = 100
    if "--windows" in args:
        i = args.index("--windows"); n_windows = int(args[i + 1]); del args[i:i + 2]
    main(args[0] if args else "attn2-seed0", n_windows)
