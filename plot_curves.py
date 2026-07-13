"""Training curves for all runs of the active corpus (TL_CORPUS), log y.

Panel A: quick-val CE vs step (log y).
Panel B: improvement rate |ΔCE|/1k steps (log y) — the stopping-point view: once a
curve drops below ~1e-3 nats/1k steps, further training buys almost nothing.
Caveat: runs use a cosine LR schedule tied to their horizon, so a longer-STEPS run is
not simply an extension of a shorter one (its LR is still high mid-run).

Usage: python plot_curves.py            -> figures/training_curves_<corpus>.png
"""

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from text_data import CORPUS, RUNS
from palette import SECONDARY

COLORS = {"attn1": "#c7b299", "attn2": "#3987e5", "attn3": "#104281", "attn4": "#0d366b",
          "attn5": "#082448", "softmax": "#e34948", "softmax-add": "#8c2b2b"}


def series():
    out = {}
    for hist in sorted(RUNS.glob("*/history.jsonl")):
        rows = [json.loads(l) for l in hist.read_text().splitlines() if "val_ce" in l]
        if len(rows) < 3:
            continue
        out[hist.parent.name] = (np.array([r["step"] for r in rows]),
                                 np.array([r["val_ce"] for r in rows]))
    return out


def color_of(tag):
    for key in ("softmax-add", "softmax"):
        if key in tag:
            return COLORS[key]
    return COLORS.get(tag.split("-")[0], SECONDARY)


if __name__ == "__main__":
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.6))
    seen = set()
    for tag, (s, v) in series().items():
        c = color_of(tag)
        fam = tag.rsplit("-seed", 1)[0].replace("-dense", "")
        label = fam if fam not in seen else None
        seen.add(fam)
        axes[0].plot(s, v, color=c, alpha=0.75, lw=1.3, label=label)
        if len(s) > 6:
            ds = np.diff(v) / (np.diff(s) / 1000.0)
            k = min(5, len(ds))
            smooth = np.convolve(-ds, np.ones(k) / k, mode="valid")
            axes[1].plot(s[1:len(smooth) + 1], np.clip(smooth, 1e-5, None),
                         color=c, alpha=0.75, lw=1.3)
    for ax in axes:
        ax.set_yscale("log")
        ax.axvline(40_000, color=SECONDARY, ls=":", lw=1)
        ax.set_xlabel("step")
    axes[0].set(ylabel="quick-val CE (nats, log)", title=f"{CORPUS}: val CE")
    axes[0].legend(fontsize=8)
    axes[1].axhline(1e-3, color=SECONDARY, ls="--", lw=1)
    axes[1].set(ylabel="improvement rate (nats / 1k steps, log)",
                title="stopping view: |ΔCE|/1k steps (5-pt smoothed); dashed = 1e-3")
    fig.tight_layout()
    out = f"figures/training_curves_{CORPUS}.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
