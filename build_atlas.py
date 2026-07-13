"""Build the HTML circuit atlas (atlas/) — one page per discovered circuit.

Self-contained pages: figures embedded as base64 PNGs, numbers pulled live from
runs_lm/*.json and the differential/fingerprint outputs. Rebuild anytime with:

    python build_atlas.py
"""

import base64
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from palette import INK, SECONDARY
from text_data import CORPUS, RUNS as RUNS_DIR

RUNS = RUNS_DIR
ATLAS = Path("atlas")
ATLAS.mkdir(exist_ok=True)

CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #fcfcfb;
       color: #0b0b0b; }
@media (prefers-color-scheme: dark) { body { background: #191918; color: #ebeae7; }
  .card { background: #242422 !important; } code { background: #333 !important; } }
.col { max-width: 960px; margin: 0 auto; padding: 24px 20px 80px; }
h1 { font-size: 26px; margin: 12px 0 4px; } h2 { font-size: 19px; margin: 32px 0 8px; }
.sub { color: #8a8883; font-size: 14px; margin-bottom: 24px; }
img { max-width: 100%; border-radius: 8px; margin: 8px 0; }
table { border-collapse: collapse; font-size: 14px; margin: 12px 0; }
th, td { border: 1px solid #8a888344; padding: 5px 12px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
code { background: #f0efec; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.card { background: #f6f5f2; border-radius: 8px; padding: 12px 16px; margin: 10px 0;
        font-size: 14px; line-height: 1.5; }
.ex { font-family: ui-monospace, Menlo, monospace; font-size: 12.5px; margin: 6px 0; }
.ex b { color: #e34948; }
.finding { border-left: 3px solid #3987e5; padding-left: 12px; margin: 14px 0; }
a { color: #3987e5; }
"""


def b64(fig):
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def img(fig, alt=""):
    return f'<img src="data:image/png;base64,{b64(fig)}" alt="{alt}">'


def png_b64(path):
    return ('<img src="data:image/png;base64,'
            + base64.b64encode(Path(path).read_bytes()).decode() + '">')


def formation_figure():
    """Bilinear vs softmax formation on the depth-2-gated tokens."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    curves = {
        "bilinear attn2": (["attn2-dense-seed0/gated2_ce_curve.json",
                            "attn2-dense-seed0/gated_depth2_ce_curve.json"], "#3987e5"),
        "bilinear attn3": (["attn3-dense-seed0/gated_depth3_ce_curve.json"], "#104281"),
        "block2 (attn+MLP ×2)": (["block2-dense-seed0/gated_depth2_ce_curve.json"], "#e39a3b"),
        # tiny-corpus softmax controls (archived; absent under runs_owt after descope)
        "softmax attn2 (lerp)": (["attn2-softmax-dense-seed0/gated_depth2_ce_curve.json"], "#e34948"),
        "softmax attn2 (add)": (["attn2-softmax-add-dense-seed0/gated_depth2_ce_curve.json"], "#8c2b2b"),
    }
    for label, (paths, color) in curves.items():
        f = next((RUNS / p for p in paths if (RUNS / p).exists()), None)
        if f is None:
            continue
        d = {int(k): v for k, v in json.loads(f.read_text()).items()}
        ax.plot(sorted(d), [d[k] for k in sorted(d)], color=color, label=label, lw=2)
    from text_data import VOCAB
    ax.axhline(np.log(VOCAB), color=SECONDARY, ls=":", lw=1, label=f"uniform (ln {VOCAB})")
    ax.set(xlabel="training step", ylabel="CE on depth-2-gated tokens (nats)",
           title="Induction formation on identical data: smooth (bilinear) vs absent/slow (softmax-lerp)")
    ax.legend()
    return fig


def examples_html(report, section, n=10):
    txt = Path(report).read_text()
    m = re.search(rf"### Examples gated at {section}.*?\n\n(.*?)(\n###|\n## |\Z)", txt, re.S)
    if not m:
        return ""
    out = []
    for line in m.group(1).strip().splitlines()[:n]:
        line = line.lstrip("- ").replace("`", "")
        line = re.sub(r"⟶ \*\*(.*?)\*\*", r"⟶ <b>\1</b>", line)
        out.append(f'<div class="ex">{line}</div>')
    return "\n".join(out)


def induction_page():
    dyn = json.loads((RUNS / "attn2-seed0/induction_dynamics.json").read_text())
    abl = dyn["ablation_last_layer"]
    heads_fig = png_b64("figures/induction_dynamics_attn2-seed0.png")
    parts = [f"<style>{CSS}</style><div class='col'>",
             "<a href='index.html'>← atlas</a>",
             "<h1>Induction (depth-2 circuit)</h1>",
             "<div class='sub'>Discovered unsupervised as the depth-2-gated datapoints; "
             "the mold for all deeper circuits. Models: bilinear attn-only, RMSNorm, "
             "TinyStories BPE-1024. 2026-07-08.</div>",

             "<h2>The datapoints</h2>",
             "<div class='card'>A token is <b>depth-2-gated</b> when the 2-layer model "
             "learns it (median CE &lt; 0.5 nats over seeds) and the 1-layer model "
             "provably does not (median CE &gt; 1.5). 175,883 tokens = 1.17% of the "
             "15M-token frozen val stream. 48.6% match the bigram-induction pattern "
             "(completion of a bigram seen earlier in context; base rate 11.7%).</div>",
             examples_html(f"differential_report_{CORPUS}.md", "attn2"),

             "<h2>Formation dynamics: smooth, no phase change</h2>",
             "<div class='finding'>On an exact dense-checkpoint replay, CE on the gated "
             "tokens falls smoothly 4.63 → 0.27 nats — <b>the classic softmax "
             "plateau→phase-change is absent in bilinear attention</b>. The softmax-lerp "
             "control never forms induction in 40k steps (plateau ≈ 1.5 nats, max "
             "induction attention 0.017).</div>",
             img(formation_figure(), "formation curves"),

             "<h2>Heads: distributed, signed, non-redundant</h2>",
             f"<div class='finding'>Strongest progress measure is a <b>negative</b>-score "
             f"head (L1H2, −0.11) — legal without softmax. No redundancy: every layer-1 "
             f"knock-out hurts (ind-token CE {dyn['ind_ce'][-1]:.2f} → "
             f"{max(abl['knockout']):.2f} worst), every solo head is far worse "
             f"({min(abl['solo']):.2f}–{max(abl['solo']):.2f}). Contrast: Singh et al. "
             f"found additive, redundant softmax induction heads.</div>",
             heads_fig,
             "</div>"]
    (ATLAS / "induction.html").write_text("\n".join(parts))


def index_page():
    rows = []
    for run in sorted(RUNS.glob("*/history.jsonl")):
        hist = [json.loads(l) for l in run.read_text().splitlines()]
        final = [h for h in hist if "val_ce" in h][-1]
        rows.append(f"<tr><td>{run.parent.name}</td><td>{final['step']}</td>"
                    f"<td>{final['val_ce']:.3f}</td></tr>")
    parts = [f"<style>{CSS}</style><div class='col'>",
             "<h1>Circuit atlas — deeper circuits on natural text</h1>",
             "<div class='sub'>Bilinear (tensor) attention ladders on TinyStories; "
             "depth-gated datapoints → circuits. See PLAN.md / results_deeper.md.</div>",
             "<h2>Circuits</h2>",
             "<div class='card'><a href='induction.html'>Induction (depth 2)</a> — "
             "match-and-copy of bigrams seen in context. Status: discovered, dynamics "
             "characterized, softmax controls in progress.</div>",
             "<div class='card'>Depth-3 candidates — 122,908 gated tokens (0.82%), only "
             "23.6% induction-pattern; fingerprint clustering v1 done (head-dominance "
             "partition). Status: characterizing.</div>",
             "<div class='card'>Chained k-hop retrieval (toy isolation, prior session) — "
             "see results_hop.md: per-layer pointer advance in a rotated basis; 1/3 seed "
             "lottery at any depth; curriculum backfires.</div>",
             "<h2>Model ladder (final quick-val CE)</h2>",
             "<table><tr><th>run</th><th>steps</th><th>val CE</th></tr>",
             *rows, "</table></div>"]
    (ATLAS / "index.html").write_text("\n".join(parts))


if __name__ == "__main__":
    induction_page()
    index_page()
    print("atlas rebuilt:", [p.name for p in ATLAS.glob("*.html")])
