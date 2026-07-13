"""Find depth-gated datapoints: val tokens that a depth-d model learns and depth-(d-1)
provably does not — PLAN.md step 3.

Reads runs_lm/<spec>-seed<k>/val_ce.npy matrices (from lm_eval.py). Definitions
(hysteresis gap per PLAN.md gotcha #5, medians over seeds so one lottery seed can't
fake or hide a gate):

    learned(d)   : median_seeds CE  <  TAU_LEARNED   (0.5 nats ~= p(correct) > 0.6)
    unlearned(d) : median_seeds CE  >  TAU_UNLEARNED (1.5 nats)
    gated at d   : learned(d) AND unlearned(d-1)

Outputs:
    runs_lm/gated_depth<d>.npy    flat indices (win * N_CTX + pos) of gated tokens
    differential_report.md        mean-CE monotonicity table (sanity gate #1),
                                  gate counts, decoded example contexts per depth

Usage: python differential.py [--specs attn1,attn2,attn3,attn4]
"""

import json
import re
import sys
from pathlib import Path

import numpy as np

from text_data import CORPUS, N_CTX, RUNS as RUNS_DIR, load_tokenizer, tokens

TAU_LEARNED = 0.5
TAU_UNLEARNED = 1.5     # override with --tau L,U
N_EXAMPLES = 25          # decoded examples per depth in the report
CTX_SHOW = 45            # tokens of context to decode before a gated token

RUNS = RUNS_DIR


def collect(specs):
    """{spec: (n_seeds, n_tok) float32 CE} for runs with val_ce.npy, default head count."""
    out = {}
    for spec in specs:
        mats = []
        for f in sorted(RUNS.glob(f"{spec}-seed*/val_ce.npy")):
            mats.append(np.load(f).astype(np.float32).reshape(-1))
        if mats:
            out[spec] = np.stack(mats)
    return out


def main(specs):
    ce = collect(specs)
    if len(ce) < 2:
        sys.exit(f"need >=2 specs with val_ce.npy, have {list(ce)}")
    tok = load_tokenizer()
    val = tokens("val")
    lines = ["# Differential datapoints: depth-gated tokens\n",
             f"Thresholds: learned < {TAU_LEARNED} nats (median over seeds), "
             f"unlearned > {TAU_UNLEARNED} nats.\n",
             "## Sanity gate: mean val CE by depth (must be monotone non-increasing)\n",
             "| spec | " + " | ".join(f"seed{i}" for i in range(len(next(iter(ce.values()))))) +
             " | median |", "|---|" + "---|" * (len(next(iter(ce.values()))) + 1)]
    med_prev, name_prev, monotone = None, None, True
    medians = {}
    for spec in specs:
        if spec not in ce:
            continue
        m = ce[spec]
        med = np.median(m, 0)
        medians[spec] = med
        row = " | ".join(f"{m[i].mean():.4f}" for i in range(len(m)))
        lines.append(f"| {spec} | {row} | {med.mean():.4f} |")
        if med_prev is not None and med.mean() > med_prev + 1e-3:
            monotone = False
            lines.append(f"| **WARNING** | {spec} mean CE above {name_prev} — optimization failure? | | |")
        med_prev, name_prev = med.mean(), spec
    lines.append("\nMonotone: " + ("YES" if monotone else "**NO — fix before believing gates**"))

    lines.append("\n## Depth-gated token counts\n")
    order = [s for s in specs if s in medians]
    for shallow, deep in zip(order, order[1:]):
        gated = np.where((medians[deep] < TAU_LEARNED) & (medians[shallow] > TAU_UNLEARNED))[0]
        depth = len(json.loads((next(RUNS.glob(f"{deep}-seed*/config.json"))).read_text())["spec"])
        np.save(RUNS / f"gated_depth{depth}.npy", gated)
        frac = len(gated) / medians[deep].size
        lines.append(f"- **{shallow} → {deep}**: {len(gated)} tokens gated "
                     f"({frac:.4%} of val stream) → `runs_lm/gated_depth{depth}.npy`")

        lines.append(f"\n### Examples gated at {deep} (context ⟶ **token**, CE {shallow}→{deep})\n")
        rng = np.random.default_rng(0)
        for idx in rng.choice(gated, min(N_EXAMPLES, len(gated)), replace=False):
            w, p = divmod(int(idx), N_CTX)
            tpos = w * N_CTX + p + 1                     # predicted token in the val stream
            ctx = tok.decode(list(val[max(0, tpos - CTX_SHOW):tpos]))
            target = tok.decode([int(val[tpos])])
            lines.append(f"- `...{ctx}` ⟶ **`{target}`** "
                         f"({medians[shallow][idx]:.2f}→{medians[deep][idx]:.2f})")
        lines.append("")
    Path(f"differential_report_{CORPUS}.md").write_text("\n".join(lines) + "\n")
    print(f"wrote differential_report_{CORPUS}.md (monotone={monotone})")


if __name__ == "__main__":
    specs = ["attn1", "attn2", "attn3", "attn4"]
    if "--specs" in sys.argv:
        specs = sys.argv[sys.argv.index("--specs") + 1].split(",")
    if "--tau" in sys.argv:
        L, U = sys.argv[sys.argv.index("--tau") + 1].split(",")
        TAU_LEARNED, TAU_UNLEARNED = float(L), float(U)
        import differential as _d
        _d.TAU_LEARNED, _d.TAU_UNLEARNED = TAU_LEARNED, TAU_UNLEARNED
    main(specs)
