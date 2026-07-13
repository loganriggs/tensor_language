"""H2 test: does the multi-family bilin-lerp-2L abandon the task-specific shortcuts
of its single-task counterparts? Same mech battery, side by side.

Usage: python mech_general.py   (writes figures/gen_mech.png and prints tables)
"""

import matplotlib.pyplot as plt
import torch

import analysis
import analysis_geo
import analysis_general
from data import N_CTX as CYC_CTX
from data import eval_sets as cyc_evals
from data import sample_cycles
from geodata import N_CTX as GEO_CTX
from geodata import TAIL
from geodata import eval_sets as geo_evals
from geodata import walk_batch
from mech import VARIANTS, ablated_logits, offset_profile, pattern_from

torch.set_grad_enabled(False)
FIGURES = analysis.FIGURES

MODELS = {
    "single-task cycle (L2_d64)": analysis.load_model("L2_d64_h1"),
    "single-task grid (L2_d128)": analysis_geo.load_model("grid_L2_d128_long"),
    "multi-family (bilin-lerp-2L)": analysis_general.load_model("bilin-lerp-2L"),
}


def cycle_battery(model, tag):
    model = model.cuda()
    evals = {k: (t.cuda(), m.cuda()) for k, (t, m) in cyc_evals().items()}
    rows = {}
    for variant in VARIANTS:
        accs = []
        for t, m in evals.values():
            preds = ablated_logits(model, t, variant)[:, :-1].argmax(-1)
            accs.append(((preds == t[:, 1:]) * m).sum().item() / m.sum().item())
        rows[variant] = accs
    print(f"\n[{tag}] cycle-doc ablations (accuracy at L=5..30):")
    for variant, accs in rows.items():
        print(f"  {variant:26s}" + "".join(f" {a:5.2f}" for a in accs))
    model.cpu()
    return rows


def grid_battery(model, tag):
    model = model.cuda()
    tokens, legal = [z.cuda() for z in geo_evals("grid", n_seq=64)["in"]]
    rows = {}
    for variant in VARIANTS:
        logits = ablated_logits(model, tokens, variant)[:, TAIL:-1]
        lg = legal[:, TAIL:-1]
        rate = lg.gather(2, logits.argmax(-1, keepdim=True)).float().mean().item()
        mass = (logits.softmax(-1) * lg).sum(-1).mean().item()
        rows[variant] = (rate, mass)
    print(f"\n[{tag}] grid-doc ablations (legal / mass):")
    for variant, (r, m) in rows.items():
        print(f"  {variant:26s} {r:6.3f} {m:6.3f}")
    model.cpu()
    return rows


def profiles(model, tokens, n_ctx):
    model = model.cuda()
    l1, l2 = model.layers[0], model.layers[-1]
    x = model.embed(tokens.cuda())
    p1 = l1.pattern(x)[:, 0]
    p2 = pattern_from(l2, l1(x), l1(x))
    out = offset_profile(p1, n_ctx, 30), offset_profile(p2, n_ctx, 30)
    model.cpu()
    return out


def main():
    cyc_tokens = sample_cycles(256, torch.full((256,), 7), generator=torch.Generator().manual_seed(5))
    grid_tokens, _, _ = walk_batch(128, (4, 4), "grid", torch.Generator().manual_seed(5))

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharex=True)
    pairs = [("single-task cycle (L2_d64)", "multi-family (bilin-lerp-2L)"),
             ("single-task grid (L2_d128)", "multi-family (bilin-lerp-2L)")]
    for row, ((a, b), tokens, n_ctx, doc) in enumerate(zip(pairs, (cyc_tokens, grid_tokens), (CYC_CTX, GEO_CTX),
                                                           ("cycle docs (L=7)", "grid docs (4×4)"))):
        for col, name in enumerate((a, b)):
            p1, p2 = profiles(MODELS[name], tokens, n_ctx)
            ax = axes[row][col]
            offs = range(len(p1))
            ax.bar([o - 0.2 for o in offs], p1, width=0.4, color="#eda100", label="layer 1")
            ax.bar([o + 0.2 for o in offs], p2, width=0.4, color="#2a78d6", label="layer 2 (final)")
            ax.axhline(0, color="#c3c2b7", lw=0.8)
            ax.set_title(f"{name} · {doc}", fontsize=9)
            if row == 1:
                ax.set_xlabel("offset t−s")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Signed attention by offset: single-task circuits vs the multi-family model on the same documents", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "gen_mech.png", bbox_inches="tight")
    plt.close(fig)

    cycle_battery(MODELS["single-task cycle (L2_d64)"], "single cycle")
    cycle_battery(MODELS["multi-family (bilin-lerp-2L)"], "multi")
    grid_battery(MODELS["single-task grid (L2_d128)"], "single grid")
    grid_battery(MODELS["multi-family (bilin-lerp-2L)"], "multi")


if __name__ == "__main__":
    main()
