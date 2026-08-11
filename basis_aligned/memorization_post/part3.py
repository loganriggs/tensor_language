#!/usr/bin/env python
"""Part 3: 200 facts in 2 bilinear blocks with a residual stream (handoff section
'Part 3', figure F13). Registered predictions: predictions/part3_predictions.md
(P1 attribution not disjoint, P2 cross terms load-bearing, P3 negation/cancellation
direction, P4 ablation asymmetry) — committed before the sizing sweep and measurements.

Architecture: x0 = z (20-dim boolean key); x_{b+1} = x_b + D_b((L_b x_b) * (R_b x_b));
logits = W x_last. No biases anywhere (all-zeros input -> uniform logits exactly).

Stages:
  sweep    size H so ONE layer fails but TWO suffice (2 seeds per point, table saved)
  verify   5-seed runs at the chosen sizes (models saved)
  control  metric positive controls: one block frozen at zero -> attribution must not
           assign facts to the dead layer (gate for every claim below)
  measure  attribution bins, cross-term magnitudes, cancellation cosines, ablations
  figure   F13
Convention: figures display seed 0; cross-seed stats printed and saved to json.
"""
import argparse, json, os, subprocess, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
MODELS = os.path.join(ROOT, "models")
PRED = os.path.join(ROOT, "predictions")
os.makedirs(FIG, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

N_BITS = 20
N_FACTS = 200
N_CLASSES = 10
SEEDS = [0, 1, 2, 3, 4]
STEPS = 15000
LR = 1e-2
DEV = "cuda" if torch.cuda.is_available() else "cpu"

H1_GRID = [8, 12, 16, 20, 24, 32, 40]     # single-block model
H2_GRID = [4, 6, 8, 10, 12, 16, 20]       # per-block H of the 2-block model
SIZE_JSON = os.path.join(FIG, "part3_sizing.json")
MET_JSON = os.path.join(FIG, "part3_metrics.json")

# ----------------------------------------------------------------------------- style
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#898781"; GRIDC = "#e1e0d9"
SURF = "#fcfcfb"; BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRIDC,
})


def save_fig(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=170)
    plt.close(fig)
    print(f"[fig] saved figures/{name}.png/.svg")


def require_committed(path, what):
    r = subprocess.run(["git", "status", "--porcelain", "--", path],
                       cwd=ROOT, capture_output=True, text=True)
    if r.stdout.strip():
        sys.exit(f"{what} prediction file has uncommitted changes — commit before measuring.")


def make_facts():
    rng = np.random.default_rng(300)
    seen, Z = set(), []
    while len(Z) < N_FACTS:
        z = tuple(int(v) for v in rng.integers(0, 2, N_BITS))
        if sum(z) > 0 and z not in seen:
            seen.add(z)
            Z.append(z)
    Z = np.array(Z, dtype=np.float64)
    y = rng.integers(0, N_CLASSES, N_FACTS)
    return Z, y


def init_params(H, n_blocks, seed, requires_grad=True):
    g = torch.Generator().manual_seed(1000 + seed)
    ps = {}
    for b in range(n_blocks):
        ps[f"L{b}"] = torch.randn(H, N_BITS, generator=g, dtype=torch.float64) * N_BITS ** -0.5
        ps[f"R{b}"] = torch.randn(H, N_BITS, generator=g, dtype=torch.float64) * N_BITS ** -0.5
        ps[f"D{b}"] = torch.randn(N_BITS, H, generator=g, dtype=torch.float64) * 0.5 * H ** -0.5
    ps["W"] = torch.randn(N_CLASSES, N_BITS, generator=g, dtype=torch.float64) * N_BITS ** -0.5
    for k in ps:
        ps[k] = ps[k].to(DEV).requires_grad_(requires_grad)
    return ps


def fwd(ps, Z, n_blocks, use=None):
    x = Z
    for b in range(n_blocks):
        if use is None or use[b]:
            h = (x @ ps[f"L{b}"].T) * (x @ ps[f"R{b}"].T)
            x = x + h @ ps[f"D{b}"].T
    return x @ ps["W"].T


def train(H, n_blocks, seed, Z, y, freeze_zero=None, steps=STEPS):
    """freeze_zero: block index whose D is fixed at zero (block disabled but present)."""
    ps = init_params(H, n_blocks, seed)
    if freeze_zero is not None:
        for k in (f"L{freeze_zero}", f"R{freeze_zero}", f"D{freeze_zero}"):
            ps[k] = (ps[k] * 0).detach().requires_grad_(False)
    Zt = torch.tensor(Z, device=DEV)
    yt = torch.tensor(y, device=DEV)
    params = [p for p in ps.values() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR / 10)
    for step in range(steps):
        logits = fwd(ps, Zt, n_blocks)
        loss = torch.nn.functional.cross_entropy(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        acc = float((fwd(ps, Zt, n_blocks).argmax(1) == yt).float().mean())
    return {k: v.detach().cpu().numpy() for k, v in ps.items()}, acc, float(loss.item())


def np_fwd(ps, Z, n_blocks, use=None):
    x = Z
    for b in range(n_blocks):
        if use is None or use[b]:
            h = (x @ ps[f"L{b}"].T) * (x @ ps[f"R{b}"].T)
            x = x + h @ ps[f"D{b}"].T
    return x @ ps["W"].T


def block_out(ps, Z, b):
    return ((Z @ ps[f"L{b}"].T) * (Z @ ps[f"R{b}"].T)) @ ps[f"D{b}"].T


def margins(logits, y):
    t = logits[np.arange(len(y)), y]
    tmp = logits.copy()
    tmp[np.arange(len(y)), y] = -np.inf
    return t - tmp.max(1)


# ============================================================================= sweep
def stage_sweep():
    print("=" * 70)
    print(f"STAGE sweep: size H so 1 block fails, 2 blocks suffice ({N_FACTS} facts, dev={DEV})")
    print("=" * 70)
    Z, y = make_facts()
    out = {"n_facts": N_FACTS, "one": {}, "two": {}}
    for nb, grid, key in ((1, H1_GRID, "one"), (2, H2_GRID, "two")):
        for H in grid:
            accs = []
            for seed in [0, 1]:
                _, acc, loss = train(H, nb, seed, Z, y)
                accs.append(acc)
            out[key][H] = accs
            print(f"  {nb}-block H={H:3d}: acc {['%.1f%%' % (a*100) for a in accs]}"
                  f"  params/block ~{H*2*N_BITS + N_BITS*H}")
    with open(SIZE_JSON, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  table -> {SIZE_JSON}")


# ============================================================================= verify
def stage_verify():
    with open(SIZE_JSON) as f:
        sz = json.load(f)
    # chosen sizes: smallest 2-block H with both sizing seeds at 100%, such that the
    # 1-block model at the SAME H (and at 2H) is clearly below 100%
    Hstar = None
    for H in H2_GRID:
        if min(sz["two"][str(H)]) == 1.0:
            one_same = max(sz["one"].get(str(H), [0.0]))
            if one_same < 1.0:
                Hstar = H
                break
    if Hstar is None:
        sys.exit("no H satisfies the sizing criterion — adjust N_FACTS/grids (document!).")
    print(f"chosen H* = {Hstar} per block")
    Z, y = make_facts()
    res = {"Hstar": Hstar, "two": [], "one_same": [], "one_double": []}
    for seed in SEEDS:
        ps, acc, loss = train(Hstar, 2, seed, Z, y)
        np.savez(os.path.join(MODELS, f"p3_two_H{Hstar}_s{seed}.npz"), **ps)
        res["two"].append(acc)
        print(f"  2-block H={Hstar} seed{seed}: acc {acc:.1%} loss {loss:.4f}")
    for tag, H in (("one_same", Hstar), ("one_double", 2 * Hstar)):
        for seed in SEEDS:
            ps, acc, loss = train(H, 1, seed, Z, y)
            res[tag].append(acc)
        print(f"  1-block H={H}: accs {['%.1f%%' % (a*100) for a in res[tag]]}")
    with open(SIZE_JSON.replace("sizing", "verify"), "w") as f:
        json.dump(res, f, indent=1)


# ============================================================================= attribution
def attribution_bins(ps, Z, y):
    """Bins over facts: linear (no blocks), l1only, l2only, both, neither —
    'correct' = argmax matches under the given single-layer evaluation."""
    full = np_fwd(ps, Z, 2).argmax(1) == y
    lin = np_fwd(ps, Z, 2, use=[0, 0]).argmax(1) == y
    c1 = np_fwd(ps, Z, 2, use=[1, 0]).argmax(1) == y
    c2 = np_fwd(ps, Z, 2, use=[0, 1]).argmax(1) == y
    bins = {
        "linear": lin,
        "l1only": c1 & ~c2 & ~lin,
        "l2only": c2 & ~c1 & ~lin,
        "both": c1 & c2 & ~lin,
        "neither": ~c1 & ~c2 & ~lin,
    }
    return full, bins, c1, c2


# ============================================================================= control
def stage_control():
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    print("=" * 70)
    print("STAGE control: attribution metric gates (one block frozen at zero)")
    print("=" * 70)
    Z, y = make_facts()
    # H large enough that a single live block memorizes well (from the sweep table)
    with open(SIZE_JSON) as f:
        sz = json.load(f)
    Hc = next((H for H in H1_GRID if min(sz["one"][str(H)]) == 1.0), H1_GRID[-1])
    print(f"  control H = {Hc} (smallest 1-block H at 100% in the sweep)")
    ok = True
    for dead in (1, 0):
        ps, acc, _ = train(Hc, 2, 0, Z, y, freeze_zero=dead)
        full, bins, _, _ = attribution_bins(ps, Z, y)
        mem = full.sum()
        dead_bin = bins["l2only" if dead == 1 else "l1only"][full].sum()
        frac = dead_bin / max(mem, 1)
        verdict = "PASS" if frac <= 0.05 else "FAIL"
        ok &= (frac <= 0.05)
        print(f"  block {dead} frozen at zero: memorized {mem}/{N_FACTS}; "
              f"facts attributed to the DEAD layer: {dead_bin} ({frac:.1%}) [{verdict}]")
        counts = {k: int(v[full].sum()) for k, v in bins.items()}
        print(f"    bins over memorized facts: {counts}")
    if not ok:
        sys.exit("attribution-metric control FAILED — repair the metric before measuring.")
    print("  metric controls PASS")


# ============================================================================= measure
def stage_measure():
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    print("=" * 70)
    print("STAGE measure: attribution, cross terms, cancellation, ablations")
    print("=" * 70)
    Z, y = make_facts()
    with open(SIZE_JSON.replace("sizing", "verify")) as f:
        Hstar = json.load(f)["Hstar"]
    met = {"Hstar": Hstar, "seeds": {}}
    for seed in SEEDS:
        ps = dict(np.load(os.path.join(MODELS, f"p3_two_H{Hstar}_s{seed}.npz")))
        full_logits = np_fwd(ps, Z, 2)
        full = full_logits.argmax(1) == y
        _, bins, c1ok, c2ok = attribution_bins(ps, Z, y)
        b1 = block_out(ps, Z, 0)                     # B1 evaluated on the raw key
        b2 = block_out(ps, Z, 1)                     # B2 evaluated on the raw key
        add_logits = (Z + b1 + b2) @ ps["W"].T       # degree-2 additive surrogate
        cross = full_logits - add_logits             # degree-3/4 composed terms
        add_ok = add_logits.argmax(1) == y
        m_full = margins(full_logits, y)
        m_add = margins(add_logits, y)
        l1vec = b1 @ ps["W"].T
        l2vec = b2 @ ps["W"].T
        def coss(A, B):
            na = np.linalg.norm(A, axis=1); nb = np.linalg.norm(B, axis=1)
            return (A * B).sum(1) / np.maximum(na * nb, 1e-12)
        cos1 = coss(cross, l1vec)
        cos2 = coss(cross, l2vec)
        # relative size of the cross term
        rel = np.linalg.norm(cross, axis=1) / np.maximum(np.linalg.norm(full_logits, axis=1), 1e-12)
        met["seeds"][seed] = {
            "acc_full": float(full.mean()),
            "bins": {k: int(v[full].sum()) for k, v in bins.items()},
            "surv_no_block2": int(c1ok.sum()), "surv_no_block1": int(c2ok.sum()),
            "acc_additive": float(add_ok.mean()),
            "facts_lost_by_additive": int((full & ~add_ok).sum()),
            "median_cos_cross_l1": float(np.median(cos1)),
            "median_cos_cross_l2": float(np.median(cos2)),
            "median_rel_cross": float(np.median(rel)),
            "mean_margin_full": float(m_full[full].mean()),
            "median_margin_delta_cross": float(np.median((m_full - m_add)[full])),
        }
        s = met["seeds"][seed]
        print(f"  seed{seed}: full {s['acc_full']:.1%}; bins {s['bins']}; "
              f"survivors no-b2 {s['surv_no_block2']}, no-b1 {s['surv_no_block1']}; "
              f"additive acc {s['acc_additive']:.1%} (lost {s['facts_lost_by_additive']}); "
              f"med cos(cross,L1) {s['median_cos_cross_l1']:+.3f}, "
              f"cos(cross,L2) {s['median_cos_cross_l2']:+.3f}; "
              f"med |cross|/|logits| {s['median_rel_cross']:.3f}")
        if seed == 0:
            np.savez(os.path.join(FIG, "F13_seed0_data.npz"),
                     m_full=m_full, m_add=m_add, cos1=cos1, cos2=cos2, rel=rel,
                     full=full, add_ok=add_ok,
                     bins=np.array([bins[k].sum() for k in
                                    ("linear", "l1only", "l2only", "both", "neither")]))
    with open(MET_JSON, "w") as f:
        json.dump(met, f, indent=1)
    print(f"  metrics -> {MET_JSON}")


# ============================================================================= figure
def stage_figure():
    d = np.load(os.path.join(FIG, "F13_seed0_data.npz"))
    with open(MET_JSON) as f:
        met = json.load(f)
    names = ["linear", "l1only", "l2only", "both", "neither"]
    seed_bins = np.array([[met["seeds"][str(s)]["bins"][k] for k in names] for s in SEEDS])
    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(names))
    ax.bar(xs, seed_bins[0], 0.62, color=[MUTED, BLUE, ORANGE, AQUA, "#c23b3b"])
    for i in range(len(names)):
        lo, hi = seed_bins[:, i].min(), seed_bins[:, i].max()
        ax.plot([i, i], [lo, hi], color=INK, lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8.5)
    ax.set_ylabel("facts (of 200)")
    ax.set_title("(i) fact attribution by single-layer evaluation\n(bars: seed 0; whiskers: range over 5 seeds)", fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.8); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0, 1])
    delta = d["m_full"] - d["m_add"]
    ax.hist(delta, bins=40, color=BLUE, alpha=0.85)
    ax.axvline(0, color=INK, lw=1)
    lost = int((d["full"] & ~d["add_ok"]).sum())
    ax.set_title(f"(ii) cross-term margin contribution per fact (seed 0)\n"
                 f"degree-2 surrogate loses {lost}/200 facts", fontsize=9.5)
    ax.set_xlabel("margin(full) $-$ margin(additive)")
    ax.set_ylabel("facts")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.8); ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0, 2])
    ax.hist(d["cos1"], bins=30, color=ORANGE, alpha=0.7, label="cos(cross, layer-1 contrib)")
    ax.hist(d["cos2"], bins=30, color=AQUA, alpha=0.7, label="cos(cross, layer-2 contrib)")
    ax.axvline(0, color=INK, lw=1)
    ax.set_title("(iii) does the composed term cancel a layer's write?\n(seed 0, one value per fact)", fontsize=9.5)
    ax.set_xlabel("cosine in logit space")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.8); ax.set_axisbelow(True)

    fig.suptitle(f"F13 — how 200 facts live in 2 bilinear blocks (H*={met['Hstar']} per block)",
                 color=INK, y=1.04)
    save_fig(fig, "F13_cross_layer")


STAGES = {"sweep": stage_sweep, "verify": stage_verify, "control": stage_control,
          "measure": stage_measure, "figure": stage_figure}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    if args.stage == "all":
        for name, fn in STAGES.items():
            fn()
    else:
        STAGES[args.stage]()
