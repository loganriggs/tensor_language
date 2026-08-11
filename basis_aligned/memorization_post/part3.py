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
N_FACTS = 200   # overridden by --nfacts; sizing outcome documented in results.md
N_CLASSES = 10
SEEDS = [0, 1, 2, 3, 4]
STEPS = 15000
LR = 1e-2
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Sizing note (2026-08-11): at 200 facts EVERY config memorized (a single quadratic
# layer in 20 booleans has ~210 free monomials — 200 facts is below the expressivity
# cap at any H). The sizing knob is the fact count. H1 grid includes H=210 = the full
# quadratic span, so a 1-block failure is an EXPRESSIVITY limit, not parameter shortage.
H1_GRID = [40, 80, 120, 210]
H2_GRID = [12, 20, 40, 80]
SIZE_JSON = os.path.join(FIG, "part3_sizing.json")   # reset per --nfacts in __main__
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


def make_facts(n=None, rng_seed=300):
    n = n if n is not None else N_FACTS
    rng = np.random.default_rng(rng_seed)
    seen, Z = set(), []
    while len(Z) < n:
        z = tuple(int(v) for v in rng.integers(0, 2, N_BITS))
        if sum(z) > 0 and z not in seen:
            seen.add(z)
            Z.append(z)
    Z = np.array(Z, dtype=np.float64)
    y = rng.integers(0, N_CLASSES, n)
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


def train(H, n_blocks, seed, Z, y, freeze_zero=None, steps=None):
    """freeze_zero: block index whose D is fixed at zero (block disabled but present)."""
    steps = steps if steps is not None else STEPS
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
    # sizing criterion: smallest 2-block H with both sizing seeds at 100%, given that
    # the 1-block model fails even at the FULL quadratic span (H=210) — an
    # expressivity failure, not a parameter shortage
    one_best = max(max(v) for v in sz["one"].values())
    if one_best == 1.0:
        sys.exit("a 1-block model memorized everything — raise --nfacts (document!).")
    Hstar = next((H for H in H2_GRID if min(sz["two"][str(H)]) == 1.0), None)
    if Hstar is None:
        sys.exit("no H satisfies the sizing criterion — adjust N_FACTS/grids (document!).")
    print(f"chosen H* = {Hstar} per block")
    Z, y = make_facts()
    res = {"Hstar": Hstar, "n_facts": N_FACTS, "two": [], "one_same": [], "one_fullspan": []}
    for seed in SEEDS:
        ps, acc, loss = train(Hstar, 2, seed, Z, y)
        np.savez(os.path.join(MODELS, f"p3_two_N{N_FACTS}_H{Hstar}_s{seed}.npz"), **ps)
        res["two"].append(acc)
        print(f"  2-block H={Hstar} seed{seed}: acc {acc:.1%} loss {loss:.4f}")
    for tag, H in (("one_same", Hstar), ("one_fullspan", 210)):
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
    # controls run on the first 200 facts (memorizable by ONE live block at H=40,
    # per the N=200 sweep) — the metric gate does not need the final sizing
    Zf, yf = make_facts()
    Z, y = Zf[:200], yf[:200]
    Hc = 40
    print(f"  control: first 200 facts, H = {Hc} (1 block at H=40 memorized 200/200)")
    ok = True
    for dead in (1, 0):
        ps, acc, _ = train(Hc, 2, 0, Z, y, freeze_zero=dead)
        full, bins, _, _ = attribution_bins(ps, Z, y)
        mem = full.sum()
        dead_bin = bins["l2only" if dead == 1 else "l1only"][full].sum()
        frac = dead_bin / max(mem, 1)
        verdict = "PASS" if frac <= 0.05 else "FAIL"
        ok &= (frac <= 0.05)
        print(f"  block {dead} frozen at zero: memorized {mem}/{len(y)}; "
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
    met = {"Hstar": Hstar, "n_facts": N_FACTS, "seeds": {}}
    for seed in SEEDS:
        ps = dict(np.load(os.path.join(MODELS, f"p3_two_N{N_FACTS}_H{Hstar}_s{seed}.npz")))
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
    n_facts = met.get("n_facts", N_FACTS)
    names = ["linear", "l1only", "l2only", "both", "neither"]
    disp = ["linear\n(no blocks)", "layer 1\nalone", "layer 2\nalone",
            "either alone\n(redundant)", "needs both\n(composed)"]
    seed_bins = np.array([[met["seeds"][str(s)]["bins"][k] for k in names] for s in SEEDS])
    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(names))
    ax.bar(xs, seed_bins[0], 0.62, color=[MUTED, BLUE, ORANGE, AQUA, "#c23b3b"])
    for i in range(len(names)):
        lo, hi = seed_bins[:, i].min(), seed_bins[:, i].max()
        ax.plot([i, i], [lo, hi], color=INK, lw=1.2)
    ax.set_xticks(xs); ax.set_xticklabels(disp, fontsize=7.8)
    ax.set_ylabel(f"facts (of {n_facts})")
    ax.axhline(0.1 * n_facts, color=INK, lw=1, ls="--")
    ax.text(len(names) - 0.45, 0.1 * n_facts, " 10% chance", fontsize=8, color=INK2, va="bottom", ha="right")
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
                 f"degree-2 surrogate loses {lost}/{n_facts} facts", fontsize=9.5)
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

    fig.suptitle(f"F13 — how {n_facts} facts live in 2 bilinear blocks (H*={met['Hstar']} per block)",
                 color=INK, y=1.04)
    save_fig(fig, "F13_cross_layer")


# ============================================================================= capacity
def stage_capacity():
    """F13b: facts learned (y) vs width (x), 1-block vs 2-block, trained on a fixed
    pool of 4000 facts (single seed per Logan's display convention)."""
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    CAP_N = 4000
    grids = {1: [20, 40, 80, 120, 210, 300], 2: [10, 20, 40, 80, 120, 210]}
    Z, y = make_facts(CAP_N)
    out = {"pool": CAP_N, "one": {}, "two": {}}
    print("=" * 70)
    print(f"STAGE capacity: facts fit vs width, pool of {CAP_N} facts (seed 0, dev={DEV})")
    print("=" * 70)
    for nb, key in ((1, "one"), (2, "two")):
        for H in grids[nb]:
            _, acc, _ = train(H, nb, 0, Z, y, steps=25000)
            out[key][H] = acc
            print(f"  {nb}-block H={H:3d}: {acc:.1%} = {acc*CAP_N:.0f} facts")
    with open(os.path.join(FIG, "part3_capacity.json"), "w") as f:
        json.dump(out, f, indent=1)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for key, col, lbl, nb in (("one", ORANGE, "1 block", 1), ("two", BLUE, "2 blocks", 2)):
        Hs = sorted(int(h) for h in out[key])
        ax.plot(Hs, [out[key][h] * CAP_N for h in Hs], "o-", color=col, lw=2, ms=6, label=lbl)
    ax.axhline(CAP_N, color=MUTED, lw=1, ls=":")
    ax.text(11, CAP_N - 130, "pool size (4000)", fontsize=8, color=INK2)
    ax.axhline(0.1 * CAP_N, color=INK, lw=1, ls="--")
    ax.text(11, 0.1 * CAP_N + 40, "10% chance", fontsize=8, color=INK2)
    ax.axvline(210, color=MUTED, lw=1, ls=":")
    ax.text(214, 1500, "H=210 = full\nquadratic span", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_xticks([10, 20, 40, 80, 120, 210, 300])
    ax.set_xticklabels([10, 20, 40, 80, 120, 210, 300])
    ax.set_xlabel("width H (hidden units per block)")
    ax.set_ylabel("facts learned (of 4000)")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("F13b — the expressivity wall: one quadratic block plateaus, two compose past it\n"
                 "(trained on a fixed pool of 4000 random 20-bit facts; seed 0)", fontsize=10)
    save_fig(fig, "F13b_capacity_vs_width")


# ============================================================================= edits
def h2_frame(ps):
    """The 2-layer model is LINEAR in its last-block output map D2 (stored as D1):
    logits = W x1 + G h2, with G = W D2, x1 = z + B1(z), h2 = (L2 x1)*(R2 x1).
    Returns (x1, h2, base_logits_without_G_term)."""
    Z, y = make_facts()
    x1 = Z + block_out(ps, Z, 0)
    h2 = (x1 @ ps["L1"].T) * (x1 @ ps["R1"].T)
    base = x1 @ ps["W"].T
    return Z, y, x1, h2, base


def joint_kkt(K, Ci, T):
    """Least-C-norm Delta s.t. Delta @ K.T = T.T for constraint keys K (m, dim),
    metric C^-1 (Ci), targets T (m, C). Returns Delta (C, dim)."""
    M = K @ Ci                       # (m, dim)
    A = K @ Ci @ K.T                 # (m, m)
    return T.T @ np.linalg.solve(A, M)


def stage_edits():
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    print("=" * 70)
    print("STAGE edits: closed-form KKT removal / injection / pull-out in the 2-layer model")
    print("=" * 70)
    res = {}
    for seed in SEEDS:
        ps = dict(np.load(os.path.join(MODELS, f"p3_two_N{N_FACTS}_H40_s{seed}.npz")))
        Z, y, x1, h2, base = h2_frame(ps)
        G = ps["W"] @ ps["D1"]                     # (C, H)
        logits0 = base + h2 @ G.T
        assert np.abs(logits0 - np_fwd(ps, Z, 2)).max() < 1e-8
        m0 = margins(logits0, y)
        N, H = h2.shape
        ridge = 1e-8 * np.trace(h2.T @ h2) / H
        Ch = h2.T @ h2 + ridge * np.eye(H)
        Cih = np.linalg.inv(Ch)
        x2 = x1 + h2 @ ps["D1"].T
        Cx = x2.T @ x2 + 1e-8 * np.trace(x2.T @ x2) / x2.shape[1] * np.eye(x2.shape[1])
        Cix = np.linalg.inv(Cx)

        rng = np.random.default_rng(7)
        rm = rng.choice(N, 10, replace=False)
        keep = np.setdiff1d(np.arange(N), rm)

        out = {}
        # --- removal, h2 frame (edit G, i.e. D2): target = uniform logits at the 10 keys
        T_rm = logits0[rm].mean(1, keepdims=True) - logits0[rm]
        dG = joint_kkt(h2[rm], Cih, T_rm)
        logits_h = logits0 + h2 @ dG.T
        # exactness check via actual weights: D2' = D2 + pinv(W) dG
        ps_ed = dict(ps); ps_ed["D1"] = ps["D1"] + np.linalg.pinv(ps["W"]) @ dG
        assert np.abs((np_fwd(ps_ed, Z, 2)) - logits_h).max() < 1e-6
        m1 = margins(logits_h, y)
        out["rm_h2"] = {
            "forget_dev": float(np.abs(logits_h[rm] - logits_h[rm].mean(1, keepdims=True)).max()),
            "retained_flips": int((logits_h[keep].argmax(1) != y[keep]).sum()),
            "median_retained_margin_drop": float(np.median((m0 - m1)[keep])),
        }
        # --- removal, readout frame (edit W, keys x2 in R^20)
        dW = joint_kkt(x2[rm], Cix, T_rm)
        logits_w = logits0 + x2 @ dW.T
        m1w = margins(logits_w, y)
        out["rm_W"] = {
            "forget_dev": float(np.abs(logits_w[rm] - logits_w[rm].mean(1, keepdims=True)).max()),
            "retained_flips": int((logits_w[keep].argmax(1) != y[keep]).sum()),
            "median_retained_margin_drop": float(np.median((m0 - m1w)[keep])),
        }
        # --- injection of 10 NEW facts (h2 frame)
        Znew_all, ynew_all = make_facts(N + 200, rng_seed=300)   # same stream; take beyond N
        Znew, ynew = Znew_all[N:N + 10], ynew_all[N:N + 10]
        x1n = Znew + block_out(ps, Znew, 0)
        h2n = (x1n @ ps["L1"].T) * (x1n @ ps["R1"].T)
        ln = x1n @ ps["W"].T + h2n @ G.T
        T_in = np.zeros_like(ln)
        for i in range(10):
            others = np.delete(ln[i], ynew[i])
            T_in[i, ynew[i]] = (others.max() + 1.0) - ln[i, ynew[i]]
        dGi = joint_kkt(h2n, Cih, T_in)
        logits_inj = logits0 + h2 @ dGi.T
        ln_after = ln + h2n @ dGi.T
        m1i = margins(logits_inj, y)
        out["inject_h2"] = {
            "injected_correct": int((ln_after.argmax(1) == ynew).sum()),
            "retained_flips": int((logits_inj.argmax(1) != y).sum()),
            "median_retained_margin_drop": float(np.median(m0 - m1i)),
        }
        # --- informed pull-out: G ~= sum_k a_k (Cih h2_k)^T
        Dm = h2 @ Cih                                  # (N, H) dictionary rows
        Acoef = G @ np.linalg.pinv(Dm)                 # (C, N); a_k = column k
        # NAIVE metric (Part 2 F10 style: component classifies its own key) is DEGENERATE
        # here: with N >> H the dictionary is 30x overcomplete and a_k <d_k, h2_k> is just
        # a whitened copy of the model's own logits — it saturates at ~100% regardless of
        # whether per-fact components exist. Reported for the record, not as recovery.
        comp = Acoef.T * (Dm * h2).sum(1, keepdims=True)
        out["pullout_naive_saturated"] = int((comp.argmax(1) == y).sum())
        # MEANINGFUL metric: component DELETION — remove fact k's component from G and ask
        # (a) does fact k break, (b) how many other facts break. Clean pull-out = own fact
        # breaks and < 5% of others do.
        S = h2 @ Dm.T                                  # (N, N): h2_j . d_k
        own_broken, other_flip_frac = np.zeros(N, bool), np.zeros(N)
        a = Acoef.T                                    # (N, C)
        for k in range(N):
            Lk = logits0 - np.outer(S[:, k], a[k])
            pred = Lk.argmax(1)
            own_broken[k] = pred[k] != y[k]
            other_flip_frac[k] = (pred != y).sum() - (pred[k] != y[k])
        other_flip_frac /= (N - 1)
        out["pullout_del_own_broken"] = int(own_broken.sum())
        out["pullout_del_median_other_flips"] = float(np.median(other_flip_frac))
        out["pullout_clean"] = int((own_broken & (other_flip_frac < 0.05)).sum())
        out["m0_median"] = float(np.median(m0))
        res[seed] = out
        if seed == 0:
            np.savez(os.path.join(FIG, "F13c_seed0_data.npz"),
                     drop_h2=(m0 - m1)[keep], drop_W=(m0 - m1w)[keep],
                     drop_inj=(m0 - m1i), m0=m0,
                     own_broken=own_broken, other_flip_frac=other_flip_frac)
        print(f"  seed{seed}: rm-h2 flips {out['rm_h2']['retained_flips']}/1190 "
              f"(med margin drop {out['rm_h2']['median_retained_margin_drop']:.2f}), "
              f"rm-W flips {out['rm_W']['retained_flips']}/1190 "
              f"(med drop {out['rm_W']['median_retained_margin_drop']:.2f}); "
              f"inject {out['inject_h2']['injected_correct']}/10 ok, "
              f"flips {out['inject_h2']['retained_flips']}/1200; "
              f"pull-out: naive {out['pullout_naive_saturated']}/1200 (degenerate), "
              f"deletion breaks own {out['pullout_del_own_broken']}/1200, "
              f"med other-flips {out['pullout_del_median_other_flips']:.1%}, "
              f"CLEAN {out['pullout_clean']}/1200; median margin {out['m0_median']:.1f}")
    with open(os.path.join(FIG, "part3_edits.json"), "w") as f:
        json.dump(res, f, indent=1)

    # F13c figure
    d = np.load(os.path.join(FIG, "F13c_seed0_data.npz"))
    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(min(d["drop_h2"].min(), d["drop_W"].min()),
                       max(d["drop_h2"].max(), d["drop_W"].max()), 50)
    ax.hist(d["drop_W"], bins=bins, color=ORANGE, alpha=0.7, label="readout frame (20-dim)")
    ax.hist(d["drop_h2"], bins=bins, color=BLUE, alpha=0.7, label="h2 key frame (40-dim)")
    ax.axvline(0, color=INK, lw=1)
    f_h2 = res[0]["rm_h2"]["retained_flips"]; f_W = res[0]["rm_W"]["retained_flips"]
    ax.set_title(f"(i) unlearn 10 of 1200: retained-fact margin drop\n"
                 f"flips: h2 frame {f_h2}/1190, readout frame {f_W}/1190", fontsize=9.5)
    ax.set_xlabel("margin before $-$ after"); ax.set_ylabel("retained facts")
    ax.legend(frameon=False, fontsize=8)
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(d["drop_inj"], bins=50, color=AQUA, alpha=0.85)
    ax.axvline(0, color=INK, lw=1)
    inj = res[0]["inject_h2"]
    ax.set_title(f"(ii) inject 10 NEW facts (h2 frame): {inj['injected_correct']}/10 land,\n"
                 f"{inj['retained_flips']}/1200 stored facts flip", fontsize=9.5)
    ax.set_xlabel("stored-fact margin before $-$ after")
    ax = fig.add_subplot(gs[0, 2])
    ax.scatter(d["other_flip_frac"] * 100, np.where(d["own_broken"], 1, 0)
               + (np.random.default_rng(0).uniform(-0.12, 0.12, len(d["own_broken"]))),
               s=5, alpha=0.25, color=BLUE)
    ax.axvline(5, color=INK, lw=1, ls="--")
    ax.text(5.5, 0.5, "5% collateral bar", fontsize=8, color=INK2, rotation=90, va="center")
    clean = res[0]["pullout_clean"]
    ax.set_yticks([0, 1]); ax.set_yticklabels(["own fact\nsurvives", "own fact\nbreaks"], fontsize=8)
    ax.set_xlabel("other facts flipped when deleting this fact's component (%)")
    ax.set_title(f"(iii) per-fact component DELETION (seed 0):\n"
                 f"clean pull-outs (own breaks, <5% collateral): {clean}/1200", fontsize=9.5)
    for a in fig.axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", color=GRIDC, lw=0.8); a.set_axisbelow(True)
    fig.suptitle("F13c — closed-form edits in the 2-layer model (last-block key frame; seed 0)",
                 color=INK, y=1.04)
    save_fig(fig, "F13c_two_layer_edits")


# ============================================================================= edits2
def edit_metrics(logits, y, rm, keep, m0):
    m = margins(logits, y)
    pred = logits.argmax(1)
    return {
        "forget": int((pred[rm] != y[rm]).sum()),
        "forget_dev": float(np.abs(logits[rm] - logits[rm].mean(1, keepdims=True)).max()),
        "retained_flips": int((pred[keep] != y[keep]).sum()),
        "median_retained_margin_drop": float(np.median((m0 - m)[keep])),
    }


def stage_edits2():
    """Logan's question: is the ~45% KKT collateral fundamental or the wrong objective?
    B: weighted LS refit of the joint linear readout [W, G] over [x1, h2].
    C: hinge LP over Delta-G — exact removal equalities + per-retained-fact margin
       constraints. Total violation 0 <=> a zero-collateral last-layer edit EXISTS.
    D: one exact repair round in the R2 frame (logits linear in R2; h2 invariant to the
       D2 edit) on top of the LP edit, holding removal equalities.
    Oracle: retrain from scratch on the 1190 retained facts (existence baseline, seed 0).
    """
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    import scipy.sparse as sp
    from scipy.optimize import linprog
    print("=" * 70)
    print("STAGE edits2: LS refit / margin LP / cross-layer repair / oracle retrain")
    print("=" * 70)
    EPS = 0.5
    res = {}
    for seed in SEEDS:
        ps = dict(np.load(os.path.join(MODELS, f"p3_two_N{N_FACTS}_H40_s{seed}.npz")))
        Z, y, x1, h2, base = h2_frame(ps)
        G = ps["W"] @ ps["D1"]
        logits0 = base + h2 @ G.T
        m0 = margins(logits0, y)
        N, H = h2.shape
        C = N_CLASSES
        rng = np.random.default_rng(7)
        rm = rng.choice(N, 10, replace=False)
        keep = np.setdiff1d(np.arange(N), rm)
        out = {}

        # ---- B: weighted LS refit of the joint readout [W | G] over M = [x1 | h2]
        M = np.hstack([x1, h2])                       # (N, d+H)
        T = logits0.copy()
        T[rm] = logits0[rm].mean(1, keepdims=True)
        w = np.ones(N); w[rm] = 100.0
        sw = np.sqrt(w)[:, None]
        Theta, *_ = np.linalg.lstsq(M * sw, T * sw, rcond=None)
        out["B_ls_refit"] = edit_metrics(M @ Theta, y, rm, keep, m0)

        # ---- C: hinge LP over Delta-G
        nkeep = len(keep)
        rows = nkeep * (C - 1)
        # retention block: row (k, c!=y_k): -(df_y - df_c) - s <= m_kc - EPS
        data, ri, ci = [], [], []
        b_ub = np.empty(rows)
        r = 0
        for kk, k in enumerate(keep):
            hk = h2[k]
            for c in range(C):
                if c == y[k]:
                    continue
                ri.extend([r] * (2 * H))
                ci.extend(range(y[k] * H, y[k] * H + H))
                data.extend(-hk)
                ci.extend(range(c * H, c * H + H))
                data.extend(hk)
                b_ub[r] = (logits0[k, y[k]] - logits0[k, c]) - EPS
                r += 1
        A_ret = sp.csr_matrix((data, (ri, ci)), shape=(rows, C * H))
        A_ub = sp.hstack([A_ret, -sp.eye(rows)], format="csr")
        # removal equalities: df_c - df_{c+1} = -(f_c - f_{c+1}) at removed keys
        data, ri, ci = [], [], []
        b_eq = np.empty(10 * (C - 1))
        r = 0
        for k in rm:
            hk = h2[k]
            for c in range(C - 1):
                ri.extend([r] * (2 * H))
                ci.extend(range(c * H, c * H + H))
                data.extend(hk)
                ci.extend(range((c + 1) * H, (c + 1) * H + H))
                data.extend(-hk)
                b_eq[r] = -(logits0[k, c] - logits0[k, c + 1])
                r += 1
        A_eq = sp.hstack([sp.csr_matrix((data, (ri, ci)), shape=(r, C * H)),
                          sp.csr_matrix((r, rows))], format="csr")
        cobj = np.concatenate([np.zeros(C * H), np.ones(rows)])
        lp = linprog(cobj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(None, None)] * (C * H) + [(0, None)] * rows,
                     method="highs")
        assert lp.status == 0, lp.message
        dG = lp.x[:C * H].reshape(C, H)
        s = lp.x[C * H:]
        viol_rows = int((s > 1e-6).sum())
        viol_facts = int(len({keep[i // (C - 1)] for i in np.where(s > 1e-6)[0]}))
        logitsC = logits0 + h2 @ dG.T
        out["C_margin_lp"] = edit_metrics(logitsC, y, rm, keep, m0)
        out["C_margin_lp"].update({"total_violation": float(s.sum()),
                                   "violated_facts": viol_facts,
                                   "feasible_zero_collateral": bool(s.sum() < 1e-6)})

        # ---- D: one exact repair round in the R2 frame on the LP-edited model
        # Edited model: D2' = D2 + pinv(W) dG; h2 map unchanged; logits_C exact.
        G2 = G + dG
        a2 = x1 @ ps["L1"].T                          # (N, H): (L2 x1)
        d_in = x1.shape[1]
        nv = H * d_in
        # coeff for row (k, c-pair) and var (h, j): G2[c,h] * a2[k,h] * x1[k,j]
        def r2_row(k, cpos, cneg):
            v = (G2[cpos][:, None] * a2[k][:, None] * x1[k][None, :]
                 - G2[cneg][:, None] * a2[k][:, None] * x1[k][None, :])
            return v.ravel()
        A_ret2 = np.empty((rows, nv))
        b_ub2 = np.empty(rows)
        mC = logitsC
        r = 0
        for kk, k in enumerate(keep):
            for c in range(C):
                if c == y[k]:
                    continue
                A_ret2[r] = -r2_row(k, y[k], c)
                b_ub2[r] = (mC[k, y[k]] - mC[k, c]) - EPS
                r += 1
        A_ub2 = sp.hstack([sp.csr_matrix(A_ret2), -sp.eye(rows)], format="csr")
        A_eq2 = np.empty((10 * (C - 1), nv))
        b_eq2 = np.zeros(10 * (C - 1))
        r = 0
        for k in rm:
            for c in range(C - 1):
                A_eq2[r] = r2_row(k, c, c + 1)
                b_eq2[r] = -(mC[k, c] - mC[k, c + 1])
                r += 1
        A_eq2 = sp.hstack([sp.csr_matrix(A_eq2), sp.csr_matrix((r, rows))], format="csr")
        cobj2 = np.concatenate([np.zeros(nv), np.ones(rows)])
        lp2 = linprog(cobj2, A_ub=A_ub2, b_ub=b_ub2, A_eq=A_eq2, b_eq=b_eq2,
                      bounds=[(None, None)] * nv + [(0, None)] * rows,
                      method="highs")
        assert lp2.status == 0, lp2.message
        dR2 = lp2.x[:nv].reshape(H, d_in)
        # exact evaluation with edited weights
        ps_d = dict(ps)
        ps_d["D1"] = ps["D1"] + np.linalg.pinv(ps["W"]) @ dG
        ps_d["R1"] = ps["R1"] + dR2
        logitsD = np_fwd(ps_d, Z, 2)
        out["D_r2_repair"] = edit_metrics(logitsD, y, rm, keep, m0)
        out["D_r2_repair"]["total_violation"] = float(lp2.x[nv:].sum())

        res[seed] = out
        print(f"  seed{seed}:")
        for tag in ("B_ls_refit", "C_margin_lp", "D_r2_repair"):
            o = out[tag]
            extra = ""
            if "feasible_zero_collateral" in o:
                extra = (f", LP {'FEASIBLE' if o['feasible_zero_collateral'] else 'infeasible'}"
                         f" (violation {o['total_violation']:.1f}, {o['violated_facts']} facts)")
            if tag == "D_r2_repair":
                extra = f", residual violation {o['total_violation']:.1f}"
            print(f"    {tag}: forget {o['forget']}/10 (dev {o['forget_dev']:.2e}), "
                  f"retained flips {o['retained_flips']}/{len(keep)}, "
                  f"med margin drop {o['median_retained_margin_drop']:.2f}{extra}")

    # oracle retrain (seed 0 only)
    Z, yy = make_facts()
    rng = np.random.default_rng(7)
    rm = rng.choice(len(yy), 10, replace=False)
    keep = np.setdiff1d(np.arange(len(yy)), rm)
    ps_o, acc, _ = train(40, 2, 0, Z[keep], yy[keep], steps=25000)
    pred_rm = np_fwd(ps_o, Z[rm], 2).argmax(1)
    oracle = {"retained_acc": acc, "removed_correct": int((pred_rm == yy[rm]).sum())}
    res["oracle_seed0"] = oracle
    print(f"  oracle retrain (seed0): retained {acc:.1%}, removed still-correct "
          f"{oracle['removed_correct']}/10 (chance ~1)")
    with open(os.path.join(FIG, "part3_edits2.json"), "w") as f:
        json.dump(res, f, indent=1)


# ============================================================================= edits3
def stage_edits3():
    """P14: alternate exact single-frame hinge LPs (G -> R2 -> L2 -> ...), removal
    equalities re-imposed every round; seed 0. Each frame is exactly linear in the
    varied parameter with the others fixed, so every round is exact (no Taylor)."""
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    import scipy.sparse as sp
    from scipy.optimize import linprog
    print("=" * 70)
    print("STAGE edits3: alternating-frame hinge LPs, seed 0")
    print("=" * 70)
    EPS = 0.5
    ps = dict(np.load(os.path.join(MODELS, f"p3_two_N{N_FACTS}_H40_s0.npz")))
    Z, y, x1, _, _ = h2_frame(ps)
    N = len(y); C = N_CLASSES; H = ps["L1"].shape[0]; d = x1.shape[1]
    rng = np.random.default_rng(7)
    rm = rng.choice(N, 10, replace=False)
    keep = np.setdiff1d(np.arange(N), rm)
    m0 = margins(np_fwd(ps, Z, 2), y)
    rows = len(keep) * (C - 1)

    def frame_rows(ps, frame):
        """Return (nv, rowfn) where rowfn(k, cpos, cneg) gives the Delta-logit-difference
        coefficient row for key k. x1 fixed (block 0 untouched throughout)."""
        G2 = ps["W"] @ ps["D1"]
        if frame == "G":
            h2 = (x1 @ ps["L1"].T) * (x1 @ ps["R1"].T)
            def rowfn(k, cp, cn):
                v = np.zeros((C, H))
                v[cp] = h2[k]; v[cn] -= h2[k]
                return v.ravel()
            return C * H, rowfn
        fac = x1 @ (ps["L1"].T if frame == "R2" else ps["R1"].T)   # (N, H)
        def rowfn(k, cp, cn):
            base = fac[k][:, None] * x1[k][None, :]
            return ((G2[cp] - G2[cn])[:, None] * base).ravel()
        return H * d, rowfn

    def apply(ps, frame, dx):
        ps = dict(ps)
        if frame == "G":
            ps["D1"] = ps["D1"] + np.linalg.pinv(ps["W"]) @ dx.reshape(C, H)
        elif frame == "R2":
            ps["R1"] = ps["R1"] + dx.reshape(H, d)
        else:
            ps["L1"] = ps["L1"] + dx.reshape(H, d)
        return ps

    hist = []
    for rnd, frame in enumerate(["G", "R2", "L2", "G", "R2", "L2", "G"]):
        logits = np_fwd(ps, Z, 2)
        nv, rowfn = frame_rows(ps, frame)
        A_ret = np.empty((rows, nv)); b_ub = np.empty(rows)
        r = 0
        for k in keep:
            for c in range(C):
                if c == y[k]:
                    continue
                A_ret[r] = -rowfn(k, y[k], c)
                b_ub[r] = (logits[k, y[k]] - logits[k, c]) - EPS
                r += 1
        A_eqd = np.empty((10 * (C - 1), nv)); b_eq = np.empty(10 * (C - 1))
        r = 0
        for k in rm:
            for c in range(C - 1):
                A_eqd[r] = rowfn(k, c, c + 1)
                b_eq[r] = -(logits[k, c] - logits[k, c + 1])
                r += 1
        A_ub = sp.hstack([sp.csr_matrix(A_ret), -sp.eye(rows)], format="csr")
        A_eq = sp.hstack([sp.csr_matrix(A_eqd), sp.csr_matrix((r, rows))], format="csr")
        lp = linprog(np.concatenate([np.zeros(nv), np.ones(rows)]),
                     A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(None, None)] * nv + [(0, None)] * rows, method="highs")
        assert lp.status == 0, lp.message
        ps = apply(ps, frame, lp.x[:nv])
        logits_a = np_fwd(ps, Z, 2)
        flips = int((logits_a[keep].argmax(1) != y[keep]).sum())
        fdev = float(np.abs(logits_a[rm] - logits_a[rm].mean(1, keepdims=True)).max())
        hist.append({"round": rnd, "frame": frame, "violation": float(lp.x[nv:].sum()),
                     "retained_flips": flips, "forget_dev": fdev})
        print(f"  round {rnd} [{frame}]: violation {lp.x[nv:].sum():9.1f}, "
              f"retained flips {flips}/1190, forget dev {fdev:.1e}")
    with open(os.path.join(FIG, "part3_edits3.json"), "w") as f:
        json.dump(hist, f, indent=1)
    edits3_figure(hist)


def edits3_figure(hist):
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    xs = [h["round"] for h in hist]
    fl = [h["retained_flips"] for h in hist]
    ax.plot(xs, fl, "o-", color=BLUE, lw=2, ms=7)
    for h in hist:
        ax.annotate(h["frame"], (h["round"], h["retained_flips"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=INK2)
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhline(522, color=ORANGE, lw=1.2, ls="--")
    ax.text(3.4, 540, "certified floor for ANY single last-layer edit (~550 broken)",
            fontsize=8, color=ORANGE)
    ax.set_xlabel("round (frame edited: G = last-block output map, R2/L2 = last-block factors)")
    ax.set_ylabel("retained facts broken (of 1190)")
    ax.set_title("F13d — certified impossible in one frame, exactly solved in six:\n"
                 "alternating single-frame margin LPs reach zero-collateral removal (seed 0)",
                 fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    save_fig(fig, "F13d_alternating_frames")


# ============================================================================= edits4
def alternate_lp(ps, Z, y, rm, keep, frames, eps=0.5):
    """Re-run the exact alternating-frame hinge LPs (stage edits3 logic, refactored so
    the edited model and per-round deltas can be captured). Returns (ps, history)."""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    N = len(y); C = N_CLASSES; H = ps["L1"].shape[0]; d = Z.shape[1]
    x1 = Z + block_out(ps, Z, 0)
    rows = len(keep) * (C - 1)
    hist = []
    for rnd, frame in enumerate(frames):
        logits = np_fwd(ps, Z, 2)
        G2 = ps["W"] @ ps["D1"]
        if frame == "G":
            h2 = (x1 @ ps["L1"].T) * (x1 @ ps["R1"].T)
            nv = C * H
            def rowfn(k, cp, cn):
                v = np.zeros((C, H)); v[cp] = h2[k]; v[cn] -= h2[k]
                return v.ravel()
        else:
            fac = x1 @ (ps["L1"].T if frame == "R2" else ps["R1"].T)
            nv = H * d
            def rowfn(k, cp, cn):
                return ((G2[cp] - G2[cn])[:, None]
                        * (fac[k][:, None] * x1[k][None, :])).ravel()
        A_ret = np.empty((rows, nv)); b_ub = np.empty(rows)
        r = 0
        for k in keep:
            for c in range(C):
                if c == y[k]:
                    continue
                A_ret[r] = -rowfn(k, y[k], c)
                b_ub[r] = (logits[k, y[k]] - logits[k, c]) - eps
                r += 1
        A_eqd = np.empty((10 * (C - 1), nv)); b_eq = np.empty(10 * (C - 1))
        r = 0
        for k in rm:
            for c in range(C - 1):
                A_eqd[r] = rowfn(k, c, c + 1)
                b_eq[r] = -(logits[k, c] - logits[k, c + 1])
                r += 1
        A_ub = sp.hstack([sp.csr_matrix(A_ret), -sp.eye(rows)], format="csr")
        A_eq = sp.hstack([sp.csr_matrix(A_eqd), sp.csr_matrix((r, rows))], format="csr")
        lp = linprog(np.concatenate([np.zeros(nv), np.ones(rows)]),
                     A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=[(None, None)] * nv + [(0, None)] * rows, method="highs")
        assert lp.status == 0, lp.message
        dx = lp.x[:nv]
        ps = dict(ps)
        if frame == "G":
            dmat = np.linalg.pinv(ps["W"]) @ dx.reshape(C, H)
            ps["D1"] = ps["D1"] + dmat
            base = np.linalg.norm(ps["D1"])
        elif frame == "R2":
            dmat = dx.reshape(H, d); ps["R1"] = ps["R1"] + dmat
            base = np.linalg.norm(ps["R1"])
        else:
            dmat = dx.reshape(H, d); ps["L1"] = ps["L1"] + dmat
            base = np.linalg.norm(ps["L1"])
        logits_a = np_fwd(ps, Z, 2)
        flips = int((logits_a[keep].argmax(1) != y[keep]).sum())
        hist.append({"round": rnd, "frame": frame, "violation": float(lp.x[nv:].sum()),
                     "flips": flips, "dnorm": float(np.linalg.norm(dmat)),
                     "relnorm": float(np.linalg.norm(dmat) / base)})
        if lp.x[nv:].sum() < 1e-6 and flips == 0:
            break
    return ps, hist


def finetune(ps, Z, y, steps_max, lr=1e-2, ft_seed=0, check_idx=None):
    """Full-batch AdamW fine-tune; returns steps until check_idx facts are all correct
    (and final retained accuracy), or (steps_max, ...) if never."""
    ps_t = {k: torch.tensor(v, device=DEV).requires_grad_(True) for k, v in ps.items()}
    torch.manual_seed(10_000 + ft_seed)
    Zt = torch.tensor(Z, device=DEV); yt = torch.tensor(y, device=DEV)
    opt = torch.optim.AdamW(list(ps_t.values()), lr=lr, weight_decay=0.0)
    hit = None
    for step in range(1, steps_max + 1):
        logits = fwd(ps_t, Zt, 2)
        loss = torch.nn.functional.cross_entropy(logits, yt)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5 == 0 or step == steps_max:
            with torch.no_grad():
                pred = fwd(ps_t, Zt, 2).argmax(1)
            if hit is None and bool((pred[check_idx] == yt[check_idx]).all()):
                hit = step
                break
    with torch.no_grad():
        acc = float((fwd(ps_t, Zt, 2).argmax(1) == yt).float().mean())
    return (hit if hit is not None else steps_max), acc


def stage_edits4():
    """Masking-diagnosis battery on the LP-edited model (registered P17-P20)."""
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    print("=" * 70)
    print("STAGE edits4: relearn speed / perturbation resurrection / lens / edit cost")
    print("=" * 70)
    ps0 = dict(np.load(os.path.join(MODELS, f"p3_two_N{N_FACTS}_H40_s0.npz")))
    Z, y = make_facts()
    rng = np.random.default_rng(7)
    rm = rng.choice(len(y), 10, replace=False)
    keep = np.setdiff1d(np.arange(len(y)), rm)
    res = {}

    # LP-alternation edit (regenerate + save) with per-round norms (P19)
    ps_lp, hist = alternate_lp(ps0, Z, y, rm, keep,
                               ["G", "R2", "L2", "G", "R2", "L2", "G"])
    np.savez(os.path.join(MODELS, "p3_two_N1200_H40_s0_LPedited.npz"), **ps_lp)
    final = hist[-1]
    print(f"  alternation regenerated: {len(hist)} rounds, final flips {final['flips']}, "
          f"violation {final['violation']:.2e}")
    assert final["flips"] == 0 and final["violation"] < 1e-6
    # KKT single-edit norm for comparison (edits2 A-method dG)
    x1 = Z + block_out(ps0, Z, 0)
    h2 = (x1 @ ps0["L1"].T) * (x1 @ ps0["R1"].T)
    G = ps0["W"] @ ps0["D1"]
    logits0 = np_fwd(ps0, Z, 2)
    H = h2.shape[1]
    ridge = 1e-8 * np.trace(h2.T @ h2) / H
    Cih = np.linalg.inv(h2.T @ h2 + ridge * np.eye(H))
    T_rm = logits0[rm].mean(1, keepdims=True) - logits0[rm]
    dG_kkt = joint_kkt(h2[rm], Cih, T_rm)
    kkt_norm = float(np.linalg.norm(np.linalg.pinv(ps0["W"]) @ dG_kkt))
    alt_total = float(sum(h["dnorm"] for h in hist))
    res["cost"] = {"per_round": hist, "alternation_total_dnorm": alt_total,
                   "kkt_dnorm": kkt_norm, "ratio": alt_total / kkt_norm}
    print(f"  P19 cost: alternation total ||d|| {alt_total:.2f} vs KKT ||d|| {kkt_norm:.2f} "
          f"-> ratio {alt_total/kkt_norm:.1f}x")

    # P20 lens: block-1-only readout is IDENTICAL pre/post (W, block 1 untouched)
    l1_pre = np_fwd(ps0, Z, 2, use=[1, 0]).argmax(1)
    l1_post = np_fwd(ps_lp, Z, 2, use=[1, 0]).argmax(1)
    assert (l1_pre == l1_post).all()
    res["lens"] = {"identical": True,
                   "removed_decodable_from_block1": int((l1_pre[rm] == y[rm]).sum())}
    print(f"  P20 lens: block-1 readout identical pre/post (verified); removed facts "
          f"decodable from block 1 alone: {res['lens']['removed_decodable_from_block1']}/10")

    # oracle: retrain from scratch on the 1190
    ps_or, acc_or, _ = train(40, 2, 0, Z[keep], y[keep], steps=25000)
    print(f"  oracle retrained on 1190: acc {acc_or:.1%}")

    # P18 perturbation resurrection
    noise_rng = np.random.default_rng(1234)
    sigmas = [0.002, 0.005, 0.01, 0.02, 0.05]
    res["perturb"] = []
    for sig in sigmas:
        row = {"sigma": sig}
        for tag, psx in (("lp", ps_lp), ("oracle", ps_or)):
            rev, brk = [], []
            for t in range(20):
                psn = {k: v + noise_rng.normal(0, sig * np.std(v), v.shape)
                       for k, v in psx.items()}
                pred = np_fwd(psn, Z, 2).argmax(1)
                rev.append((pred[rm] == y[rm]).mean())
                brk.append((pred[keep] != y[keep]).mean())
            row[tag + "_revert"] = float(np.mean(rev))
            row[tag + "_retained_broken"] = float(np.mean(brk))
        res["perturb"].append(row)
        print(f"  P18 sigma {sig}: LP revert {row['lp_revert']:.2f} "
              f"(retained broken {row['lp_retained_broken']:.2%}); "
              f"oracle revert {row['oracle_revert']:.2f} "
              f"(broken {row['oracle_retained_broken']:.2%})")

    # P17 relearn speed (3 finetune seeds each)
    res["relearn"] = {"lp": [], "oracle": []}
    for fs in range(3):
        s_lp, a_lp = finetune(ps_lp, Z, y, 3000, ft_seed=fs, check_idx=rm)
        s_or, a_or = finetune(ps_or, Z, y, 3000, ft_seed=fs, check_idx=rm)
        res["relearn"]["lp"].append({"steps": s_lp, "final_acc": a_lp})
        res["relearn"]["oracle"].append({"steps": s_or, "final_acc": a_or})
        print(f"  P17 ft-seed {fs}: LP-edited relearns 10/10 in {s_lp} steps "
              f"(final acc {a_lp:.1%}); oracle learns them in {s_or} steps "
              f"(final acc {a_or:.1%})")
    with open(os.path.join(FIG, "part3_edits4.json"), "w") as f:
        json.dump(res, f, indent=1)


def stage_edits4b():
    """P21: margin-floor robustness fix — alternation at eps = 10, then noise test."""
    require_committed(os.path.join(PRED, "part3_predictions.md"), "part3")
    print("=" * 70)
    print("STAGE edits4b: alternation with retention floor eps = 10")
    print("=" * 70)
    ps0 = dict(np.load(os.path.join(MODELS, f"p3_two_N{N_FACTS}_H40_s0.npz")))
    Z, y = make_facts()
    rng = np.random.default_rng(7)
    rm = rng.choice(len(y), 10, replace=False)
    keep = np.setdiff1d(np.arange(len(y)), rm)
    ps_lp, hist = alternate_lp(ps0, Z, y, rm, keep,
                               ["G", "R2", "L2", "G", "R2", "L2", "G"], eps=10.0)
    for h in hist:
        print(f"  round {h['round']} [{h['frame']}]: violation {h['violation']:9.1f}, "
              f"flips {h['flips']}, ||d|| {h['dnorm']:.2f}")
    logits = np_fwd(ps_lp, Z, 2)
    m = margins(logits, y)
    print(f"  final: flips {int((logits[keep].argmax(1) != y[keep]).sum())}, "
          f"retained margin min {m[keep].min():.2f} median {np.median(m[keep]):.2f}, "
          f"total ||d|| {sum(h['dnorm'] for h in hist):.2f}")
    noise_rng = np.random.default_rng(1234)
    out = {"rounds": hist}
    for sig in (0.002, 0.005, 0.01, 0.02):
        rev, brk = [], []
        for t in range(20):
            psn = {k: v + noise_rng.normal(0, sig * np.std(v), v.shape)
                   for k, v in ps_lp.items()}
            pred = np_fwd(psn, Z, 2).argmax(1)
            rev.append((pred[rm] == y[rm]).mean())
            brk.append((pred[keep] != y[keep]).mean())
        out[f"sigma_{sig}"] = {"revert": float(np.mean(rev)),
                               "retained_broken": float(np.mean(brk))}
        print(f"  sigma {sig}: revert {np.mean(rev):.2f}, retained broken {np.mean(brk):.2%}")
    with open(os.path.join(FIG, "part3_edits4b.json"), "w") as f:
        json.dump(out, f, indent=1)


STAGES = {"sweep": stage_sweep, "verify": stage_verify, "control": stage_control,
          "measure": stage_measure, "figure": stage_figure,
          "capacity": stage_capacity, "edits": stage_edits, "edits2": stage_edits2,
          "edits3": stage_edits3, "edits4": stage_edits4, "edits4b": stage_edits4b}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    ap.add_argument("--nfacts", type=int, default=400)
    ap.add_argument("--h1grid", type=str, default="")
    ap.add_argument("--h2grid", type=str, default="")
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()
    N_FACTS = args.nfacts
    if args.h1grid:
        H1_GRID = [int(v) for v in args.h1grid.split(",")]
    if args.h2grid:
        H2_GRID = [int(v) for v in args.h2grid.split(",")]
    if args.steps:
        STEPS = args.steps
    SIZE_JSON = os.path.join(FIG, f"part3_sizing_N{N_FACTS}.json")
    MET_JSON = os.path.join(FIG, f"part3_metrics_N{N_FACTS}.json")
    if args.stage == "all":
        for name, fn in STAGES.items():
            fn()
    else:
        STAGES[args.stage]()
