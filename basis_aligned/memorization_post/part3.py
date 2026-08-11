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
        # --- informed pull-out: G ~= sum_k a_k (Cih h2_k)^T, recovery per fact
        Dm = h2 @ Cih                                  # (N, H) dictionary rows
        Acoef = G @ np.linalg.pinv(Dm)                 # (C, N)
        comp = Acoef.T * (Dm * h2).sum(1, keepdims=True)   # (N, C): a_k * <d_k, h2_k>
        out["pullout_recovery"] = int((comp.argmax(1) == y).sum())
        out["m0_median"] = float(np.median(m0))
        res[seed] = out
        if seed == 0:
            np.savez(os.path.join(FIG, "F13c_seed0_data.npz"),
                     drop_h2=(m0 - m1)[keep], drop_W=(m0 - m1w)[keep],
                     drop_inj=(m0 - m1i), m0=m0)
        print(f"  seed{seed}: rm-h2 flips {out['rm_h2']['retained_flips']}/1190 "
              f"(med margin drop {out['rm_h2']['median_retained_margin_drop']:.2f}), "
              f"rm-W flips {out['rm_W']['retained_flips']}/1190 "
              f"(med drop {out['rm_W']['median_retained_margin_drop']:.2f}); "
              f"inject {out['inject_h2']['injected_correct']}/10 ok, "
              f"flips {out['inject_h2']['retained_flips']}/1200; "
              f"pull-out recovery {out['pullout_recovery']}/1200; "
              f"median margin {out['m0_median']:.1f}")
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
    rec = [res[s]["pullout_recovery"] / 12 for s in SEEDS]  # percent
    ax.bar(np.arange(5), rec, 0.6, color=BLUE)
    ax.axhline(10, color=INK, lw=1, ls="--")
    ax.text(3.6, 11, "10% chance", fontsize=8, color=INK2)
    ax.axhline(47.5, color=ORANGE, lw=1.2, ls=":")
    ax.text(0.1, 49, "Part 2 (100 facts, 1 layer): 44-51%", fontsize=8, color=ORANGE)
    ax.set_xticks(np.arange(5)); ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=8)
    ax.set_ylabel("facts recovered (%)")
    ax.set_title("(iii) informed per-fact pull-out\n(dictionary = $C^{-1}$-weighted h2 keys)", fontsize=9.5)
    for a in fig.axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.grid(axis="y", color=GRIDC, lw=0.8); a.set_axisbelow(True)
    fig.suptitle("F13c — closed-form edits in the 2-layer model (last-block key frame; seed 0)",
                 color=INK, y=1.04)
    save_fig(fig, "F13c_two_layer_edits")


STAGES = {"sweep": stage_sweep, "verify": stage_verify, "control": stage_control,
          "measure": stage_measure, "figure": stage_figure,
          "capacity": stage_capacity, "edits": stage_edits}

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
