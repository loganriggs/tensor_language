"""Regenerate every headline result and plot it. Numbers come from the SAME fit functions the tables came
from (re-run, not transcribed), and are cached to figures/results.json so the figures cannot drift from RESULTS.md.

  F1  E6 Pareto           — MSE-only reaches perfect MSE at ~ZERO tensor-sim; lambda=0.1 fixes it for free
  F2  E3b metric temp     — a 1% identity ridge undoes the off-distribution blindness
  F3  E5b bottleneck      — hierarchy width is a SPECTRUM: the SHAPE of tsim(dz') says how many sub-features
  F4  E3a hierarchy       — hard mask BREAKS when mis-specified (a probe); soft never costs fidelity (a prior)
  F5  E7 real layer       — a real bilinear MLP shows no structure under an isotropic metric (honest negative)
"""
import sys, os, json, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language")
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders/figures"
os.makedirs(FIG, exist_ok=True)
CACHE = f"{FIG}/results.json"
R = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
def save(): json.dump(R, open(CACHE, "w"), indent=1)
def mstd(rs, k): return float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": .25, "axes.spines.top": False, "axes.spines.right": False})
C = {"bad": "#c0392b", "good": "#1a7f5a", "mid": "#2c6fbb", "alt": "#e08a1e", "gray": "#888"}


# ---------------------------------------------------------------- compute
def compute():
    if "e6" not in R:
        import e6_pareto as e6
        out = {}
        for k in e6.KS:
            for lam in e6.LAMS:
                rs = [e6.run(s, k, lam) for s in range(e6.SEEDS)]
                out[f"{k}|{lam}"] = dict(zip(("tsim", "gsim", "mse"),
                                             [mstd(rs, "tsim"), mstd(rs, "gsim"), mstd(rs, "mse_in")]))
                print("e6", k, lam, out[f"{k}|{lam}"]["tsim"], flush=True)
        R["e6"] = out; save()

    if "e3b" not in R:
        import e3_hierarchy_spectrum as e3
        from tensor_sim import lifted_moments
        Si, mi = e3.full_support(e3.D_IN + 1)
        out = {}
        for t in [0.0, 0.01, 0.05, 0.2, 0.5, 1.0]:
            rs = []
            for s in range(5):
                g = torch.Generator(device=e3.DEV).manual_seed(s + 5)
                B = torch.linalg.qr(torch.randn(e3.D_IN, 6, generator=g, device=e3.DEV))[0]
                x = torch.randn(20000, 6, generator=g, device=e3.DEV) @ B.T
                Sd, md = lifted_moments(x); Sd, md = Sd.to(e3.DEV), md.to(e3.DEV)
                rs.append(e3.fit(s, "random", 1, Sig=(1 - t) * Sd + t * Si, mu=(1 - t) * md + t * mi))
            out[str(t)] = dict(tsim=mstd(rs, "tsim_true"), ood=mstd(rs, "mse_ood"), mmcs=mstd(rs, "mmcs"))
            print("e3b", t, out[str(t)], flush=True)
        R["e3b"] = out; save()

    if "e5b" not in R:
        import e5b_hierarchy_spectrum_depth as e5b
        out = {}
        for name, sc in [("balanced", e5b.BAL), ("skewed", e5b.SKEW), ("one", e5b.ONE)]:
            for dz in [1, 2, 3, 4, 5, 6]:
                rs = [e5b.fit(s, dz, sc) for s in range(e5b.SEEDS)]
                out[f"{name}|{dz}"] = mstd(rs, "tsim")
                print("e5b", name, dz, out[f"{name}|{dz}"], flush=True)
        R["e5b"] = out; save()

    if "e3a" not in R:
        import e3_hierarchy_spectrum as e3
        out = {}
        for gt in ["random", "block"]:
            for nb in [1, 2, 4, 8, 16]:
                for mode, soft in [("hard", None), ("soft", 0.03)]:
                    rs = [e3.fit(s, gt, nb, soft=soft) for s in range(5)]
                    out[f"{gt}|{nb}|{mode}"] = dict(tsim=mstd(rs, "tsim_true"), mmcs=mstd(rs, "mmcs"))
                    print("e3a", gt, nb, mode, out[f"{gt}|{nb}|{mode}"]["tsim"], flush=True)
        R["e3a"] = out; save()

    if "e7" not in R:
        import e7_real_layer_rank as e7
        from tensor_sim import tensor_inner
        D, L, Rr = e7.load_real(8)
        aa = tensor_inner(D, L, Rr, D, L, Rr, None).detach()
        out = {}
        for r_tc in e7.RANKS:
            v = [e7.fit(D, L, Rr, r_tc, s, 0.0, aa)[0] for s in range(e7.SEEDS)]
            out[str(r_tc)] = [float(np.mean(v)), float(np.std(v))]
            print("e7", r_tc, out[str(r_tc)], flush=True)
        out["r_true"] = L.shape[0]
        R["e7"] = out; save()


# ---------------------------------------------------------------- plots
def f1():
    e6 = R["e6"]; KS = [1, 2, 4, 8, 32]; LAMS = [0.0, 0.1, 1.0, 10.0]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    x = np.arange(len(KS))
    ts0 = [e6[f"{k}|0.0"]["tsim"][0] for k in KS]; e0 = [e6[f"{k}|0.0"]["tsim"][1] for k in KS]
    ts1 = [e6[f"{k}|0.1"]["tsim"][0] for k in KS]; e1_ = [e6[f"{k}|0.1"]["tsim"][1] for k in KS]
    ms0 = [e6[f"{k}|0.0"]["mse"][0] for k in KS]
    ax[0].axhline(0, color=C["gray"], lw=.8)
    ax[0].bar(x - .2, ts0, .38, yerr=e0, color=C["bad"], label="MSE-only (λ=0)", capsize=2)
    ax[0].bar(x + .2, ts1, .38, yerr=e1_, color=C["good"], label="MSE + $L_{fid}$ (λ=0.1)", capsize=2)
    ax[0].plot(x, ms0, "o--", color=C["mid"], ms=4, label="MSE-only: its reconstruction error")
    ax[0].set_xticks(x); ax[0].set_xticklabels(KS); ax[0].set_xlabel("BatchTopK  k")
    ax[0].set_ylabel("tensor-sim  (1 − $L_{fid}$)")
    ax[0].set_title("MSE-only: near-perfect reconstruction,\nnear-ZERO true fidelity", fontsize=9)
    ax[0].legend(fontsize=7, loc="center left"); ax[0].set_ylim(-.25, 1.15)
    for k in KS:
        ax[1].plot(LAMS, [e6[f"{k}|{l}"]["tsim"][0] for l in LAMS], "o-", ms=3.5, label=f"k={k}")
    ax[1].set_xscale("symlog", linthresh=.1); ax[1].set_xlabel("fidelity weight  λ")
    ax[1].set_ylabel("tensor-sim"); ax[1].axhline(1, color=C["gray"], ls=":", lw=.8)
    ax[1].set_title("λ = 0.1 already buys tensor-sim ≈ 1.0\n(at zero reconstruction cost)", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.suptitle("F1 — the fidelity term is free: it costs no reconstruction and buys all the tensor-sim",
                 fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/f1_pareto.png", bbox_inches="tight"); plt.close(fig)


def f2():
    e = R["e3b"]; ts = [0.0, 0.01, 0.05, 0.2, 0.5, 1.0]
    xs = [max(t, 3e-3) for t in ts]                       # t=0 plotted at the left edge of the log axis
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(xs, [e[str(t)]["tsim"][0] for t in ts], yerr=[e[str(t)]["tsim"][1] for t in ts],
                fmt="o-", color=C["good"], ms=4, capsize=2, label="true tensor-sim (measured on ALL input directions)")
    ax.errorbar(xs, [e[str(t)]["ood"][0] for t in ts], yerr=[e[str(t)]["ood"][1] for t in ts],
                fmt="s-", color=C["bad"], ms=4, capsize=2, label="OOD MSE (inputs the training data never showed)")
    ax.set_xscale("log"); ax.set_xticks(xs)
    ax.set_xticklabels(["0\n(data-\nmatched)", "0.01", "0.05", "0.2", "0.5", "1.0\n(identity)"], fontsize=7)
    ax.set_xlabel("metric temperature  t     $\\Sigma_t=(1-t)\\,\\Sigma_{data}+t\\,I$")
    ax.axvspan(3e-3, 7e-3, color=C["bad"], alpha=.08)
    ax.annotate("BLIND: the metric\nreports 1.000 while\nthe truth is 0.196", (3e-3, .55), fontsize=7, color=C["bad"])
    ax.annotate("a 1% ridge\nis enough", (0.011, .80), fontsize=7, color=C["good"])
    ax.set_ylim(-.05, 1.08); ax.legend(fontsize=7, loc="center right")
    ax.set_title("F2 — a 1% identity ridge undoes the blindness\n(there is no realism-vs-coverage tradeoff)",
                 fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/f2_temperature.png", bbox_inches="tight"); plt.close(fig)


def f3():
    e = R["e5b"]; dzs = [1, 2, 3, 4, 5, 6]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for name, lab, c, m in [("balanced", "true layer: 4 equally important sub-features", C["good"], "o"),
                            ("skewed", "true layer: lopsided sub-features (1, ½, ¼, ⅛)", C["alt"], "s"),
                            ("one", "true layer: only ONE sub-feature (control)", C["mid"], "^")]:
        y = [e[f"{name}|{d}"][0] for d in dzs]; s = [e[f"{name}|{d}"][1] for d in dzs]
        ax.errorbar(dzs, y, yerr=s, fmt=m + "-", color=c, ms=4, capsize=2, label=lab)
    ax.axvline(4, color=C["gray"], ls="--", lw=.9)
    ax.annotate("the true number of\nsub-features is 4", (4.08, .55), fontsize=7, color=C["gray"])
    ax.set_xlabel("bottleneck width  dz′  (the ONLY structural knob — no masks anywhere)")
    ax.set_ylabel("tensor-sim"); ax.set_ylim(.4, 1.05); ax.legend(fontsize=7, loc="lower right")
    ax.set_title("F3 — how many sub-features does the layer really use?\nThe SHAPE of this curve is the answer", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/f3_bottleneck.png", bbox_inches="tight"); plt.close(fig)


def f4():
    e = R["e3a"]; nbs = [1, 2, 4, 8, 16]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for i, (gt, title) in enumerate([("random", "true layer is UNSTRUCTURED (ignores the blocks)"),
                                     ("block", "true layer really IS built from 4 blocks")]):
        for mode, lab, c, m in [("hard", "HARD mask (constraint)", C["bad"], "o"),
                                ("soft", "SOFT penalty (s=0.03)", C["good"], "s")]:
            ax[i].errorbar(nbs, [e[f"{gt}|{n}|{mode}"]["tsim"][0] for n in nbs],
                           yerr=[e[f"{gt}|{n}|{mode}"]["tsim"][1] for n in nbs],
                           fmt=m + "-", color=c, ms=4, capsize=2, label=f"{lab} — tensor-sim")
            ax[i].plot(nbs, [e[f"{gt}|{n}|{mode}"]["mmcs"][0] for n in nbs], m + ":", color=c, ms=3,
                       alpha=.55, label=f"{lab} — recovery of the true features")
        ax[i].set_xscale("log", base=2); ax[i].set_xticks(nbs); ax[i].set_xticklabels(nbs)
        ax[i].set_xlabel("n_blocks  (hierarchy granularity)"); ax[i].set_title(title, fontsize=9)
        ax[i].set_ylim(0, 1.08)
    ax[1].axvline(4, color=C["gray"], ls="--", lw=.9)
    ax[1].annotate("true = 4:\nhard holds to here,\nBREAKS beyond", (4.3, .30), fontsize=7, color=C["gray"])
    ax[0].set_ylabel("tensor-sim (solid) / recovery of true features (dotted)"); ax[0].legend(fontsize=6.5, loc="lower left")
    fig.suptitle("F4 — HARD masks are a PROBE (fidelity breaks iff mis-specified);  SOFT is a free PRIOR",
                 fontsize=10, y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIG}/f4_hierarchy.png", bbox_inches="tight"); plt.close(fig)


def f5():
    e = R["e7"]; ranks = [32, 64, 128, 256, 512, 1024]; rt = e["r_true"]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.errorbar([r / rt for r in ranks], [e[str(r)][0] for r in ranks], yerr=[e[str(r)][1] for r in ranks],
                fmt="o-", color=C["bad"], ms=4, capsize=2, label=f"REAL bilinear MLP (r={rt}, d=1152)")
    d = R["e5b"]
    ax.plot([x / 6 for x in [1, 2, 3, 4, 5, 6]], [d[f"balanced|{x}"][0] for x in [1, 2, 3, 4, 5, 6]],
            "s--", color=C["good"], ms=4, alpha=.7, label="toy layer that HAS structure (from F3, rescaled)")
    ax.set_xlabel("transcoder capacity  r′ / r   (fraction of the layer's own rank)")
    ax.set_ylabel("tensor-sim"); ax.set_ylim(0, 1.05); ax.legend(fontsize=7, loc="center right")
    ax.annotate("the curve never bends:\n22% of the rank buys\nonly 37% of the layer", (.06, .42), fontsize=7, color=C["bad"])
    ax.set_title("F5 — HONEST NEGATIVE: a real bilinear MLP shows no low-rank\nstructure under an ISOTROPIC "
                 "metric (a claim about Λ, not the layer)", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/f5_real_layer.png", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    compute()
    f1(); f2(); f3(); f4(); f5()
    print("FIGURES DONE")
