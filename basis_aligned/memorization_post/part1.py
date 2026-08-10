#!/usr/bin/env python
"""Part 1 of the memorization-in-bilinear-layers post: the 3-class Dog/Cat/Catfish toy.

Spec: ../memorization_post_handoff.md (sections 1a-1d, figures F1-F8, claim check C1).

Conventions (from the handoff):
  - Bilinear layer: y = D((Lx) * (Rx)), elementwise product over hidden dim H. No biases.
  - Folded tensor: T[i,j,c] = sum_h D[c,h] * sym(L[h,i]*R[h,j]);
    per-class symmetric interaction matrix B_c = T[:,:,c].
    Asymmetric per-class matrix M_c = sum_h D[c,h] * L[h,:] outer R[h,:] (NOT symmetrized).
  - Boolean inputs: x_i^2 = x_i, so DIAGONAL entries of B_c act as linear terms;
    diagonals are outlined in every heatmap.
  - 5 seeds per trained result; mean/range reported; sign flips flagged.
  - CPU only (GPU is busy with another program's run).

Stages (run `python part1.py --stage all` or a single stage):
  1a          hand-coded clean model, F1, rank claims
  predict_1b  write the pre-registered 1b prediction file (must be git-committed before 1b)
  1b          train H=8, weight-decay sweep, F2 + F3          [gated on committed 1b prediction]
  1c          train H=2 undercomplete, F4 + F5 (Gram)
  predict_c1  write Gram-based collateral predictions to disk  (must be committed before measure_c1)
  measure_c1  apply the KKT edits, measure collateral, F6      [gated on committed C1 predictions]
  1d          pull-out, two-class sub-model + discriminator (F7), path removal (F8)

Registration rule (handoff Pitfalls): prediction files are written to predictions/ and must be
git-committed BEFORE the corresponding training/measurement stage runs; the git commit time is
the registration time. The gates below enforce this via `git log`/`git diff`.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

torch.set_num_threads(4)

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
PRED = os.path.join(ROOT, "predictions")
MODELS = os.path.join(ROOT, "models")
for d in (FIG, PRED, MODELS):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------- setup
FEATS3 = ["furry", "happy", "whiskers"]
CLASSES3 = ["Dog", "Cat", "Catfish"]
# (first feature, second feature) of each class rule, per handoff 1a
PAIRS3 = {"Dog": (0, 1), "Cat": (0, 2), "Catfish": (2, 1)}
ABSENT3 = {"Dog": 2, "Cat": 1, "Catfish": 0}  # the class's absent feature
KEYS3 = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=np.float64)  # z_Dog, z_Cat, z_Catfish

# extended setup for 1d path removal
FEATS5 = ["furry", "happy", "whiskers", "hands", "dog-ears"]
CLASSES4 = ["Dog", "Cat", "Catfish", "Human"]
# keys: Dog has TWO paths (furry+happy and furry+dog-ears); Human = hands+dog-ears
KEYS5 = np.array([
    [1, 1, 0, 0, 0],   # Dog path 1 (furry, happy)
    [1, 0, 0, 0, 1],   # Dog path 2 (furry, dog-ears)
    [1, 0, 1, 0, 0],   # Cat (furry, whiskers)
    [0, 1, 1, 0, 0],   # Catfish (whiskers, happy)
    [0, 0, 0, 1, 1],   # Human (hands, dog-ears)
], dtype=np.float64)
KEYS5_NAMES = ["Dog(f,h)", "Dog(f,de)", "Cat", "Catfish", "Human"]
LABELS5 = np.array([0, 0, 1, 2, 3])

SEEDS = [0, 1, 2, 3, 4]
WDS = [0.0, 1e-3, 1e-2]

# ----------------------------------------------------------------------------- style
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRIDC = "#e1e0d9"
SURF = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
DIV = LinearSegmentedColormap.from_list(
    "brand_div", ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8a1f1e"])

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.facecolor": SURF, "savefig.facecolor": SURF,
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRIDC,
})


def heat(ax, B, feats, title, vmax, annot=True, outline_diag=True, fs=8.5, ylabels=True):
    """Interaction-matrix heatmap: diverging map centered at 0, diagonal cells outlined."""
    im = ax.imshow(B, cmap=DIV, vmin=-vmax, vmax=vmax)
    n = B.shape[0]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=fs)
    ax.set_yticklabels(feats if ylabels else [""] * n, fontsize=fs)
    ax.set_title(title, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    if outline_diag:
        for i in range(n):
            ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False,
                                   edgecolor=INK, lw=2.0, zorder=5))
    if annot:
        for i in range(n):
            for j in range(n):
                v = B[i, j]
                c = "#ffffff" if abs(v) > 0.55 * vmax else INK
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=c, fontsize=fs)
    return im


def save_fig(fig, name):
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved figures/{name}.png/.svg")


def diag_note(fig, y=-0.01):
    fig.text(0.5, y, "outlined diagonal cells act as LINEAR terms on boolean inputs (x$_i^2$ = x$_i$)",
             ha="center", fontsize=9, color=INK2, style="italic")


# ----------------------------------------------------------------------------- model / fold
def fold(L, R, D):
    """Return (B, M): symmetric folded slices B[c] and asymmetric M[c] = sum_h D[c,h] l_h r_h^T."""
    M = np.einsum("ch,hi,hj->cij", D, L, R)
    B = 0.5 * (M + np.transpose(M, (0, 2, 1)))
    return B, M


def forward_np(L, R, D, X):
    return ((X @ L.T) * (X @ R.T)) @ D.T


def all_bool_inputs(d):
    X = np.array([[(k >> i) & 1 for i in range(d)] for k in range(2 ** d)],
                 dtype=np.float64)
    return X


def train_bilinear(d, H, C, X, targets, seed, wd, steps=6000, lr=1e-2):
    """Full-batch AdamW on softmax CE with (possibly soft) targets. Returns numpy L,R,D + info."""
    torch.manual_seed(seed)
    L = (torch.randn(H, d, dtype=torch.float64) / np.sqrt(d)).requires_grad_()
    R = (torch.randn(H, d, dtype=torch.float64) / np.sqrt(d)).requires_grad_()
    D = (torch.randn(C, H, dtype=torch.float64) / np.sqrt(H)).requires_grad_()
    Xt = torch.tensor(X)
    Pt = torch.tensor(targets)
    opt = torch.optim.AdamW([L, R, D], lr=lr, weight_decay=wd)
    for step in range(steps):
        logits = ((Xt @ L.T) * (Xt @ R.T)) @ D.T
        loss = -(Pt * torch.log_softmax(logits, dim=1)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    Ln, Rn, Dn = L.detach().numpy(), R.detach().numpy(), D.detach().numpy()
    return Ln, Rn, Dn, float(loss.item())


def make_targets(labels_hard, n_uniform, C):
    """Hard one-hot rows followed by n_uniform uniform rows."""
    P = np.zeros((len(labels_hard) + n_uniform, C))
    for i, y in enumerate(labels_hard):
        P[i, y] = 1.0
    P[len(labels_hard):] = 1.0 / C
    return P


def key_accuracy(L, R, D, keys, labels):
    pred = forward_np(L, R, D, keys).argmax(1)
    return float((pred == labels).mean())


def margins(L, R, D, keys, labels):
    """Per-key margin: logit_true - max other logit."""
    lg = forward_np(L, R, D, keys)
    out = []
    for i, y in enumerate(labels):
        others = np.delete(lg[i], y)
        out.append(lg[i, y] - others.max())
    return np.array(out)


def mean_range(vals):
    vals = np.asarray(vals, dtype=np.float64)
    return float(vals.mean()), float(vals.min()), float(vals.max())


def sign_flip_flag(vals):
    s = np.sign(vals)
    return bool(not (np.all(s >= 0) or np.all(s <= 0)))


def fmt_mr(vals):
    m, lo, hi = mean_range(vals)
    flag = "  [SIGN FLIP ACROSS SEEDS]" if sign_flip_flag(vals) else ""
    return f"{m:+.4f} (range {lo:+.4f} .. {hi:+.4f}){flag}"


# ----------------------------------------------------------------------------- git gates
def repo_root():
    r = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT,
                       capture_output=True, text=True, check=True)
    return r.stdout.strip()


def require_committed(path, what):
    """Exit unless `path` is committed to git and identical to HEAD."""
    root = repo_root()
    rel = os.path.relpath(path, root)
    r = subprocess.run(["git", "log", "-1", "--format=%H %cI", "--", rel],
                       cwd=root, capture_output=True, text=True)
    if not r.stdout.strip():
        sys.exit(f"[GATE FAILED] {rel} has never been committed. "
                 f"Commit the {what} prediction file first (git commit time = registration time).")
    d = subprocess.run(["git", "diff", "HEAD", "--", rel],
                       cwd=root, capture_output=True, text=True)
    if d.stdout.strip():
        sys.exit(f"[GATE FAILED] {rel} differs from HEAD. Commit it before running the {what} stage.")
    sha, when = r.stdout.strip().split(" ", 1)
    print(f"[gate ok] {what} prediction registered in commit {sha[:9]} at {when}")


# ============================================================================= 1a
def stage_1a():
    print("=" * 70)
    print("STAGE 1a: hand-coded clean model (H=3), F1 + rank claims")
    print("=" * 70)
    d, H, C = 3, 3, 3
    L = np.zeros((H, d))
    R = np.zeros((H, d))
    D = np.eye(C, H)
    for c, cls in enumerate(CLASSES3):
        a, b = PAIRS3[cls]
        L[c, a] = 1.0   # L-row = first feature of class c
        R[c, b] = 1.0   # R-row = second feature
    B, M = fold(L, R, D)
    np.save(os.path.join(FIG, "F1_clean_B.npy"), B)
    np.save(os.path.join(FIG, "F1_clean_M_asym.npy"), M)
    np.savez(os.path.join(MODELS, "clean_h3.npz"), L=L, R=R, D=D)

    # F1
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    vmax = np.abs(B).max()
    for c, ax in enumerate(axes):
        im = heat(ax, B[c], FEATS3, f"{CLASSES3[c]}  (B$_{{{CLASSES3[c]}}}$)", vmax,
                  ylabels=(c == 0))
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    fig.suptitle("F1 — hand-coded clean model: per-class interaction matrices", y=1.04, color=INK)
    diag_note(fig, y=-0.06)
    save_fig(fig, "F1_clean_interactions")

    # acceptance: single symmetric off-diagonal pair, nothing else
    for c, cls in enumerate(CLASSES3):
        a, b = PAIRS3[cls]
        mask = np.zeros((3, 3), bool)
        mask[a, b] = mask[b, a] = True
        ok = np.allclose(B[c][mask], 0.5) and np.allclose(B[c][~mask], 0.0)
        print(f"  F1 acceptance {cls}: single symmetric off-diagonal pair at "
              f"({FEATS3[a]},{FEATS3[b]}) = 0.5, all else 0: {'PASS' if ok else 'FAIL'}")

    # rank claims
    print("\n  Symmetric eigendecompositions of B_c:")
    for c, cls in enumerate(CLASSES3):
        w, V = np.linalg.eigh(B[c])
        a, b = PAIRS3[cls]
        print(f"    {cls}: eigenvalues {np.round(w, 6)}  (rank {np.sum(np.abs(w) > 1e-12)})")
        for k in np.argsort(-np.abs(w))[:2]:
            print(f"      lambda={w[k]:+.3f}  v={np.round(V[:, k], 4)}")
        vp = (np.eye(3)[a] + np.eye(3)[b]) / np.sqrt(2)
        vm = (np.eye(3)[a] - np.eye(3)[b]) / np.sqrt(2)
        chk = (np.isclose(sorted(np.abs(w))[-2:], [0.5, 0.5]).all()
               and (np.isclose(np.abs(V[:, np.argmax(w)] @ vp), 1) or np.isclose(np.abs(V[:, np.argmax(w)] @ vm), 1)))
        print(f"      claim (eigenvalues +-1/2 on (e_a +- e_b)/sqrt2): {'PASS' if chk else 'FAIL'}")
    print("\n  Asymmetric M_c = sum_h D[c,h] l_h r_h^T (NOT symmetrized):")
    for c, cls in enumerate(CLASSES3):
        sv = np.linalg.svd(M[c], compute_uv=False)
        print(f"    {cls}: singular values {np.round(sv, 6)}  -> asymmetric rank "
              f"{np.sum(sv > 1e-12)} (claim: 1) {'PASS' if np.sum(sv > 1e-12) == 1 else 'FAIL'}")

    # functional check on all 8 boolean inputs
    X8 = all_bool_inputs(3)
    lg = forward_np(L, R, D, X8)
    quad = np.einsum("ni,cij,nj->nc", X8, B, X8)
    print(f"\n  fold consistency: max |D((Lx)*(Rx)) - x^T B x| over 8 inputs = "
          f"{np.abs(lg - quad).max():.2e}")
    print(f"  clean-model argmax on the 3 keys: "
          f"{[CLASSES3[i] for i in forward_np(L, R, D, KEYS3).argmax(1)]}")


# ============================================================================= predict_1b
PRED_1B = os.path.join(PRED, "part1_1b_prediction.md")


def stage_predict_1b():
    txt = """# Pre-registered prediction for Part 1, section 1b (trained H=8 model)

REGISTRATION: this file must be git-committed BEFORE the 1b training stage runs
(`part1.py` enforces this with a git gate). The git commit time of this file is
the registration time.

Setup being predicted: bilinear layer y = D((Lx)*(Rx)), d=3 features
[furry, happy, whiskers], C=3 classes (Dog=furry&happy, Cat=furry&whiskers,
Catfish=whiskers&happy), H=8 (overcomplete), softmax CE, full-batch AdamW,
weight decay in {0, 1e-3, 1e-2}, 5 seeds, trained on the 3 single-class keys
plus the all-zeros "none" input with uniform target.

## Prediction (from the handoff, quoted)

For each class c with feature pair (a,b) and absent feature m, the trained
interaction matrix B_c shows:

1. POSITIVE off-diagonal on the class's feature pair: B_c[a,b] > 0.
2. PLUS negative structure involving the class's absent feature m — one or both of:
   (i) negative off-diagonals coupling m to a and/or b: B_c[a,m] < 0, B_c[b,m] < 0;
   (ii) negative DIAGONAL on m: B_c[m,m] < 0 — i.e. linear "not-whiskers"-style
   logic carried by the diagonal (x_m^2 = x_m on booleans).

To report: which channel SGD actually used ((i) off-diagonal, (ii) diagonal/linear,
or both), and whether the weight-decay setting {0, 1e-3, 1e-2} changes the channel.
"""
    with open(PRED_1B, "w") as f:
        f.write(txt)
    print(f"[predict_1b] wrote {os.path.relpath(PRED_1B, ROOT)} — COMMIT IT before running --stage 1b")


# ============================================================================= 1b
def train_3class(H, seed, wd):
    X = np.vstack([KEYS3, np.zeros((1, 3))])
    P = make_targets([0, 1, 2], 1, 3)
    return train_bilinear(3, H, 3, X, P, seed, wd)


def stage_1b():
    require_committed(PRED_1B, "1b")
    print("=" * 70)
    print("STAGE 1b: trained overcomplete model H=8, weight-decay sweep, F2 + F3")
    print("=" * 70)
    print("dataset: 3 single-class keys + all-zeros 'none' with uniform target")
    print("note: pure bilinear layer has logits(0)=0 exactly, so the all-zeros row")
    print("      is automatically at uniform and contributes zero gradient.")

    B_clean = np.load(os.path.join(FIG, "F1_clean_B.npy"))
    allB = {}
    X8 = all_bool_inputs(3)
    for wd in WDS:
        Bs = []
        for seed in SEEDS:
            L, R, D, loss = train_3class(8, seed, wd)
            np.savez(os.path.join(MODELS, f"h8_wd{wd:g}_s{seed}.npz"), L=L, R=R, D=D)
            B, M = fold(L, R, D)
            Bs.append(B)
            acc = key_accuracy(L, R, D, KEYS3, np.array([0, 1, 2]))
            if seed == 0:
                gen = forward_np(L, R, D, X8).argmax(1)
                print(f"  wd={wd:g} seed0: loss={loss:.4f} key-acc={acc:.0%}; argmax on all 8 inputs "
                      f"(order 000,100,010,110,001,101,011,111): {[CLASSES3[i] for i in gen]}")
            assert acc == 1.0, f"training failed wd={wd} seed={seed}"
        Bs = np.array(Bs)  # (seed, class, i, j)
        allB[wd] = Bs
        np.save(os.path.join(FIG, f"F2_B_H8_wd{wd:g}.npy"), Bs)

        # cross-seed consistency
        cos = []
        for i in range(5):
            for j in range(i + 1, 5):
                a, b = Bs[i].ravel(), Bs[j].ravel()
                cos.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        print(f"  wd={wd:g}: cross-seed folded-tensor cosine mean={np.mean(cos):.4f} min={np.min(cos):.4f}")

        print(f"  wd={wd:g}: headline entries per class (mean/range over 5 seeds):")
        for c, cls in enumerate(CLASSES3):
            a, b = PAIRS3[cls]
            m = ABSENT3[cls]
            print(f"    {cls}: pair B[{FEATS3[a]},{FEATS3[b]}] = {fmt_mr(Bs[:, c, a, b])}")
            print(f"        absent-diag B[{FEATS3[m]},{FEATS3[m]}] = {fmt_mr(Bs[:, c, m, m])}")
            print(f"        absent-offdiag B[{FEATS3[a]},{FEATS3[m]}] = {fmt_mr(Bs[:, c, a, m])}")
            print(f"        absent-offdiag B[{FEATS3[b]},{FEATS3[m]}] = {fmt_mr(Bs[:, c, b, m])}")
            print(f"        own-diags B[{FEATS3[a]},{FEATS3[a]}] = {fmt_mr(Bs[:, c, a, a])}, "
                  f"B[{FEATS3[b]},{FEATS3[b]}] = {fmt_mr(Bs[:, c, b, b])}")

    # ---- F2: clean row + one row per weight decay (seed-mean), shared scale per row
    rows = [("hand-coded\n(clean)", B_clean)] + \
           [(f"trained H=8\nwd={wd:g}\n(5-seed mean)", allB[wd].mean(0)) for wd in WDS]
    fig, axes = plt.subplots(len(rows), 3, figsize=(10.2, 3.15 * len(rows)))
    for r, (label, Bset) in enumerate(rows):
        vmax = np.abs(Bset).max()
        for c in range(3):
            im = heat(axes[r, c], Bset[c], FEATS3, CLASSES3[c] if r == 0 else "", vmax,
                      ylabels=(c == 0))
        axes[r, 0].set_ylabel(label, fontsize=9.5, color=INK, labelpad=14)
        cb = fig.colorbar(im, ax=list(axes[r, :]), shrink=0.85, pad=0.02)
        cb.outline.set_visible(False)
    fig.suptitle("F2 — trained interaction matrices next to the hand-coded clean model",
                 y=1.005, color=INK)
    diag_note(fig, y=-0.015)
    save_fig(fig, "F2_trained_vs_clean")

    # ---- F3: diagonal entries per class, wd=1e-3, bar chart with seed range
    Bs = allB[1e-3]
    diags = np.stack([np.diagonal(Bs[:, c], axis1=1, axis2=2) for c in range(3)], axis=1)  # (seed, class, feat)
    np.save(os.path.join(FIG, "F3_diagonals_wd1e-3.npy"), diags)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    width = 0.25
    xs = np.arange(3)
    cols = [BLUE, ORANGE, AQUA]
    for f in range(3):
        m = diags[:, :, f].mean(0)
        lo = diags[:, :, f].min(0)
        hi = diags[:, :, f].max(0)
        ax.bar(xs + (f - 1) * width, m, width * 0.92, label=FEATS3[f], color=cols[f],
               yerr=[m - lo, hi - m], error_kw=dict(ecolor=INK2, lw=1, capsize=2))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASSES3)
    ax.set_ylabel("diagonal entry B$_c$[f,f]  (linear term on booleans)")
    ax.set_title("F3 — linear logic via the diagonal (H=8, wd=1e-3; mean $\\pm$ range over 5 seeds)",
                 color=INK)
    ax.legend(title="feature", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    save_fig(fig, "F3_diagonal_bars")


# ============================================================================= 1c
def stage_1c():
    print("=" * 70)
    print("STAGE 1c: undercomplete H=2 (3 classes), F4 + F5 Gram")
    print("=" * 70)
    wd = 1e-3
    Bs, fails = [], 0
    for seed in SEEDS:
        L, R, D, loss = train_3class(2, seed, wd)
        acc = key_accuracy(L, R, D, KEYS3, np.array([0, 1, 2]))
        print(f"  H=2 seed{seed}: loss={loss:.4f} key-acc={acc:.0%}")
        if acc < 1.0:
            fails += 1
        np.savez(os.path.join(MODELS, f"h2_wd{wd:g}_s{seed}.npz"), L=L, R=R, D=D)
        B, M = fold(L, R, D)
        Bs.append(B)
    if fails > 2:
        sys.exit("H=2 failed to train on most seeds -> would need the 6-feature/6-class fallback "
                 "(handoff 1c). Not implemented automatically; investigate first.")
    Bs = np.array(Bs)
    np.save(os.path.join(FIG, "F4_B_H2.npy"), Bs)

    cos = []
    for i in range(5):
        for j in range(i + 1, 5):
            a, b = Bs[i].ravel(), Bs[j].ravel()
            cos.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    consistent = np.min(cos) > 0.9
    print(f"  cross-seed folded-tensor cosine mean={np.mean(cos):.4f} min={np.min(cos):.4f} "
          f"-> {'seed-mean shown in F4' if consistent else 'seed 0 shown in F4 (solutions differ)'}")
    Bshow = Bs.mean(0) if consistent else Bs[0]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4))
    vmax = np.abs(Bshow).max()
    for c, ax in enumerate(axes):
        im = heat(ax, Bshow[c], FEATS3, CLASSES3[c], vmax, ylabels=(c == 0))
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.outline.set_visible(False)
    lbl = "mean of 5 seeds" if consistent else "seed 0"
    fig.suptitle(f"F4 — undercomplete H=2 interaction matrices ({lbl}): shared hidden units mix classes",
                 y=1.04, color=INK)
    diag_note(fig, y=-0.06)
    save_fig(fig, "F4_undercomplete_interactions")

    # ---- F5: stored-key Gram matrices, overcomplete vs undercomplete
    # Stored key of class c = the model's hidden representation h_c = (L z_c)*(R z_c).
    # (The raw input Gram Z^T Z is regime-independent: diag 2, off-diag 1 — printed below.)
    print(f"  raw input-key Gram Z^T Z (regime-independent):\n{KEYS3 @ KEYS3.T}")
    grams = {}
    for regime, tag in (("overcomplete (H=8)", "h8_wd0.001"), ("undercomplete (H=2)", "h2_wd0.001")):
        Gs = []
        for seed in SEEDS:
            z = np.load(os.path.join(MODELS, f"{tag}_s{seed}.npz"))
            hk = (KEYS3 @ z["L"].T) * (KEYS3 @ z["R"].T)   # (3, H) stored keys
            hn = hk / np.linalg.norm(hk, axis=1, keepdims=True)
            Gs.append(hn @ hn.T)
        Gs = np.array(Gs)
        grams[regime] = Gs
        short = "over" if "over" in regime else "under"
        np.save(os.path.join(FIG, f"F5_gram_{short}.npy"), Gs)
        offd = np.abs(Gs[:, ~np.eye(3, dtype=bool)])
        print(f"  {regime}: |off-diag| stored-key Gram cosines mean={offd.mean():.4f} "
              f"max={offd.max():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8))
    for ax, (regime, Gs) in zip(axes, grams.items()):
        im = heat(ax, Gs.mean(0), CLASSES3, regime, 1.0, outline_diag=False, fs=9)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.outline.set_visible(False)
    fig.suptitle("F5 — Gram matrix of stored keys $h_c = (Lz_c)*(Rz_c)$ (cosine, mean of 5 seeds)",
                 y=1.03, color=INK)
    fig.text(0.5, -0.04, "overcomplete: near-diagonal; undercomplete: off-diagonal mass "
             "(shared hidden units)", ha="center", fontsize=9, color=INK2, style="italic")
    save_fig(fig, "F5_stored_key_gram")


# ============================================================================= C1
PRED_C1_JSON = os.path.join(PRED, "part1_c1_predictions.json")
PRED_C1_MD = os.path.join(PRED, "part1_c1_predictions.md")


def load_model(tag, seed):
    z = np.load(os.path.join(MODELS, f"{tag}_s{seed}.npz"))
    return z["L"], z["R"], z["D"]


def kkt_edit_to_uniform(L, R, D, key):
    """Minimal-norm rank-1 edit to D making f(key) uniform (equal logits).
    Weights + the key only, no corpus. Returns edited D."""
    h = (key @ L.T) * (key @ R.T)
    y = D @ h
    ystar = np.full_like(y, y.mean())
    dD = np.outer(ystar - y, h) / (h @ h)
    return D + dD


def stage_predict_c1():
    print("=" * 70)
    print("STAGE predict_c1: Gram-based collateral predictions (written BEFORE measurement)")
    print("=" * 70)
    entries = []
    for regime, tag in (("overcomplete", "h8_wd0.001"), ("undercomplete", "h2_wd0.001")):
        for seed in SEEDS:
            L, R, D = load_model(tag, seed)
            hk = (KEYS3 @ L.T) * (KEYS3 @ R.T)
            hn = hk / np.linalg.norm(hk, axis=1, keepdims=True)
            G = hn @ hn.T
            for c in range(3):
                for v in range(3):
                    if v == c:
                        continue
                    entries.append({
                        "regime": regime, "seed": seed,
                        "edit_class": CLASSES3[c], "victim": CLASSES3[v],
                        "predicted_collateral": abs(float(G[c, v])),
                    })
    payload = {
        "definition": {
            "edit": "KKT-edit class c to uniform: minimal-norm rank-1 update to D along the stored "
                    "key h_c=(Lz_c)*(Rz_c) so that logits(z_c) become equal (uniform softmax). "
                    "Weights + the key only; no corpus.",
            "predicted_collateral": "|cosine(h_c, h_v)| — the (c,v) off-diagonal of the stored-key "
                                    "Gram matrix (F5). Computed from model weights BEFORE any edit "
                                    "is applied or measured.",
            "measured_collateral": "victim margin drop = margin_v(before) - margin_v(after) on the "
                                   "victim's key z_v, margin = true logit minus max other logit.",
            "prediction": "predicted_collateral rank-correlates strongly with measured margin drop "
                          "across all (edit, victim) pairs, seeds, and both regimes; undercomplete "
                          "points have large Gram off-diagonals and large measured collateral, "
                          "overcomplete points small on both.",
        },
        "registration_note": "The git commit time of this file is the registration time; part1.py "
                             "refuses to run measure_c1 unless this file is committed and unmodified.",
        "entries": entries,
    }
    with open(PRED_C1_JSON, "w") as f:
        json.dump(payload, f, indent=1)
    with open(PRED_C1_MD, "w") as f:
        f.write("# C1 pre-registered collateral predictions (Part 1)\n\n")
        f.write("Registration: the git commit time of this file (and the .json next to it) is the\n"
                "registration time; the measurement stage is git-gated on it.\n\n")
        f.write(payload["definition"]["edit"] + "\n\n")
        f.write("Predicted collateral = " + payload["definition"]["predicted_collateral"] + "\n\n")
        f.write("Measured collateral = " + payload["definition"]["measured_collateral"] + "\n\n")
        f.write("Prediction: " + payload["definition"]["prediction"] + "\n\n")
        f.write("| regime | seed | edit | victim | predicted |\n|---|---|---|---|---|\n")
        for e in entries:
            f.write(f"| {e['regime']} | {e['seed']} | {e['edit_class']} | {e['victim']} "
                    f"| {e['predicted_collateral']:.4f} |\n")
    print(f"[predict_c1] wrote {os.path.relpath(PRED_C1_JSON, ROOT)} and .md "
          f"({len(entries)} (edit,victim) predictions) — COMMIT BEFORE --stage measure_c1")


def stage_measure_c1():
    require_committed(PRED_C1_JSON, "C1")
    print("=" * 70)
    print("STAGE measure_c1: apply KKT edits, measure collateral, F6")
    print("=" * 70)
    with open(PRED_C1_JSON) as f:
        payload = json.load(f)
    pred_map = {(e["regime"], e["seed"], e["edit_class"], e["victim"]): e["predicted_collateral"]
                for e in payload["entries"]}
    labels = np.array([0, 1, 2])
    rows = []
    for regime, tag in (("overcomplete", "h8_wd0.001"), ("undercomplete", "h2_wd0.001")):
        for seed in SEEDS:
            L, R, D = load_model(tag, seed)
            m_before = margins(L, R, D, KEYS3, labels)
            acc_before = forward_np(L, R, D, KEYS3).argmax(1) == labels
            for c in range(3):
                D2 = kkt_edit_to_uniform(L, R, D, KEYS3[c])
                # edit success: logits at z_c equal
                lg_c = forward_np(L, R, D2, KEYS3[c:c + 1])[0]
                assert np.allclose(lg_c, lg_c.mean()), "KKT edit failed to reach uniform"
                m_after = margins(L, R, D2, KEYS3, labels)
                acc_after = forward_np(L, R, D2, KEYS3).argmax(1) == labels
                for v in range(3):
                    if v == c:
                        continue
                    rows.append({
                        "regime": regime, "seed": seed,
                        "edit_class": CLASSES3[c], "victim": CLASSES3[v],
                        "predicted": pred_map[(regime, seed, CLASSES3[c], CLASSES3[v])],
                        "measured_margin_drop": float(m_before[v] - m_after[v]),
                        "victim_margin_before": float(m_before[v]),
                        "victim_margin_after": float(m_after[v]),
                        "victim_acc_flip": bool(acc_before[v] and not acc_after[v]),
                    })
    with open(os.path.join(FIG, "F6_measured.json"), "w") as f:
        json.dump(rows, f, indent=1)
    P = np.array([r["predicted"] for r in rows])
    Md = np.array([r["measured_margin_drop"] for r in rows])
    np.save(os.path.join(FIG, "F6_pred_measured.npy"), np.stack([P, Md], 1))

    from scipy import stats  # local import; scipy is available in the venv
    r_p = stats.pearsonr(P, Md)
    r_s = stats.spearmanr(P, Md)
    print(f"  ALL points (n={len(P)}): Pearson r={r_p.statistic:.4f} (p={r_p.pvalue:.2e}); "
          f"Spearman rho={r_s.statistic:.4f} (p={r_s.pvalue:.2e})")
    for regime in ("overcomplete", "undercomplete"):
        idx = [i for i, r in enumerate(rows) if r["regime"] == regime]
        if len(set(P[idx])) > 1:
            rp = stats.pearsonr(P[idx], Md[idx])
            rs = stats.spearmanr(P[idx], Md[idx])
            print(f"  {regime} (n={len(idx)}): Pearson r={rp.statistic:.4f}; "
                  f"Spearman rho={rs.statistic:.4f}")
        flips = sum(rows[i]["victim_acc_flip"] for i in idx)
        print(f"  {regime}: victim accuracy flips: {flips}/{len(idx)}; "
              f"measured margin drop mean={Md[idx].mean():+.4f} max={Md[idx].max():+.4f}")

    # F6 scatter
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for regime, col, mk in (("overcomplete", BLUE, "o"), ("undercomplete", ORANGE, "s")):
        idx = [i for i, r in enumerate(rows) if r["regime"] == regime]
        ax.scatter(P[idx], Md[idx], s=42, c=col, marker=mk, label=regime,
                   edgecolors=SURF, linewidths=1.2, zorder=3)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("predicted collateral: stored-key Gram |cos(h$_c$, h$_v$)|  (pre-registered)")
    ax.set_ylabel("measured collateral: victim margin drop")
    ax.set_title("F6 — Gram-predicted vs measured collateral of single-class KKT edits\n"
                 "one point per (edit, victim) pair, 5 seeds, both regimes", color=INK)
    ax.legend(frameon=False)
    ax.text(0.02, 0.98, f"Pearson r = {r_p.statistic:.3f}\nSpearman $\\rho$ = {r_s.statistic:.3f}",
            transform=ax.transAxes, va="top", fontsize=10, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    save_fig(fig, "F6_collateral_scatter")


# ============================================================================= 1d
def fit_min_H(B_target, d, C, Hmax=6, tol=1e-8, restarts=3, steps=6000):
    """Smallest H such that a bilinear layer folds exactly (Frobenius, relative) to B_target."""
    Bt = torch.tensor(B_target)
    nrm = float(np.linalg.norm(B_target))
    for H in range(1, Hmax + 1):
        best = np.inf
        for rs in range(restarts):
            torch.manual_seed(100 + rs)
            L = (torch.randn(H, d, dtype=torch.float64) * 0.5).requires_grad_()
            R = (torch.randn(H, d, dtype=torch.float64) * 0.5).requires_grad_()
            D = (torch.randn(C, H, dtype=torch.float64) * 0.5).requires_grad_()
            opt = torch.optim.Adam([L, R, D], lr=3e-2)
            for step in range(steps):
                M = torch.einsum("ch,hi,hj->cij", D, L, R)
                Bf = 0.5 * (M + M.transpose(1, 2))
                loss = ((Bf - Bt) ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            rel = float(loss.item()) / nrm ** 2
            best = min(best, rel)
        print(f"    H={H}: best relative sq. Frobenius error {best:.2e}")
        if best < tol:
            return H, best
    return None, best


def train_extended(H, seed, wd=1e-3):
    X = np.vstack([KEYS5, np.zeros((1, 5))])
    P = make_targets(list(LABELS5), 1, 4)
    return train_bilinear(5, H, 4, X, P, seed, wd, steps=8000)


def path_removal_edit(L, R, D, f_idx, de_idx, dog_idx, thresh=0.05):
    """Greedily zero whole hidden units (D[:,h]=0) to remove the (furry,dog-ears)->Dog
    interaction entry; returns edited D, list of zeroed units, residual entry."""
    B, M = fold(L, R, D)
    target = B[dog_idx, f_idx, de_idx]
    contrib = D[dog_idx, :] * 0.5 * (L[:, f_idx] * R[:, de_idx] + L[:, de_idx] * R[:, f_idx])
    D2 = D.copy()
    zeroed = []
    resid = target
    for _ in range(D.shape[1]):
        # pick the unit whose removal most reduces |residual|
        gains = np.array([abs(resid - contrib[h]) if h not in zeroed else np.inf
                          for h in range(D.shape[1])])
        h = int(np.argmin(gains))
        if gains[h] >= abs(resid):
            break
        D2[:, h] = 0.0
        zeroed.append(h)
        resid = resid - contrib[h]
        if abs(resid) <= thresh * abs(target):
            break
    return D2, zeroed, float(resid), float(target)


def stage_1d():
    print("=" * 70)
    print("STAGE 1d: edits on the toy (pull-out, two-class sub-model + F7, path removal + F8)")
    print("=" * 70)
    X8 = all_bool_inputs(3)

    # ---------- pull-out
    print("\n-- 1d pull-out: slice T[:,:,Dog], reconstruct standalone dog-detector --")
    for name, tag in (("hand-coded", "clean_h3"), ("trained H=8 wd=1e-3 seed0", "h8_wd0.001_s0")):
        z = np.load(os.path.join(MODELS, f"{tag}.npz")) if tag == "clean_h3" \
            else np.load(os.path.join(MODELS, "h8_wd0.001_s0.npz"))
        L, R, D = z["L"], z["R"], z["D"]
        B, M = fold(L, R, D)
        Bdog, Mdog = B[0], M[0]
        # standalone via symmetric eigendecomposition: L-rows v_k, R-rows lam_k v_k, D=1s
        w, V = np.linalg.eigh(Bdog)
        keep = np.abs(w) > 1e-12
        Lp = V[:, keep].T
        Rp = (V[:, keep] * w[keep]).T
        Dp = np.ones((1, keep.sum()))
        standalone = forward_np(Lp, Rp, Dp, X8)[:, 0]
        full_dog = forward_np(L, R, D, X8)[:, 0]
        err = np.abs(standalone - full_dog).max()
        print(f"  {name}: standalone dog-detector H={keep.sum()}, "
              f"max |mismatch| over all 8 inputs = {err:.2e} "
              f"({'EXACT' if err < 1e-10 else 'NOT exact'})")
        sv = np.linalg.svd(Mdog, compute_uv=False)
        print(f"    CP/SVD of asymmetric Dog slice: singular values {np.round(sv, 5)} "
              f"-> asymmetric rank {np.sum(sv > 1e-8 * sv[0])}"
              + ("  (rank-1 recovery: "
                 + ("PASS" if np.sum(sv > 1e-8 * sv[0]) == 1 else "no — trained slice not rank 1")
                 + ")"))
        if tag == "clean_h3":
            # explicit rank-1 CP recovery back to L/R/D form
            U, S, Vt = np.linalg.svd(Mdog)
            L1 = (U[:, :1] * S[:1]).T
            R1 = Vt[:1, :]
            D1 = np.ones((1, 1))
            e2 = np.abs(forward_np(L1, R1, D1, X8)[:, 0] - full_dog).max()
            print(f"    rank-1 L/R/D reconstruction functional mismatch: {e2:.2e}")

    # ---------- two-class sub-model + discriminator
    print("\n-- 1d two-class sub-model (Dog, Cat): minimal hidden dim --")
    zc = np.load(os.path.join(MODELS, "clean_h3.npz"))
    B_clean, _ = fold(zc["L"], zc["R"], zc["D"])
    print("  clean model, target = symmetric slices (functional equality on booleans):")
    Hmin, err = fit_min_H(B_clean[[0, 1]], 3, 2)
    print(f"  clean two-class sub-model needs H = {Hmin} (rel err {err:.1e})")
    zt = np.load(os.path.join(MODELS, "h8_wd0.001_s0.npz"))
    B_tr0, _ = fold(zt["L"], zt["R"], zt["D"])
    print("  trained model (H=8, wd=1e-3, seed 0):")
    Hmin_t, err_t = fit_min_H(B_tr0[[0, 1]], 3, 2, tol=1e-6)
    print(f"  trained two-class sub-model needs H = {Hmin_t} (rel err {err_t:.1e})")

    # discriminator
    Bs_tr = np.load(os.path.join(FIG, "F2_B_H8_wd0.001.npy"))
    disc_clean = B_clean[0] - B_clean[1]
    disc_tr = Bs_tr.mean(0)[0] - Bs_tr.mean(0)[1]
    np.save(os.path.join(FIG, "F7_discriminator_clean.npy"), disc_clean)
    np.save(os.path.join(FIG, "F7_discriminator_trained_meanseeds.npy"), disc_tr)
    for nm, Dm in (("clean", disc_clean), ("trained (seed-mean)", disc_tr)):
        w, V = np.linalg.eigh(Dm)
        order = np.argsort(-np.abs(w))
        print(f"  discriminator T[:,:,Dog]-T[:,:,Cat] ({nm}) eigenvalues: "
              f"{np.round(w[order], 4)}")
        for k in order[:2]:
            print(f"    lambda={w[k]:+.4f} v={np.round(V[:, k], 4)}")

    fig = plt.figure(figsize=(10.5, 6.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.55, wspace=0.35)
    for j, (nm, Dm) in enumerate((("hand-coded clean", disc_clean),
                                  ("trained H=8 (mean of 5 seeds)", disc_tr))):
        ax = fig.add_subplot(gs[0, j])
        vmax = np.abs(Dm).max()
        im = heat(ax, Dm, FEATS3, f"Dog $-$ Cat discriminator: {nm}", vmax)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.03)
        cb.outline.set_visible(False)
        axv = fig.add_subplot(gs[1, j])
        w, V = np.linalg.eigh(Dm)
        order = np.argsort(-np.abs(w))[:2]
        xs = np.arange(3)
        wdt = 0.36
        for k, (ev, col) in enumerate(zip(order, (BLUE, ORANGE))):
            vv = V[:, ev].copy()
            if vv[np.argmax(np.abs(vv))] < 0:
                vv = -vv  # sign convention: largest component positive
            axv.bar(xs + (k - 0.5) * wdt, vv, wdt * 0.9, color=col,
                    label=f"$\\lambda$={w[ev]:+.3f}")
        axv.axhline(0, color=MUTED, lw=1)
        axv.set_xticks(xs)
        axv.set_xticklabels(FEATS3)
        axv.set_title("top eigenvectors (by |$\\lambda$|)", fontsize=10, color=INK)
        axv.legend(frameon=False, fontsize=9)
        for s in ("top", "right"):
            axv.spines[s].set_visible(False)
        axv.grid(axis="y", color=GRIDC, lw=0.8)
        axv.set_axisbelow(True)
    fig.suptitle("F7 — the Dog$-$Cat discriminator matrix and its eigendecomposition", color=INK)
    diag_note(fig, y=0.02)
    save_fig(fig, "F7_discriminator")

    # ---------- path removal (extended 5-feature, 4-class setup)
    print("\n-- 1d path removal: extended setup [furry,happy,whiskers,hands,dog-ears] --")
    f_idx, de_idx, dog_idx = 0, 4, 0
    regimes = {}
    for regime, Hs in (("overcomplete", [12]), ("undercomplete", [3, 4])):
        for H in Hs:
            models, accs = [], []
            for seed in SEEDS:
                L, R, D, loss = train_extended(H, seed)
                a = key_accuracy(L, R, D, KEYS5, LABELS5)
                models.append((L, R, D))
                accs.append(a)
            ok = sum(a == 1.0 for a in accs)
            print(f"  {regime} H={H}: {ok}/5 seeds fit all 5 keys "
                  f"(accs {[f'{a:.0%}' for a in accs]})")
            if ok >= 4:
                regimes[regime] = (H, models)
                for seed, (L, R, D) in enumerate(models):
                    np.savez(os.path.join(MODELS, f"ext_{regime}_H{H}_s{seed}.npz"), L=L, R=R, D=D)
                break
        else:
            sys.exit(f"extended {regime} failed to train at H in {Hs}")

    table = {}
    for regime, (H, models) in regimes.items():
        print(f"\n  {regime} (H={H}): removing the (furry,dog-ears)->Dog path "
              f"by zeroing whole hidden units (D[:,h]=0)")
        deltas, kept_entries = [], []
        for seed, (L, R, D) in enumerate(models):
            B0, _ = fold(L, R, D)
            m0 = margins(L, R, D, KEYS5, LABELS5)
            D2, zeroed, resid, target = path_removal_edit(L, R, D, f_idx, de_idx, dog_idx)
            B1, _ = fold(L, R, D2)
            m1 = margins(L, R, D2, KEYS5, LABELS5)
            pred0 = forward_np(L, R, D, KEYS5).argmax(1)
            pred1 = forward_np(L, R, D2, KEYS5).argmax(1)
            deltas.append(((pred1 == LABELS5).astype(float) - (pred0 == LABELS5).astype(float),
                           m1 - m0))
            kept_entries.append([target, resid,
                                 B0[dog_idx, 0, 1], B1[dog_idx, 0, 1],       # (furry,happy)->Dog
                                 B0[3, 3, 4], B1[3, 3, 4]])                  # (hands,dog-ears)->Human
            if seed == 0:
                np.save(os.path.join(FIG, f"F8_B_ext_{regime}_before.npy"), B0)
                np.save(os.path.join(FIG, f"F8_B_ext_{regime}_after.npy"), B1)
                table[regime + "_seed0"] = (B0, B1)
            print(f"    seed{seed}: zeroed units {zeroed} (of H={H}); "
                  f"target entry {target:+.4f} -> residual {resid:+.4f}; "
                  f"argmax before {[CLASSES4[i] for i in pred0]} -> after {[CLASSES4[i] for i in pred1]}")
        ke = np.array(kept_entries)
        print(f"    preserved (furry,happy)->Dog entry: before {fmt_mr(ke[:, 2])} after {fmt_mr(ke[:, 3])}")
        print(f"    preserved (hands,dog-ears)->Human entry: before {fmt_mr(ke[:, 4])} after {fmt_mr(ke[:, 5])}")
        acc_d = np.array([d[0] for d in deltas])   # (seed, key)
        mar_d = np.array([d[1] for d in deltas])
        table[regime] = (acc_d, mar_d, H)
        for k, kn in enumerate(KEYS5_NAMES):
            print(f"    key {kn}: acc delta {fmt_mr(acc_d[:, k])}, margin delta {fmt_mr(mar_d[:, k])}")
        with open(os.path.join(FIG, f"F8_table_{regime}.json"), "w") as fjs:
            json.dump({"H": H, "keys": KEYS5_NAMES,
                       "acc_delta_mean": acc_d.mean(0).tolist(),
                       "acc_delta_per_seed": acc_d.tolist(),
                       "margin_delta_mean": mar_d.mean(0).tolist()}, fjs, indent=1)

    # F8 figure: before/after B_Dog and B_Human per regime (seed 0) + accuracy-delta table
    fig = plt.figure(figsize=(15.5, 9.6))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.55], hspace=0.75, wspace=0.42)
    for r, regime in enumerate(("overcomplete", "undercomplete")):
        B0, B1 = table[regime + "_seed0"]
        vmax = max(np.abs(B0[[0, 3]]).max(), np.abs(B1[[0, 3]]).max())
        panels = [(B0[0], "Dog before"), (B1[0], "Dog after"),
                  (B0[3], "Human before"), (B1[3], "Human after")]
        for j, (Bp, ttl) in enumerate(panels):
            ax = fig.add_subplot(gs[r, j])
            im = heat(ax, Bp, FEATS5, ttl, vmax, fs=7, ylabels=(j == 0))
            if j == 0:
                ax.set_ylabel(f"{regime}\n(H={table[regime][2]}, seed 0)", fontsize=10, color=INK)
        cb = fig.colorbar(im, ax=[fig.axes[-4], fig.axes[-3], fig.axes[-2], fig.axes[-1]],
                          shrink=0.8, pad=0.015)
        cb.outline.set_visible(False)
    axt = fig.add_subplot(gs[2, :])
    axt.axis("off")
    lines = ["per-key accuracy delta after path removal (mean over 5 seeds; 0 = unchanged, "
             "-1 = key now misclassified):", ""]
    hdr = f"{'':>16}" + "".join(f"{kn:>13}" for kn in KEYS5_NAMES)
    lines.append(hdr)
    for regime in ("overcomplete", "undercomplete"):
        acc_d = table[regime][0]
        lines.append(f"{regime:>16}" + "".join(f"{v:>13.2f}" for v in acc_d.mean(0)))
    axt.text(0.5, 0.9, "\n".join(lines), family="monospace", fontsize=10.5,
             ha="center", va="top", color=INK)
    fig.suptitle("F8 — removing the (furry, dog-ears)$\\to$Dog path by zeroing its hidden units:\n"
                 "surgical when overcomplete, collateral when undercomplete", color=INK, y=0.99)
    save_fig(fig, "F8_path_removal")


# ============================================================================= main
STAGES = {
    "1a": stage_1a,
    "predict_1b": stage_predict_1b,
    "1b": stage_1b,
    "1c": stage_1c,
    "predict_c1": stage_predict_c1,
    "measure_c1": stage_measure_c1,
    "1d": stage_1d,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=list(STAGES) + ["all"])
    args = ap.parse_args()
    if args.stage == "all":
        for name, fn in STAGES.items():
            fn()
    else:
        STAGES[args.stage]()
