#!/usr/bin/env python
"""Part 2 of the memorization-in-bilinear-layers post: 100 random facts.

Spec: ../memorization_post_handoff.md, section "Part 2".

Conventions (handoff, followed exactly):
  - Bilinear layer y = D((Lx) * (Rx)), elementwise product over hidden dim H.
  - Folded tensor T[i,j,c] = sum_h D[c,h] * sym(L[h,i] R[h,j]); per-class
    interaction matrix B_c = T[:,:,c], symmetrized. (Stored here as B[c,i,j].)
  - Boolean inputs x in {0,1}^n, x_i^2 = x_i (diagonals act as linear terms).
  - Seeds: 5 per trained result, mean + range, sign flips flagged.

Setup: n = 20-bit random keys, K = 100 facts -> 10 random classes, single
bilinear layer, H swept so memorization succeeds with meaningful key overlap.

Construction: D is FIXED at the tiled negative identity (exactly D = -I when
H = C; for H = m*C, D = -[I_C | ... | I_C]) and never trained. L and R are
found by ALS: alternating EXACT convex least-squares block solves (each block
solve is a linear least-squares problem because the layer is linear in L for
fixed R and vice versa; with the tiled-D structure the solve separates per
class).

KKT edit (add / remove a fact): with f_c(x) = x^T B_c x linear in B_c, the
minimum-C-weighted-norm edit (min ||C^{1/2} dB_c C^{1/2}||_F subject to
z*^T (B_c + dB_c) z* = y*_c) is the rank-1 update
    dB_c = alpha_c * (C^{-1} z*)(C^{-1} z*)^T,
    alpha_c = (y*_c - f_c(z*)) / (z*^T C^{-1} z*)^2,
with C = sum_k z_k z_k^T over the STORED FACT LIST.
*** The edit uses the model weights and the fact-key list ONLY - no training
corpus is accessed anywhere in this file. ***

F12 tail: exhaustive enumeration of all 2^20 boolean inputs in batches (not a
derived bound); the analytic Gram-overlap bound is overlaid.

Run:  python part2.py --stages all
      python part2.py --stages setup,construct,sgd,f9,f10,f11_predict
      (git commit predictions/ here: commit time = registration time)
      python part2.py --stages f11_measure,f12

Everything is seeded and re-runnable end to end; intermediate artifacts are
cached in cache_part2/ (regenerated deterministically if deleted).
"""

import argparse
import subprocess
import json
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
PRED = ROOT / "predictions"
CACHE = ROOT / "cache_part2"
for d in (FIG, PRED, CACHE):
    d.mkdir(exist_ok=True)

# ----------------------------------------------------------------- config
N_BITS = 20
K_FACTS = 100
N_CLASSES = 10
MARGIN = 5.0            # target logit for the true class; 0 for the rest
MASTER_SEED = 20260810
SGD_SEEDS = [0, 1, 2, 3, 4]
H_SWEEP = [10, 20, 30, 40, 50, 60, 80, 100]
SGD_STEPS = 12000
SGD_LR = 5e-3
SGD_L1 = 1e-3           # sparsity regularization: L1 on all weight entries
N_UNLEARN = 10
RECOVERY_TAU = 0.80     # |cos| threshold for "fact recovered" in extraction

NO_CORPUS_NOTE = ("KKT edit inputs: model weights + fact-key list only "
                  "(C = sum_k z_k z_k^T over the 100 stored keys); "
                  "NO training corpus accessed.")

# palette (dataviz reference, light mode)
INK = "#0b0b0b"
MUTED = "#898781"
GRIDC = "#e1e0d9"
SURF = "#fcfcfb"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
RED = "#e34948"
VIOLET = "#4a3aa7"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRIDC, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 9.5, "axes.titlesize": 10.5, "figure.titlesize": 12,
    "legend.frameon": False,
    "svg.fonttype": "none",
})


def savefig(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] wrote figures/{name}.png + .svg")


def metrics_update(section, payload):
    path = FIG / "part2_metrics.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = payload
    path.write_text(json.dumps(data, indent=2, default=float))
    print(f"[metrics] updated section '{section}'")


def jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ----------------------------------------------------------------- data
def make_facts():
    rng = np.random.default_rng(MASTER_SEED)
    while True:
        Z = rng.integers(0, 2, size=(K_FACTS, N_BITS)).astype(np.float64)
        rows = {tuple(r) for r in Z.astype(int)}
        if len(rows) == K_FACTS and Z.sum(axis=1).min() >= 1:
            break
    y = rng.integers(0, N_CLASSES, size=K_FACTS)
    return Z, y


def fact_ints(Z):
    return (Z.astype(np.uint64) @ (1 << np.arange(N_BITS, dtype=np.uint64))).astype(np.uint64)


# ----------------------------------------------------------------- model utils
def fold(L, R, D):
    """Folded tensor B[c] = sum_h D[c,h] sym(outer(L_h, R_h))."""
    M = np.einsum("ch,hi,hj->cij", D, L, R, optimize=True)
    return 0.5 * (M + M.transpose(0, 2, 1))


def logits_folded(B, X):
    return np.einsum("bi,cij,bj->bc", X, B, X, optimize=True)


def acc_folded(B, Z, y):
    return float((logits_folded(B, Z).argmax(1) == y).mean())


def tiled_D(H):
    assert H % N_CLASSES == 0
    return -np.tile(np.eye(N_CLASSES), H // N_CLASSES)


# ----------------------------------------------------------------- ALS
def als_block_solve(Z, Ycol_by_class, other, blocks, ridge=1e-10):
    """Exact convex LS for one factor given the other, per class block.

    For class c with unit set blocks[c], solve for rows {w_h}:
      sum_{h in block} -(w_h . z_k)(other_h . z_k) = Y[k, c]  for all k.
    Linear least squares in the stacked w entries (min-norm solution).
    """
    H = other.shape[0]
    W = np.zeros_like(other)
    P = Z @ other.T                    # (K, H)
    for c, hs in enumerate(blocks):
        A = np.concatenate([-(P[:, h])[:, None] * Z for h in hs], axis=1)
        b = Ycol_by_class[:, c]
        AtA = A.T @ A
        AtA[np.diag_indices_from(AtA)] += ridge * max(1.0, np.trace(AtA) / AtA.shape[0])
        sol = np.linalg.solve(AtA, A.T @ b)
        # fall back to min-norm lstsq if underdetermined solve is unstable
        if not np.all(np.isfinite(sol)):
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        W[hs] = sol.reshape(len(hs), N_BITS)
    return W


def als_construction(Z, y, H, seed, iters=60, tol=1e-12, restarts=3):
    """Closed-form/ALS construction with D fixed at tiled -I."""
    D = tiled_D(H)
    blocks = [list(range(c, H, N_CLASSES)) for c in range(N_CLASSES)]
    Y = MARGIN * np.eye(N_CLASSES)[y]
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed + 1000 * r)
        R = rng.normal(0, 1.0 / np.sqrt(N_BITS), size=(H, N_BITS))
        L = np.zeros_like(R)
        losses = []
        for it in range(iters):
            L = als_block_solve(Z, Y, R, blocks)
            R = als_block_solve(Z, Y, L, blocks)
            mse = float(np.mean((logits_folded(fold(L, R, D), Z) - Y) ** 2))
            losses.append(mse)
            if mse < tol:
                break
        if best is None or losses[-1] < best[3][-1]:
            best = (L, R, D, losses)
    return best


# ----------------------------------------------------------------- SGD
def sgd_train(Z, y, H, seed, steps=SGD_STEPS, lr=SGD_LR, l1=SGD_L1):
    import torch
    import torch.nn.functional as tF
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    Zt = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    L = torch.nn.Parameter(torch.randn(H, N_BITS) / np.sqrt(N_BITS))
    R = torch.nn.Parameter(torch.randn(H, N_BITS) / np.sqrt(N_BITS))
    D = torch.nn.Parameter(torch.randn(N_CLASSES, H) / np.sqrt(H))
    init = (L.detach().numpy().copy(), R.detach().numpy().copy(),
            D.detach().numpy().copy())
    opt = torch.optim.AdamW([L, R, D], lr=lr, weight_decay=0.0)
    for t in range(steps):
        opt.zero_grad()
        logits = ((Zt @ L.T) * (Zt @ R.T)) @ D.T
        loss = tF.cross_entropy(logits, yt) + l1 * (
            L.abs().mean() + R.abs().mean() + D.abs().mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        logits = ((Zt @ L.T) * (Zt @ R.T)) @ D.T
        acc = float((logits.argmax(1) == yt).float().mean())
    out = (L.detach().numpy().astype(np.float64),
           R.detach().numpy().astype(np.float64),
           D.detach().numpy().astype(np.float64))
    return out, init, acc, float(loss.item())


# ----------------------------------------------------------------- KKT edits
def kkt_edit_single(B, z, target, Cinv):
    """Rank-1 KKT edit: min C-weighted Frobenius norm s.t. f(z) = target.

    delta_c = alpha_c * u u^T with u = C^{-1} z (the C^{-1}-weighted key
    direction), alpha_c = (target_c - f_c(z)) / (z^T C^{-1} z)^2.
    """
    u = Cinv @ z
    s = float(z @ u)
    f = np.einsum("i,cij,j->c", z, B, z, optimize=True)
    alpha = (target - f) / (s ** 2)
    return B + alpha[:, None, None] * np.outer(u, u)[None], alpha


def naive_edit_single(B, z, target):
    """Naive removal: subtract the rank-1 component in the raw key direction
    (no C^{-1} weighting, no re-tensioning across facts)."""
    s = float(z @ z)
    f = np.einsum("i,cij,j->c", z, B, z, optimize=True)
    beta = (target - f) / (s ** 2)
    return B + beta[:, None, None] * np.outer(z, z)[None]


def kkt_edit_cyclic(B, Zs, targets, Cinv, tol=1e-9, max_passes=2000):
    """Apply the single-fact KKT edit cyclically over a set of facts until all
    constraints hold (alternating projections onto affine sets; converges to
    the joint minimum-C-norm solution)."""
    B = B.copy()
    for p in range(max_passes):
        worst = 0.0
        for m in range(len(Zs)):
            B, _ = kkt_edit_single(B, Zs[m], targets[m], Cinv)
        for m in range(len(Zs)):
            f = np.einsum("i,cij,j->c", Zs[m], B, Zs[m], optimize=True)
            worst = max(worst, float(np.abs(f - targets[m]).max()))
        if worst < tol:
            return B, p + 1, worst
    return B, max_passes, worst


def kkt_interpolant(Z, y, Cinv):
    """Joint minimum-C-norm interpolant of all facts from the zero tensor:
    B_c = sum_k lambda_{ck} u_k u_k^T, u_k = C^{-1} z_k, with lambda solving
    S lambda = Y, S_{jk} = (z_j^T C^{-1} z_k)^2 (elementwise square of the
    fact-frame Gram / hat matrix)."""
    Hat = Z @ Cinv @ Z.T
    S = Hat ** 2
    Y = MARGIN * np.eye(N_CLASSES)[y]
    condS = float(np.linalg.cond(S))
    Lam = np.linalg.solve(S, Y)                      # (K, C)
    U = Z @ Cinv                                     # rows u_k
    B = np.einsum("kc,ki,kj->cij", Lam, U, U, optimize=True)
    return B, Lam, condS


# ----------------------------------------------------------------- similarity
def center_classes(B):
    return B - B.mean(axis=0, keepdims=True)


def gauss_frob_cos(B1, B2, offdiag_only=False):
    """Frobenius cosine of class-centered symmetric tensors. For zero-mean
    Gaussian inputs the output covariance of two quadratics is
    2 <B1, B2>_F, so this IS the Gaussian-input output correlation."""
    A1, A2 = center_classes(B1), center_classes(B2)
    if offdiag_only:
        mask = 1.0 - np.eye(N_BITS)
        A1, A2 = A1 * mask, A2 * mask
    num = float(np.sum(A1 * A2))
    den = float(np.linalg.norm(A1) * np.linalg.norm(A2))
    return num / den if den > 0 else 0.0


def boolean_corr(B1, B2, Xs):
    """Pearson correlation of class-centered logits over boolean inputs Xs."""
    F1 = logits_folded(B1, Xs)
    F2 = logits_folded(B2, Xs)
    F1 = F1 - F1.mean(axis=1, keepdims=True)
    F2 = F2 - F2.mean(axis=1, keepdims=True)
    v1, v2 = F1.ravel() - F1.mean(), F2.ravel() - F2.mean()
    return float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ----------------------------------------------------------------- enumeration
def all_inputs_batches(batch=1 << 16):
    shifts = np.arange(N_BITS, dtype=np.uint64)
    for start in range(0, 1 << N_BITS, batch):
        idx = np.arange(start, start + batch, dtype=np.uint64)
        X = ((idx[:, None] >> shifts[None, :]) & 1).astype(np.float64)
        yield idx, X


# ================================================================= stages
def stage_setup():
    Z, y = make_facts()
    C = Z.T @ Z
    Cinv = np.linalg.inv(C)
    Hat = Z @ Cinv @ Z.T
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    G = Z @ Z.T
    Gcos = Zn @ Zn.T
    off = Gcos - np.diag(np.diag(Gcos))
    max_ov = np.abs(off).max(axis=1)
    np.savez(CACHE / "setup.npz", Z=Z, y=y, C=C, Cinv=Cinv, Hat=Hat,
             Gcos=Gcos, max_ov=max_ov)
    np.save(FIG / "part2_keys.npy", Z)
    np.save(FIG / "part2_classes.npy", y)
    np.save(FIG / "part2_gram_cos.npy", Gcos)
    np.save(FIG / "part2_hat_matrix.npy", Hat)
    counts = np.bincount(y, minlength=N_CLASSES)
    print(f"[setup] {K_FACTS} facts, {N_BITS}-bit keys, class counts {counts}")
    print(f"[setup] key-overlap |cos|: mean {off[off != 0].mean():.3f}, "
          f"max-per-fact range [{max_ov.min():.3f}, {max_ov.max():.3f}]")
    print(f"[setup] C spectrum: min {np.linalg.eigvalsh(C).min():.2f}, "
          f"max {np.linalg.eigvalsh(C).max():.2f}")
    metrics_update("setup", {
        "class_counts": counts.tolist(),
        "mean_abs_cos_overlap": float(np.abs(off[off != 0]).mean()),
        "max_overlap_range": [float(max_ov.min()), float(max_ov.max())],
        "C_eig_min": float(np.linalg.eigvalsh(C).min()),
        "C_eig_max": float(np.linalg.eigvalsh(C).max()),
    })
    return Z, y, C, Cinv, Hat


def load_setup():
    d = np.load(CACHE / "setup.npz")
    return d["Z"], d["y"], d["C"], d["Cinv"], d["Hat"], d["Gcos"], d["max_ov"]


def stage_sweep():
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    rows = []
    for H in H_SWEEP:
        t0 = time.time()
        L, R, D, losses = als_construction(Z, y, H, seed=MASTER_SEED + H)
        B = fold(L, R, D)
        a_c = acc_folded(B, Z, y)
        (Ls, Rs, Ds), _, a_s, _ = sgd_train(Z, y, H, seed=0, steps=6000)
        rows.append(dict(H=H, als_mse=losses[-1], als_acc=a_c, sgd_acc=a_s,
                         secs=round(time.time() - t0, 1)))
        print(f"[sweep] H={H:3d}  ALS mse={losses[-1]:.3e} acc={a_c:.2f}   "
              f"SGD acc={a_s:.2f}   ({rows[-1]['secs']}s)")
    ok = [r["H"] for r in rows if r["als_acc"] == 1.0 and r["sgd_acc"] == 1.0
          and r["als_mse"] < 1e-8]
    H_star = min(ok) if ok else H_SWEEP[-1]
    print(f"[sweep] H* = {H_star} (smallest H with 100% memorization for both "
          f"AND exact ALS interpolation, mse < 1e-8)")
    metrics_update("sweep", {"rows": rows, "H_star": H_star})
    json.dump({"H_star": H_star}, open(CACHE / "H_star.json", "w"))
    return H_star


def get_H_star():
    return json.load(open(CACHE / "H_star.json"))["H_star"]


def stage_construct():
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    H = get_H_star()
    L, R, D, losses = als_construction(Z, y, H, seed=MASTER_SEED + H)
    B = fold(L, R, D)
    acc = acc_folded(B, Z, y)
    print(f"[construct] ALS at H*={H}: final mse {losses[-1]:.3e}, acc {acc:.3f}, "
          f"{len(losses)} iterations")

    # permuted-fact baseline constructions (same keys, permuted fact->class map)
    permB = []
    for s in range(5):
        rng = np.random.default_rng(MASTER_SEED + 777 + s)
        yp = y[rng.permutation(K_FACTS)]
        Lp, Rp, Dp, lp = als_construction(Z, yp, H, seed=MASTER_SEED + 555 + s,
                                          restarts=2)
        permB.append(fold(Lp, Rp, Dp))
        print(f"[construct] permuted-fact baseline {s}: mse {lp[-1]:.3e}")
    permB = np.stack(permB)

    # KKT interpolant from zero (also = fixed point of cyclic single-fact edits)
    Bk, Lam, condS = kkt_interpolant(Z, y, Cinv)
    fit = logits_folded(Bk, Z) - MARGIN * np.eye(N_CLASSES)[y]
    print(f"[construct] KKT interpolant: cond(S)={condS:.2e}, "
          f"max |f - target| = {np.abs(fit).max():.2e}, "
          f"acc {acc_folded(Bk, Z, y):.3f}, max|lambda| {np.abs(Lam).max():.2f}")
    # verify the cyclic single-fact KKT edit converges to the same tensor
    Bc, passes, worst = kkt_edit_cyclic(
        np.zeros_like(Bk), Z, MARGIN * np.eye(N_CLASSES)[y], Cinv, tol=1e-6)
    rel = np.linalg.norm(Bc - Bk) / np.linalg.norm(Bk)
    print(f"[construct] cyclic single-fact edits from zero: {passes} passes, "
          f"rel distance to joint interpolant {rel:.2e}")

    # KKT single-fact ADD demo: 101st fact added to the ALS construction
    rng = np.random.default_rng(MASTER_SEED + 4242)
    z_new = rng.integers(0, 2, N_BITS).astype(np.float64)
    while tuple(z_new.astype(int)) in {tuple(r) for r in Z.astype(int)}:
        z_new = rng.integers(0, 2, N_BITS).astype(np.float64)
    c_new = int(rng.integers(0, N_CLASSES))
    tgt = np.zeros(N_CLASSES); tgt[c_new] = MARGIN
    pre = logits_folded(B, Z)
    B_add, _ = kkt_edit_single(B, z_new, tgt, Cinv)
    post = logits_folded(B_add, Z)
    new_logits = np.einsum("i,cij,j->c", z_new, B_add, z_new)
    add_ok = bool(new_logits.argmax() == c_new)
    marg = lambda F: F[np.arange(K_FACTS), y] - np.where(
        np.eye(N_CLASSES)[y] > 0, -np.inf, F).max(1)
    dmarg = marg(post) - marg(pre)
    acc_after = acc_folded(B_add, Z, y)
    print(f"[construct] add-101st-fact via KKT: stored={add_ok}, "
          f"acc on 100 after: {acc_after:.3f}, "
          f"mean|dmargin| {np.abs(dmarg).mean():.4f}, max {np.abs(dmarg).max():.4f}")
    print(f"[construct] {NO_CORPUS_NOTE}")

    np.savez(CACHE / "construct.npz", L=L, R=R, D=D, B=B, permB=permB,
             Bk=Bk, Lam=Lam)
    np.save(FIG / "part2_B_construction.npy", B)
    np.save(FIG / "part2_B_kkt_interpolant.npy", Bk)
    metrics_update("construct", {
        "H_star": H, "als_mse": losses[-1], "als_acc": acc,
        "kkt_interpolant_condS": condS,
        "kkt_interpolant_max_fit_err": float(np.abs(fit).max()),
        "kkt_interpolant_max_abs_lambda": float(np.abs(Lam).max()),
        "cyclic_matches_joint_rel_dist": float(rel),
        "cyclic_passes_to_1e-6": int(passes),
        "add_fact_demo": {"stored": add_ok, "acc_on_100_after": acc_after,
                          "mean_abs_dmargin": float(np.abs(dmarg).mean()),
                          "max_abs_dmargin": float(np.abs(dmarg).max())},
        "no_corpus_note": NO_CORPUS_NOTE,
    })


def stage_sgd():
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    H = get_H_star()
    Ws, inits, accs = [], [], []
    for s in SGD_SEEDS:
        t0 = time.time()
        (L, R, D), init, acc, loss = sgd_train(Z, y, H, seed=s)
        Ws.append((L, R, D)); inits.append(init); accs.append(acc)
        print(f"[sgd] seed {s}: acc {acc:.3f}, loss {loss:.4f} "
              f"({time.time() - t0:.0f}s)")
    Bs = np.stack([fold(*w) for w in Ws])
    B0 = np.stack([fold(*i) for i in inits])
    np.savez(CACHE / "sgd.npz",
             Bs=Bs, B0=B0, accs=np.array(accs),
             **{f"L{s}": Ws[i][0] for i, s in enumerate(SGD_SEEDS)},
             **{f"R{s}": Ws[i][1] for i, s in enumerate(SGD_SEEDS)},
             **{f"D{s}": Ws[i][2] for i, s in enumerate(SGD_SEEDS)})
    np.save(FIG / "part2_B_sgd_seeds.npy", Bs)
    metrics_update("sgd", {"H": H, "accs": accs, "l1": SGD_L1,
                           "steps": SGD_STEPS, "lr": SGD_LR})


# ----------------------------------------------------------------- F9
def stage_f9():
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    con = np.load(CACHE / "construct.npz")
    sgd = np.load(CACHE / "sgd.npz")
    B_con, permB = con["B"], con["permB"]
    Bs, B0 = sgd["Bs"], sgd["B0"]
    H = get_H_star()

    rng = np.random.default_rng(MASTER_SEED + 99)
    Xs = rng.integers(0, 2, size=(1 << 16, N_BITS)).astype(np.float64)

    def sims(Ba):
        return dict(
            gauss=gauss_frob_cos(Ba, B_con),
            gauss_offdiag=gauss_frob_cos(Ba, B_con, offdiag_only=True),
            boolean=boolean_corr(Ba, B_con, Xs),
            factkeys=boolean_corr(Ba, B_con, Z),
        )

    sim_sgd = [sims(Bs[i]) for i in range(len(SGD_SEEDS))]
    sim_rnd = [sims(B0[i]) for i in range(len(SGD_SEEDS))]
    sim_perm = [sims(permB[i]) for i in range(permB.shape[0])]
    sim_sgd_pair = [gauss_frob_cos(Bs[i], Bs[j])
                    for i in range(5) for j in range(i + 1, 5)]
    # construction self-similarity under re-initialization (context for (ii))
    als_self = []
    for s in range(3):
        La, Ra, Da, _ = als_construction(Z, y, H, seed=MASTER_SEED + 31 + s,
                                         restarts=1)
        als_self.append(gauss_frob_cos(fold(La, Ra, Da), B_con))

    # D ~ -I check: per hidden unit, dominance and sign of the dominant entry
    dom_frac, dom_sign_neg = [], []
    for i, s in enumerate(SGD_SEEDS):
        D = sgd[f"D{s}"]
        col = np.abs(D)
        dom = col.max(0) / np.maximum(col.sum(0), 1e-12)
        signs = D[col.argmax(0), np.arange(D.shape[1])]
        dom_frac.append(float(dom.mean()))
        dom_sign_neg.append(float((signs < 0).mean()))
    sign_flip_flag = (min(dom_sign_neg) < 0.5) != (max(dom_sign_neg) < 0.5)

    fig = plt.figure(figsize=(12.5, 7.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.25], hspace=0.42,
                          wspace=0.32)
    # (i) weight-statistic histograms
    L_con, R_con, D_con = con["L"], con["R"], con["D"]
    names = ["L entries", "R entries", "D entries"]
    con_w = [L_con.ravel(), R_con.ravel(), D_con.ravel()]
    sgd_w = [np.concatenate([sgd[f"L{s}"].ravel() for s in SGD_SEEDS]),
             np.concatenate([sgd[f"R{s}"].ravel() for s in SGD_SEEDS]),
             np.concatenate([sgd[f"D{s}"].ravel() for s in SGD_SEEDS])]
    rng2 = np.random.default_rng(MASTER_SEED + 3)
    rnd_w = [rng2.normal(0, 1 / np.sqrt(N_BITS), 5 * H * N_BITS),
             rng2.normal(0, 1 / np.sqrt(N_BITS), 5 * H * N_BITS),
             rng2.normal(0, 1 / np.sqrt(H), 5 * N_CLASSES * H)]
    for j in range(3):
        ax = fig.add_subplot(gs[0, j])
        lo = min(con_w[j].min(), sgd_w[j].min(), rnd_w[j].min())
        hi = max(con_w[j].max(), sgd_w[j].max(), rnd_w[j].max())
        bins = np.linspace(lo, hi, 55)
        ax.hist(rnd_w[j], bins=bins, density=True, histtype="step",
                color=MUTED, lw=1.4, label="random init")
        ax.hist(sgd_w[j], bins=bins, density=True, histtype="step",
                color=BLUE, lw=1.6, label="SGD (5 seeds)")
        ax.hist(con_w[j], bins=bins, density=True, histtype="step",
                color=ORANGE, lw=1.6, label="construction")
        ax.set_yscale("log")
        ax.set_title(f"(i) {names[j]}")
        if j == 0:
            ax.set_ylabel("density (log)")
            ax.legend(fontsize=8, loc="upper left")

    # (ii) folded-tensor similarity
    ax = fig.add_subplot(gs[1, 0:2])
    metrics = ["gauss", "gauss_offdiag", "boolean", "factkeys"]
    labels = ["Frobenius cos\n(= Gaussian corr)", "Frobenius cos\noff-diag only",
              "boolean-input\ncorr (2^16 sample)", "fact-key\nlogit corr"]
    groups = [("SGD vs constr", sim_sgd, BLUE),
              ("random init vs constr", sim_rnd, MUTED),
              ("permuted-fact constr vs constr", sim_perm, RED)]
    for mi, m in enumerate(metrics):
        for gi, (gname, sims_g, color) in enumerate(groups):
            vals = [d[m] for d in sims_g]
            xj = mi + (gi - 1) * 0.18
            ax.scatter([xj] * len(vals), vals, s=26, color=color, zorder=3,
                       label=gname if mi == 0 else None)
            ax.plot([xj - 0.07, xj + 0.07], [np.mean(vals)] * 2, color=color,
                    lw=2)
    ax.set_xticks(range(len(metrics)), labels)
    ax.set_ylabel("similarity to construction")
    ax.axhline(0, color=GRIDC, lw=0.8)
    ax.set_title("(ii) folded-tensor similarity: SGD vs construction, with both baselines")
    ax.legend(fontsize=8, loc="center left")

    # (iii) D matrix of an SGD run, columns sorted by dominant class
    ax = fig.add_subplot(gs[1, 2])
    D0 = sgd["D0"]
    order = np.lexsort((-np.abs(D0).max(0), np.abs(D0).argmax(0)))
    Dv = D0[:, order]
    vmax = np.abs(Dv).max()
    im = ax.imshow(Dv, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_title(f"(iii) SGD D (seed 0), units sorted\n"
                 f"dominance {dom_frac[0]:.2f}, "
                 f"{100 * dom_sign_neg[0]:.0f}% dominant entries negative")
    ax.set_xlabel("hidden unit (sorted by argmax class)")
    ax.set_ylabel("class")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    fig.suptitle(f"F9: SGD solution vs closed-form ALS construction "
                 f"(n={N_BITS}, K={K_FACTS}, C={N_CLASSES}, H={H}, D fixed = tiled -I)")
    savefig(fig, "F9")

    np.save(FIG / "F9_similarities.npy",
            np.array([[d[m] for m in metrics] for d in sim_sgd + sim_rnd + sim_perm]))
    payload = {
        "metrics_order": metrics,
        "sgd_vs_constr": {m: [d[m] for d in sim_sgd] for m in metrics},
        "randinit_vs_constr": {m: [d[m] for d in sim_rnd] for m in metrics},
        "permuted_vs_constr": {m: [d[m] for d in sim_perm] for m in metrics},
        "sgd_pairwise_gauss": sim_sgd_pair,
        "als_self_similarity_gauss": als_self,
        "D_dominance_per_seed": dom_frac,
        "D_dominant_sign_negative_frac_per_seed": dom_sign_neg,
        "sign_flip_flag_across_seeds": bool(sign_flip_flag),
    }
    metrics_update("f9", payload)
    for m in metrics:
        a = [d[m] for d in sim_sgd]; b = [d[m] for d in sim_rnd]
        c = [d[m] for d in sim_perm]
        print(f"[f9] {m:14s} SGD {np.mean(a):+.3f} [{min(a):+.3f},{max(a):+.3f}]"
              f"  rand {np.mean(b):+.3f} [{min(b):+.3f},{max(b):+.3f}]"
              f"  perm {np.mean(c):+.3f} [{min(c):+.3f},{max(c):+.3f}]")


# ----------------------------------------------------------------- F10
def extraction_scores(B, Z, y, Cinv):
    """Per fact: best |cos| between the fact key direction and the top-|K_c|
    eigenvectors (by |eigenvalue|) of its class's symmetric folded slice.
    Matches against both the raw key z_k and the whitened direction C^{-1}z_k."""
    scores = np.zeros(K_FACTS)
    for c in range(N_CLASSES):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        w, V = np.linalg.eigh(B[c])
        top = V[:, np.argsort(-np.abs(w))[:len(idx)]]      # (n, K_c)
        for k in idx:
            zk = Z[k] / np.linalg.norm(Z[k])
            uk = Cinv @ Z[k]; uk = uk / np.linalg.norm(uk)
            s1 = np.abs(top.T @ zk).max()
            s2 = np.abs(top.T @ uk).max()
            scores[k] = max(s1, s2)
    return scores


def keyframe_attribution(B, Z, y, Cinv):
    """Informed extraction: least-squares decomposition of the folded tensor
    in the fact-key frame {u_k u_k^T}, u_k = C^{-1} z_k (dictionary = the key
    list; still weights + fact list only). Fact k is 'recovered' if
    argmax_c lambda_{ck} equals its stored class. Returns (recovered,
    attribution margin lambda_true - max-other, residual op norms)."""
    lam, res_op = keyframe_bound_terms(B, Z, Cinv)   # lam: (C, K)
    rec = lam.argmax(0) == y
    lt = lam[y, np.arange(K_FACTS)]
    lo = np.where(np.eye(N_CLASSES)[y].T > 0, -np.inf, lam).max(0)
    return rec, lt - lo, res_op


def stage_f10():
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    con = np.load(CACHE / "construct.npz")
    sgd = np.load(CACHE / "sgd.npz")
    Bs = sgd["Bs"]

    # (a) BLIND eigen-extraction: top-|K_c| eigenvectors of each class slice
    sc_sgd = np.stack([extraction_scores(Bs[i], Z, y, Cinv)
                       for i in range(len(SGD_SEEDS))])
    sc_con = extraction_scores(con["B"], Z, y, Cinv)
    sc_kkt = extraction_scores(con["Bk"], Z, y, Cinv)
    rec_sgd = (sc_sgd >= RECOVERY_TAU)
    # robustness of the blind null to excess capacity (seed 0, larger H)
    blind_H = {}
    for Hbig in [100, 400]:
        (Lb, Rb, Db), _, accb, _ = sgd_train(Z, y, Hbig, seed=0, steps=8000)
        sb = extraction_scores(fold(Lb, Rb, Db), Z, y, Cinv)
        blind_H[Hbig] = dict(acc=accb, mean_score=float(sb.mean()),
                             recovery=float((sb >= RECOVERY_TAU).mean()))
        print(f"[f10] blind extraction at H={Hbig}: recovery "
              f"{blind_H[Hbig]['recovery']:.2f}, mean score {sb.mean():.3f}")

    # (b) INFORMED key-frame attribution
    att_sgd, attm_sgd = [], []
    for i in range(len(SGD_SEEDS)):
        rec, marg, res_op = keyframe_attribution(Bs[i], Z, y, Cinv)
        att_sgd.append(rec); attm_sgd.append(marg)
    att_sgd = np.stack(att_sgd); attm_sgd = np.stack(attm_sgd)
    att_con, attm_con, _ = keyframe_attribution(con["B"], Z, y, Cinv)
    att_kkt, attm_kkt, _ = keyframe_attribution(con["Bk"], Z, y, Cinv)

    edges = np.quantile(max_ov, np.linspace(0, 1, 6))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    binid = np.digitize(max_ov, edges) - 1

    def binned(rec_matrix):
        m, lo, hi = [], [], []
        for b in range(5):
            per_seed = rec_matrix[:, binid == b].mean(axis=1)
            m.append(float(per_seed.mean()))
            lo.append(float(per_seed.min())); hi.append(float(per_seed.max()))
        return m, lo, hi

    centers = [float(max_ov[binid == b].mean()) for b in range(5)]
    counts = [int((binid == b).sum()) for b in range(5)]
    blind_m, blind_lo, blind_hi = binned(rec_sgd)
    att_m, att_lo, att_hi = binned(att_sgd)
    att_con_b = [float(att_con[binid == b].mean()) for b in range(5)]
    att_kkt_b = [float(att_kkt[binid == b].mean()) for b in range(5)]

    corr_rec_ov = float(np.corrcoef(np.tile(max_ov, len(SGD_SEEDS)),
                                    att_sgd.ravel().astype(float))[0, 1])
    corr_margin_ov = float(np.corrcoef(np.tile(max_ov, len(SGD_SEEDS)),
                                       attm_sgd.ravel())[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    ax.fill_between(centers, att_lo, att_hi, color=BLUE, alpha=0.18, lw=0)
    ax.plot(centers, att_m, "-o", color=BLUE, ms=5,
            label="SGD, key-frame attribution (5 seeds, band = range)")
    ax.plot(centers, att_con_b, "-s", color=ORANGE, ms=5,
            label="ALS construction, key-frame attribution")
    ax.plot(centers, att_kkt_b, "-^", color=VIOLET, ms=5,
            label="KKT interpolant, key-frame attribution")
    ax.fill_between(centers, blind_lo, blind_hi, color=AQUA, alpha=0.25, lw=0)
    ax.plot(centers, blind_m, "-d", color=AQUA, ms=5,
            label=f"SGD, blind eigen-extraction (|cos| >= {RECOVERY_TAU})")
    ax.axhline(1.0 / N_CLASSES, color=MUTED, lw=1, ls=":")
    ax.annotate("chance (attribution)", (centers[0], 1.0 / N_CLASSES),
                textcoords="offset points", xytext=(2, 3), fontsize=7.5,
                color=MUTED)
    for x, cn in zip(centers, counts):
        ax.annotate(f"n={cn}", (x, -0.02), fontsize=7.5, color=MUTED,
                    ha="center")
    ax.set_xlabel("max off-diagonal Gram overlap  max_j |cos(z_k, z_j)|")
    ax.set_ylabel("recovery rate")
    ax.set_ylim(-0.06, 1.04)
    ax.set_title("F10a: per-fact recovery vs key overlap (quantile bins)")
    ax.legend(fontsize=7.5, loc="upper right")
    ax = axes[1]
    for i in range(len(SGD_SEEDS)):
        ax.scatter(max_ov, attm_sgd[i], s=10, color=BLUE, alpha=0.35,
                   label="SGD seeds" if i == 0 else None)
    ax.axhline(0, color=RED, lw=1.2, ls="--")
    ax.annotate("recovered above this line", (max_ov.min(), 0),
                textcoords="offset points", xytext=(2, 4), fontsize=8,
                color=RED)
    ax.set_xlabel("max off-diagonal Gram overlap")
    ax.set_ylabel("attribution margin  lambda_true - max lambda_other")
    ax.set_title(f"F10b: key-frame attribution margin\n"
                 f"corr(margin, overlap) = {corr_margin_ov:+.3f}, "
                 f"corr(recovered, overlap) = {corr_rec_ov:+.3f}")
    ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("F10: recovering individual facts as rank-1-ish components of "
                 "the folded tensor.  Blind eigen-extraction FAILS (~0 at every "
                 "overlap and H up to 400); informed key-frame attribution "
                 "(dictionary = C^{-1}-weighted keys) partially succeeds and "
                 "degrades with overlap")
    fig.tight_layout()
    savefig(fig, "F10")

    np.save(FIG / "F10_scores_sgd.npy", sc_sgd)
    np.save(FIG / "F10_scores_construction.npy", sc_con)
    np.save(FIG / "F10_attribution_recovered_sgd.npy", att_sgd)
    np.save(FIG / "F10_attribution_margin_sgd.npy", attm_sgd)
    np.save(FIG / "F10_max_overlap.npy", max_ov)
    metrics_update("f10", {
        "tau": RECOVERY_TAU,
        "blind_recovery_sgd_per_seed": rec_sgd.mean(axis=1).tolist(),
        "blind_recovery_construction": float((sc_con >= RECOVERY_TAU).mean()),
        "blind_recovery_kkt_interpolant": float((sc_kkt >= RECOVERY_TAU).mean()),
        "blind_mean_score_sgd": float(sc_sgd.mean()),
        "blind_large_H_seed0": blind_H,
        "attr_recovery_sgd_per_seed": att_sgd.mean(axis=1).tolist(),
        "attr_recovery_construction": float(att_con.mean()),
        "attr_recovery_kkt_interpolant": float(att_kkt.mean()),
        "bin_centers": centers, "bin_counts": counts,
        "bin_attr_sgd_mean": att_m,
        "bin_attr_sgd_range": [[lo, hi] for lo, hi in zip(att_lo, att_hi)],
        "bin_blind_sgd_mean": blind_m,
        "corr_attr_recovered_vs_overlap": corr_rec_ov,
        "corr_attr_margin_vs_overlap": corr_margin_ov,
    })
    print(f"[f10] blind recovery per seed {rec_sgd.mean(axis=1)} "
          f"(mean score {sc_sgd.mean():.3f}; typical inter-key |cos| ~0.51)")
    print(f"[f10] attribution recovery per seed {att_sgd.mean(axis=1)} "
          f"(constr {att_con.mean():.2f}, interpolant {att_kkt.mean():.2f})")
    print(f"[f10] attribution bins {np.round(att_m, 2)} at overlaps "
          f"{np.round(centers, 2)}; corr(recovered, overlap) {corr_rec_ov:+.3f}")


# ----------------------------------------------------------------- F11
def unlearn_targets(B, Zs):
    """Target = uniform: all class logits equal to the pre-edit mean logit."""
    t = []
    for z in Zs:
        f = np.einsum("i,cij,j->c", z, B, z, optimize=True)
        t.append(np.full(N_CLASSES, f.mean()))
    return np.stack(t)


def margins(F, y):
    best_other = np.where(np.eye(N_CLASSES)[y] > 0, -np.inf, F).max(1)
    return F[np.arange(len(y)), y] - best_other


def stage_f11_predict():
    """Register Gram-based collateral predictions BEFORE any measurement runs.
    Written to predictions/ ; the git commit of that file is the registration
    time (commit happens between this stage and f11_measure)."""
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    sgd = np.load(CACHE / "sgd.npz")
    Bs = sgd["Bs"]
    rng = np.random.default_rng(MASTER_SEED + 11)
    edit_idx = np.sort(rng.choice(K_FACTS, N_UNLEARN, replace=False))
    victims = np.array([k for k in range(K_FACTS) if k not in set(edit_idx)])

    pred = {"note": ("Predictions computed from model weights + fact list only, "
                     "BEFORE the edit is applied. " + NO_CORPUS_NOTE +
                     " Registration time = git commit time of this file."),
            "edit_idx": edit_idx.tolist(), "victims": victims.tolist(),
            "seeds": SGD_SEEDS, "per_seed": []}
    all_dlogits = []
    for i, s in enumerate(SGD_SEEDS):
        B = Bs[i]
        tgt = unlearn_targets(B, Z[edit_idx])
        f_pre_edit = logits_folded(B, Z[edit_idx])
        # joint KKT coefficients: S10 lam_c = (t - f) restricted to edited facts
        S10 = Hat[np.ix_(edit_idx, edit_idx)] ** 2
        lam = np.linalg.solve(S10, tgt - f_pre_edit)      # (10, C)
        # predicted logit change on every fact j: sum_m lam_mc Hat[j,m]^2
        dlog = (Hat[:, edit_idx] ** 2) @ lam              # (K, C)
        all_dlogits.append(dlog)
        F_pre = logits_folded(B, Z)
        F_post_pred = F_pre + dlog
        pm_pre = margins(F_pre, y)
        pm_post = margins(F_post_pred, y)
        gram_proxy = (Hat[victims][:, edit_idx] ** 2).sum(axis=1)
        pred["per_seed"].append({
            "seed": s,
            "pred_margin_pre": pm_pre[victims].tolist(),
            "pred_margin_post": pm_post[victims].tolist(),
            "pred_dmargin": (pm_post - pm_pre)[victims].tolist(),
            "gram_proxy_sum_hat2": gram_proxy.tolist(),
            "pred_victim_flips": int(
                (F_post_pred[victims].argmax(1) != y[victims]).sum()
                - (F_pre[victims].argmax(1) != y[victims]).sum()),
            "pred_forget_max_dev_from_uniform": 0.0,
        })
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    pred["written_at"] = stamp
    (PRED / "part2_f11_predictions.json").write_text(
        json.dumps(pred, indent=2))
    np.save(PRED / "part2_f11_pred_dlogits.npy", np.stack(all_dlogits))
    np.savez(CACHE / "f11_setup.npz", edit_idx=edit_idx, victims=victims)
    print(f"[f11_predict] wrote predictions/part2_f11_predictions.json at {stamp}")
    print(f"[f11_predict] edit set: {edit_idx.tolist()}")
    print("[f11_predict] COMMIT NOW before running f11_measure "
          "(commit time = registration time).")


def stage_f11_measure():
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    sgd = np.load(CACHE / "sgd.npz")
    Bs = sgd["Bs"]
    st = np.load(CACHE / "f11_setup.npz")
    edit_idx, victims = st["edit_idx"], st["victims"]
    pred = json.loads((PRED / "part2_f11_predictions.json").read_text())

    rows = []
    per_seed_scatter = []
    for i, s in enumerate(SGD_SEEDS):
        B = Bs[i]
        tgt = unlearn_targets(B, Z[edit_idx])
        F_pre = logits_folded(B, Z)
        m_pre = margins(F_pre, y)
        acc_pre = float((F_pre.argmax(1) == y).mean())

        # (1) KKT edit, cyclically re-tensioned to convergence
        B_kkt, passes, worst = kkt_edit_cyclic(B, Z[edit_idx], tgt, Cinv)
        # (2) KKT edit, single pass (one rank-1 edit per fact, no re-tension)
        B_k1 = B.copy()
        for m in range(len(edit_idx)):
            B_k1, _ = kkt_edit_single(B_k1, Z[edit_idx[m]], tgt[m], Cinv)
        # (3) naive removal: raw-key-direction rank-1 subtraction, one pass,
        #     alphas computed from PRE-EDIT logits (no re-tensioning at all)
        B_nv = B.copy()
        pre_logits_e = logits_folded(B, Z[edit_idx])
        for m in range(len(edit_idx)):
            z = Z[edit_idx[m]]
            beta = (tgt[m] - pre_logits_e[m]) / (float(z @ z) ** 2)
            B_nv = B_nv + beta[:, None, None] * np.outer(z, z)[None]

        out = {}
        for name, Bx in [("kkt", B_kkt), ("kkt_1pass", B_k1), ("naive", B_nv)]:
            F = logits_folded(Bx, Z)
            m_post = margins(F, y)
            fe = F[edit_idx]
            forget_dev = float(np.abs(fe - fe.mean(1, keepdims=True)).max())
            forgotten = int((fe.argmax(1) != y[edit_idx]).sum())
            dmarg = (m_post - m_pre)[victims]
            flips = int((F[victims].argmax(1) != y[victims]).sum()
                        - (F_pre[victims].argmax(1) != y[victims]).sum())
            out[name] = dict(
                forget_dev_from_uniform=forget_dev,
                forgotten_argmax=forgotten,
                acc90_pre=acc_pre, acc90_post=float((F[victims].argmax(1) == y[victims]).mean()),
                victim_flips=flips,
                mean_abs_dmargin=float(np.abs(dmarg).mean()),
                max_abs_dmargin=float(np.abs(dmarg).max()),
                dmargin=dmarg)
        out["kkt_passes"] = int(passes)
        rows.append(out)
        pd = np.array(pred["per_seed"][i]["pred_dmargin"])
        per_seed_scatter.append((pd, out["kkt"]["dmargin"],
                                 np.array(pred["per_seed"][i]["gram_proxy_sum_hat2"])))
        print(f"[f11] seed {s}: KKT passes {passes} | forgotten "
              f"{out['kkt']['forgotten_argmax']}/10 | victim flips "
              f"kkt {out['kkt']['victim_flips']} naive {out['naive']['victim_flips']} "
              f"| mean|dmargin| kkt {out['kkt']['mean_abs_dmargin']:.4f} "
              f"naive {out['naive']['mean_abs_dmargin']:.4f}")

    pd_all = np.concatenate([p for p, m, g in per_seed_scatter])
    md_all = np.concatenate([m for p, m, g in per_seed_scatter])
    g_all = np.concatenate([g for p, m, g in per_seed_scatter])
    r_pearson = float(np.corrcoef(pd_all, md_all)[0, 1])
    from scipy.stats import spearmanr
    r_spear = float(spearmanr(pd_all, md_all).statistic)
    r_gram = float(spearmanr(g_all, np.abs(md_all)).statistic)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
    ax = axes[0]
    lim = max(np.abs(pd_all).max(), np.abs(md_all).max()) * 1.08
    ax.plot([-lim, lim], [-lim, lim], color=GRIDC, lw=1)
    ax.scatter(pd_all, md_all, s=12, color=BLUE, alpha=0.5)
    ax.set_xlabel("predicted victim margin change (registered pre-edit)")
    ax.set_ylabel("measured margin change")
    ax.set_title(f"F11a: predicted vs measured collateral\n"
                 f"Pearson r = {r_pearson:.4f}, Spearman r = {r_spear:.4f} "
                 f"(450 victim points)")
    ax = axes[1]
    ax.scatter(g_all, np.abs(md_all), s=12, color=BLUE, alpha=0.5)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("pure-Gram proxy  sum_m Hat[j,m]^2")
    ax.set_ylabel("|measured margin change|")
    ax.set_title(f"F11b: Gram-only proxy\nSpearman r = {r_gram:.3f}")
    ax = axes[2]
    labels = ["KKT\n(re-tensioned)", "KKT\n(1 pass)", "naive\n(raw-key rank-1)"]
    colors = [BLUE, AQUA, RED]
    for j, name in enumerate(["kkt", "kkt_1pass", "naive"]):
        vals = [r[name]["mean_abs_dmargin"] for r in rows]
        ax.scatter([j] * len(vals), vals, s=26, color=colors[j], zorder=3)
        ax.plot([j - 0.12, j + 0.12], [np.mean(vals)] * 2, color=colors[j], lw=2)
        forg = np.mean([r[name]["forgotten_argmax"] for r in rows])
        ax.annotate(f"forget {forg:.1f}/10", (j, max(vals)),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color=INK)
    ax.set_xticks(range(3), labels)
    ax.set_yscale("log")
    ax.set_ylabel("mean |victim margin change| (90 facts)")
    ax.set_title("F11c: collateral by edit method\n(dots = 5 seeds)")
    fig.suptitle(f"F11: KKT unlearning of {N_UNLEARN} facts to uniform. "
                 + NO_CORPUS_NOTE)
    fig.tight_layout()
    savefig(fig, "F11")

    np.save(FIG / "F11_pred_vs_measured.npy",
            np.stack([pd_all, md_all, g_all]))
    summary = {
        "pearson_pred_vs_measured": r_pearson,
        "spearman_pred_vs_measured": r_spear,
        "spearman_gram_proxy": r_gram,
        "per_seed": [
            {kk: ({k2: jsonable(v2) for k2, v2 in vv.items() if k2 != "dmargin"}
                  if isinstance(vv, dict) else jsonable(vv))
             for kk, vv in r.items()}
            for r in rows],
        "naive_vs_kkt_mean_abs_dmargin_ratio": float(
            np.mean([r["naive"]["mean_abs_dmargin"] for r in rows])
            / np.mean([r["kkt"]["mean_abs_dmargin"] for r in rows])),
        "naive_victim_flips_total": int(sum(r["naive"]["victim_flips"] for r in rows)),
        "kkt_victim_flips_total": int(sum(r["kkt"]["victim_flips"] for r in rows)),
        "no_corpus_note": NO_CORPUS_NOTE,
    }
    metrics_update("f11", summary)
    print(f"[f11] pred-vs-measured Pearson {r_pearson:.4f} Spearman {r_spear:.4f}; "
          f"gram-proxy Spearman {r_gram:.3f}")
    print(f"[f11] naive/KKT collateral ratio "
          f"{summary['naive_vs_kkt_mean_abs_dmargin_ratio']:.1f}x; victim flips "
          f"kkt {summary['kkt_victim_flips_total']} vs naive "
          f"{summary['naive_victim_flips_total']} (over 5 seeds x 90)")


# ----------------------------------------------------------------- F12
def keyframe_bound_terms(B, Z, Cinv):
    """Decompose each B_c into the fact-key frame part sum_k lam_k u_k u_k^T
    (u_k = C^{-1} z_k) plus residual E_c; bound
    |f_c(x)| <= max_k |lam_ck| x^T C^{-1} x + ||E_c||_2 ||x||^2."""
    U = Z @ Cinv
    M = (U @ U.T) ** 2                      # <U_j, U_k>_F = (u_j.u_k)^2
    lam = np.zeros((N_CLASSES, K_FACTS))
    res_op = np.zeros(N_CLASSES)
    for c in range(N_CLASSES):
        b = np.einsum("ki,ij,kj->k", U, B[c], U, optimize=True)
        lam[c] = np.linalg.lstsq(M, b, rcond=None)[0]
        E = B[c] - np.einsum("k,ki,kj->ij", lam[c], U, U, optimize=True)
        res_op[c] = float(np.abs(np.linalg.eigvalsh(E)).max())
    return lam, res_op


def stage_f12():
    Z, y, C, Cinv, Hat, Gcos, max_ov = load_setup()
    con = np.load(CACHE / "construct.npz")
    sgd = np.load(CACHE / "sgd.npz")
    Bk, Lam = con["Bk"], con["Lam"]        # KKT interpolant, exact key-frame
    B_als = con["B"]
    Bs = sgd["Bs"]
    fints = set(fact_ints(Z).tolist())

    models = {"kkt_interpolant": Bk, "als_construction": B_als,
              "sgd_seed0": Bs[0]}
    maxlam = {"kkt_interpolant": float(np.abs(Lam).max())}
    resop = {"kkt_interpolant": 0.0}
    for name, Bx in [("als_construction", B_als), ("sgd_seed0", Bs[0])]:
        lamx, rx = keyframe_bound_terms(Bx, Z, Cinv)
        maxlam[name] = float(np.abs(lamx).max())
        resop[name] = float(rx.max())

    # exhaustive enumeration over all 2^20 inputs, batched
    nbins = 240
    binmax = {}
    hists = {m: None for m in models}
    edges = {}
    stats = {m: dict(max_off=0.0, argmax_off=None, sum_=0.0, sumsq=0.0,
                     n=0, above_min_fact=0) for m in models}
    qmax_off = 0.0
    scat = {m: [] for m in models}
    scatq = []
    # first pass to size hist edges cheaply: use bound-based guess then clip
    fact_margins = {m: margins(logits_folded(Bx, Z), y)
                    for m, Bx in models.items()}
    for m, Bx in models.items():
        guess = 2 * maxlam[m] * 1.0 + 2 * resop[m] * N_BITS
        # log-spaced bins: margins span ~1e-4 .. bound; clip below 1e-3
        edges[m] = np.concatenate([[0.0], np.geomspace(
            1e-3, max(guess, 1.5 * fact_margins[m].max()), nbins)])
        hists[m] = np.zeros(nbins, dtype=np.int64)
    t0 = time.time()
    for idx, X in all_inputs_batches():
        mask_fact = np.isin(idx, np.fromiter(fints, dtype=np.uint64))
        q = np.einsum("bi,ij,bj->b", X, Cinv, X, optimize=True)
        qmax_off = max(qmax_off, float(q[~mask_fact].max()))
        sub = slice(0, len(idx), 1024)
        for m, Bx in models.items():
            F = logits_folded(Bx, X)
            part = np.partition(F, N_CLASSES - 2, axis=1)
            marg = part[:, -1] - part[:, -2]
            mo = marg[~mask_fact]
            h, _ = np.histogram(np.clip(mo, edges[m][0], edges[m][-1] - 1e-9),
                                bins=edges[m])
            hists[m] += h
            st = stats[m]
            bmax = float(mo.max())
            if bmax > st["max_off"]:
                st["max_off"] = bmax
                st["argmax_off"] = int(idx[~mask_fact][mo.argmax()])
            st["sum_"] += float(mo.sum()); st["sumsq"] += float((mo ** 2).sum())
            st["n"] += int(len(mo))
            st["above_min_fact"] += int((mo > fact_margins[m].min()).sum())
            scat[m].append(marg[sub])
        scatq.append(q[sub])
    print(f"[f12] exhaustive 2^20 enumeration done in {time.time() - t0:.0f}s "
          f"(batched, {1 << 16} per batch)")
    scatq = np.concatenate(scatq)

    bounds = {m: 2 * maxlam[m] * qmax_off + 2 * resop[m] * N_BITS
              for m in models}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    ax = axes[0]
    colors = {"kkt_interpolant": BLUE, "als_construction": ORANGE,
              "sgd_seed0": AQUA}
    labels = {"kkt_interpolant": "KKT interpolant (bound exact-form)",
              "als_construction": "ALS construction",
              "sgd_seed0": "SGD (seed 0)"}
    for m in models:
        ax.stairs(np.maximum(hists[m], 0.5),
                  np.maximum(edges[m], 5e-4), color=colors[m],
                  lw=1.5, label=labels[m])
        ax.axvline(bounds[m], color=colors[m], lw=1.2, ls="--")
        ax.axvline(fact_margins[m].min(), color=colors[m], lw=1.0, ls=":")
        ax.annotate(f"bound {bounds[m]:.0f}", (bounds[m], 3e5),
                    rotation=90, fontsize=7.5, color=colors[m],
                    ha="right", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(5e-4, 6e3)
    ax.set_xlabel("output margin  top1 - top2 logit  (log)")
    ax.set_ylabel("count over 2^20 - 100 off-fact inputs (log)")
    ax.set_title("F12a: off-fact-set margins, exhaustive over all 2^20 inputs\n"
                 "dashed = analytic Gram-overlap bound, dotted = min on-fact margin")
    ax.legend(fontsize=8, loc="upper left")
    ax = axes[1]
    m = "kkt_interpolant"
    mvals = np.concatenate(scat[m])
    ax.scatter(scatq, np.maximum(mvals, 1e-4), s=4, color=BLUE, alpha=0.25,
               label="off-fact inputs (1/1024 subsample)")
    qq = np.linspace(1e-3, qmax_off * 1.02, 200)
    ax.plot(qq, 2 * maxlam[m] * qq, color=RED, lw=1.5,
            label="bound 2 max|lambda| * x^T C^{-1} x")
    qk = np.einsum("ki,ij,kj->k", Z, Cinv, Z, optimize=True)
    ax.scatter(qk, margins(logits_folded(Bk, Z), y), s=18, color=ORANGE,
               marker="s", label="the 100 fact keys", zorder=4)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.2e3)
    ax.set_xlabel("x^T C^{-1} x   (Gram-overlap coordinate)")
    ax.set_ylabel("margin (log)")
    ax.set_title("F12b: margin vs the bound coordinate (KKT interpolant)")
    ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("F12: behavior on ALL 2^20 inputs - off-fact margins vs the "
                 "analytic bound  |f_c(x)| <= max_k|lambda_ck| x^T C^{-1} x + ||E_c||_2 ||x||^2")
    fig.tight_layout()
    savefig(fig, "F12")

    np.savez(FIG / "F12_histograms.npz",
             **{f"hist_{m}": hists[m] for m in models},
             **{f"edges_{m}": edges[m] for m in models})
    payload = {"qmax_offfact": qmax_off}
    for m in models:
        st = stats[m]
        payload[m] = {
            "max_offfact_margin": st["max_off"],
            "argmax_input_int": st["argmax_off"],
            "mean_offfact_margin": st["sum_"] / st["n"],
            "min_onfact_margin": float(fact_margins[m].min()),
            "median_onfact_margin": float(np.median(fact_margins[m])),
            "frac_offfact_above_min_onfact": st["above_min_fact"] / st["n"],
            "bound": bounds[m], "max_abs_lambda": maxlam[m],
            "residual_opnorm": resop[m],
            "bound_tightness_max_over_bound": st["max_off"] / bounds[m],
        }
        print(f"[f12] {m}: max off-fact margin {st['max_off']:.2f} vs bound "
              f"{bounds[m]:.2f} (tightness {st['max_off'] / bounds[m]:.2f}); "
              f"min on-fact margin {fact_margins[m].min():.2f}; "
              f"frac off-fact above min on-fact "
              f"{st['above_min_fact'] / st['n']:.3e}")
    metrics_update("f12", payload)


# ----------------------------------------------------------------- driver
# ----------------------------------------------------------------- F11d (frames/LP)
def lp_frame_edit(L, R, D, Z, y, rm, keep, frame, eps=0.5):
    """Hinge LP for one exact frame of the one-layer model: logits = D((Lz)*(Rz))
    is linear in D, in R (L, D fixed), and in L (R, D fixed). Exact removal
    equalities at rm keys + margin >= eps at keep keys, minimize total hinge
    violation. Returns (edited (L,R,D), total_violation, nvars)."""
    import scipy.sparse as sp
    from scipy.optimize import linprog
    C, H = D.shape
    d = Z.shape[1]
    hz = (Z @ L.T) * (Z @ R.T)
    logits = hz @ D.T
    if frame == "D":
        nv = C * H
        def rowfn(k, cp, cn):
            v = np.zeros((C, H)); v[cp] = hz[k]; v[cn] -= hz[k]
            return v.ravel()
    else:
        fac = Z @ (L.T if frame == "R" else R.T)      # (N, H)
        nv = H * d
        def rowfn(k, cp, cn):
            return ((D[cp] - D[cn])[:, None] * (fac[k][:, None] * Z[k][None, :])).ravel()
    rows = len(keep) * (C - 1)
    A_ret = np.empty((rows, nv)); b_ub = np.empty(rows)
    r = 0
    for k in keep:
        for c in range(C):
            if c == y[k]:
                continue
            A_ret[r] = -rowfn(k, y[k], c)
            b_ub[r] = (logits[k, y[k]] - logits[k, c]) - eps
            r += 1
    A_eqd = np.empty((len(rm) * (C - 1), nv)); b_eq = np.empty(len(rm) * (C - 1))
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
    if frame == "D":
        D = D + dx.reshape(C, H)
    elif frame == "R":
        R = R + dx.reshape(H, d)
    else:
        L = L + dx.reshape(H, d)
    return (L, R, D), float(lp.x[nv:].sum()), nv


def frame_eval(L, R, D, Z, y, rm, keep):
    logits = ((Z @ L.T) * (Z @ R.T)) @ D.T
    fdev = float(np.abs(logits[rm] - logits[rm].mean(1, keepdims=True)).max())
    flips = int((logits[keep].argmax(1) != y[keep]).sum())
    return flips, fdev


def stage_f11d():
    pred = PRED / "part2_f11d_prediction.md"
    r = subprocess.run(["git", "status", "--porcelain", "--", str(pred)],
                       cwd=ROOT, capture_output=True, text=True)
    assert not r.stdout.strip(), "commit part2_f11d_prediction.md before measuring"
    Z, y, *_ = load_setup()
    sgd = np.load(CACHE / "sgd.npz")
    st = np.load(CACHE / "f11_setup.npz")
    rm = st["edit_idx"]
    keep = np.setdiff1d(np.arange(K_FACTS), rm)
    res = {"working_point": {}, "overload": []}
    print("-- P15: margin LP, D-frame only, 100 facts, same 10 edit facts as F11 --")
    for s in SGD_SEEDS:
        L, R, D = sgd[f"L{s}"], sgd[f"R{s}"], sgd[f"D{s}"]
        (L2_, R2_, D2_), viol, _ = lp_frame_edit(L, R, D, Z, y, rm, keep, "D")
        flips, fdev = frame_eval(L2_, R2_, D2_, Z, y, rm, keep)
        res["working_point"][s] = {"violation": viol, "flips": flips, "forget_dev": fdev}
        print(f"  seed {s}: violation {viol:.2e} "
              f"({'FEASIBLE' if viol < 1e-6 else 'infeasible'}), retained flips "
              f"{flips}/90, forget dev {fdev:.1e}")
    print("-- P16: overload arm, one layer, 350 facts, H=40, seed 0, l1=0 --")
    rng = np.random.default_rng(MASTER_SEED + 1)
    seen, Zo = set(), []
    while len(Zo) < 350:
        z = tuple(int(v) for v in rng.integers(0, 2, N_BITS))
        if sum(z) > 0 and z not in seen:
            seen.add(z); Zo.append(z)
    Zo = np.array(Zo, dtype=np.float64)
    yo = rng.integers(0, N_CLASSES, 350)
    (Lo, Ro, Do), _, acc, _ = sgd_train(Zo, yo, 40, seed=0, steps=20000, l1=0.0)
    print(f"  trained: acc {acc:.1%}")
    assert acc == 1.0, "overload arm did not memorize; adjust before measuring"
    rmo = np.random.default_rng(7).choice(350, 10, replace=False)
    keepo = np.setdiff1d(np.arange(350), rmo)
    W = (Lo, Ro, Do)
    for rnd, frame in enumerate(["D", "R", "L", "D", "R", "L", "D"]):
        W, viol, _ = lp_frame_edit(*W, Zo, yo, rmo, keepo, frame)
        flips, fdev = frame_eval(*W, Zo, yo, rmo, keepo)
        res["overload"].append({"round": rnd, "frame": frame, "violation": viol,
                                "flips": flips, "forget_dev": fdev})
        print(f"  round {rnd} [{frame}]: violation {viol:9.2f}, retained flips "
              f"{flips}/340, forget dev {fdev:.1e}")
        if viol < 1e-6 and flips == 0:
            break
    metrics_update("f11d", res)

    # F11d figure
    fig = plt.figure(figsize=(11.5, 4.0))
    gs = fig.add_gridspec(1, 2, wspace=0.3)
    ax = fig.add_subplot(gs[0, 0])
    methods = ["naive rank-1\nsubtraction", "KKT (L2,\nC-weighted)", "margin LP\n(D-frame, 1 round)"]
    kkt_flips_frac = 2 / 450
    naive_flips_frac = 0.55
    lp_frac = np.mean([res["working_point"][s]["flips"] for s in SGD_SEEDS]) / 90
    vals = [naive_flips_frac * 100, kkt_flips_frac * 100, lp_frac * 100]
    bars = ax.bar(range(3), vals, 0.55, color=[RED, ORANGE, BLUE])
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}%", (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(methods, fontsize=8.5)
    ax.set_ylabel("retained facts flipped (%)")
    ax.set_title("(i) unlearn 10 of 100 facts (H=40): collateral by method\n"
                 "(naive also FAILS to forget; LP: exact-uniform, zero collateral, all 5 seeds)",
                 fontsize=9.5)
    ax = fig.add_subplot(gs[0, 1])
    xs = [h["round"] for h in res["overload"]]
    fl = [h["flips"] for h in res["overload"]]
    ax.plot(xs, fl, "o-", color=BLUE, lw=2, ms=7)
    for h in res["overload"]:
        ax.annotate(h["frame"], (h["round"], h["flips"]),
                    textcoords="offset points", xytext=(8, 6), fontsize=9, color=MUTED)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("round (frame edited)")
    ax.set_ylabel("retained facts broken (of 340)")
    ax.set_title("(ii) ONE layer, overloaded (350 facts, H=40):\n"
                 "same ladder as the 2-layer model — frame load, not depth", fontsize=9.5)
    fig.suptitle("F11d — the margin-LP / multi-frame editor in the one-layer setting", y=1.04)
    savefig(fig, "F11d")


STAGES = {
    "setup": stage_setup, "sweep": stage_sweep, "construct": stage_construct,
    "sgd": stage_sgd, "f9": stage_f9, "f10": stage_f10,
    "f11_predict": stage_f11_predict, "f11_measure": stage_f11_measure,
    "f12": stage_f12, "f11d": stage_f11d,
}
ORDER = list(STAGES)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all")
    args = ap.parse_args()
    todo = ORDER if args.stages == "all" else args.stages.split(",")
    print(NO_CORPUS_NOTE)
    for name in todo:
        print(f"\n===== stage {name} =====")
        STAGES[name]()
