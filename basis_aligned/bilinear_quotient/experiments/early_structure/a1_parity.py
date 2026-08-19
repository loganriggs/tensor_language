"""A1 — parity / XOR: kernel verification.

DGP. z = [k bits in ±1 coding | n_dist continuous distractors], embedded as
x = W z (random orthogonal, optionally anisotropic). Targets are the pairwise
parities of designated bit pairs, which in ±1 coding are exactly the degree-2
monomials  y_c = z_i z_j  — so the ground-truth forms are known exactly:
Q_c = ½(e_i e_j^T + e_j e_i^T) in z coordinates.

Two arms differing only in whether the *diagonal* of the lift is probed:
  pure   — bits are exactly ±1, so z_i² ≡ 1: the magnitude/diagonal coordinates
           of the lift are constant on-distribution and hence unidentifiable.
  graded — bits carry a random magnitude, z_i = b_i·m_i, y_c = z_i z_j: now the
           diagonal is probed to be zero.

Predictions (doc A1): (i) recovered Q match planted up to gauge; (ii) ker(W)
contains the magnitude directions and every distractor direction; (iii) effective
rank of W equals the number of planted pairs. Verification is causal: perturbing
along the measured blind subspace must not move the output, and input pairs whose
*lift difference* is a kernel direction must produce identical outputs.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, interaction, train, effective_rank,
                       gauge_refactor, whiten, sqrtm_psd, input_moment, lam_relerr,
                       lam_cos)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

K_BITS = 8
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]
H = 32
STEPS = 12000
BATCH = 512


# ------------------------------------------------------------------ DGP

def make_embedding(d, seed, kappa=1.0):
    """Random orthogonal embedding, optionally ill-conditioned (anisotropy knob)."""
    g = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    if kappa != 1.0:
        s = torch.logspace(0, math.log10(kappa), d, dtype=W.dtype)
        W = W @ torch.diag(s)
    return W.to(DEV)


def sampler(W, n_dist, arm, noise=0.0, seed=0):
    d = K_BITS + n_dist
    g = torch.Generator(device=DEV).manual_seed(seed)

    def gen(batch):
        b = (torch.randint(0, 2, (batch, K_BITS), generator=g, device=DEV) * 2 - 1).to(W)
        if arm == 'graded':
            b = b * (0.5 + torch.rand(batch, K_BITS, generator=g, device=DEV).to(W))
        dist = torch.randn(batch, n_dist, generator=g, device=DEV).to(W) if n_dist else \
            torch.zeros(batch, 0, device=DEV, dtype=W.dtype)
        z = torch.cat([b, dist], 1)
        y = torch.stack([z[:, i] * z[:, j] for i, j in PAIRS], 1)
        if noise:
            y = y + noise * torch.randn_like(y)
        return z @ W.T, y, z

    return gen, d


def handcoded(d, W):
    """Exact minimal solution: one hidden unit per planted pair, reading e_i and
    e_j in z coordinates. In x coordinates the rows are W^{-T}e_i, so the model
    is weight-sparse exactly in the privileged (z) basis."""
    Winv = torch.linalg.inv(W)
    L = torch.zeros(len(PAIRS), d, device=DEV, dtype=W.dtype)
    R = torch.zeros(len(PAIRS), d, device=DEV, dtype=W.dtype)
    for k, (i, j) in enumerate(PAIRS):
        L[k] = Winv[i]                      # (W^{-1})_i· so that L@x = z_i
        R[k] = Winv[j]
    return {'L': L, 'R': R, 'D': torch.eye(len(PAIRS), device=DEV, dtype=W.dtype)}


def planted_forms(d):
    Q = torch.zeros(len(PAIRS), d, d, device=DEV)
    for c, (i, j) in enumerate(PAIRS):
        Q[c, i, j] = Q[c, j, i] = 0.5
    return Q


# ------------------------------------------------------------------ measurements

def forms_in_z(p, W):
    """Pull the learned forms back to z coordinates: y = z^T (W^T Q W) z."""
    Q = interaction(p)
    return torch.einsum('ai,cab,bj->cij', W, Q, W)


def blind_subspace(Q, thresh=1e-6):
    """N = ∩_c ker(Q_c): input directions the layer is exactly blind to,
    to first AND second order. Returns (basis (d,k), singular values)."""
    m, d, _ = Q.shape
    s = torch.linalg.svdvals(Q.reshape(m * d, d))
    U, sv, Vh = torch.linalg.svd(Q.reshape(m * d, d))
    keep = sv <= thresh * sv.max()
    return Vh[len(sv) - int(keep.sum()):].T if int(keep.sum()) else Vh[:0].T, sv


def subspace_overlap(A, B):
    """Mean captured energy of B's directions inside span(A), in [0,1]."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return float('nan')
    Qa = torch.linalg.qr(A)[0]
    Qb = torch.linalg.qr(B)[0]
    return float(((Qa.T @ Qb) ** 2).sum() / Qb.shape[1])


@torch.no_grad()
def causal_blindness(p, W, N_z, gen, n=4096, eps=1.0):
    """Perturb along the measured blind subspace (in z coords) vs along a random
    non-blind direction; report the RMS output change of each."""
    x, y, z = gen(n)
    d = z.shape[1]

    def perturb(dirs_z):
        k = dirs_z.shape[1]
        coef = torch.randn(n, k, device=DEV, dtype=z.dtype)
        coef = coef / coef.norm(dim=1, keepdim=True)
        dz = coef @ dirs_z.T
        return ((forward(p, (z + eps * dz) @ W.T) - forward(p, x)) ** 2).mean().sqrt()

    base = (forward(p, x) ** 2).mean().sqrt()
    blind = perturb(N_z) if N_z.shape[1] else torch.tensor(float('nan'))
    g = torch.Generator(device=DEV).manual_seed(0)
    rnd = torch.randn(d, 4, generator=g, device=DEV, dtype=z.dtype)
    if N_z.shape[1]:                       # project OUT the blind part
        Qn = torch.linalg.qr(N_z)[0]
        rnd = rnd - Qn @ (Qn.T @ rnd)
    rnd = rnd / rnd.norm(dim=0, keepdim=True)
    return {'rms_out': float(base), 'rms_delta_blind': float(blind),
            'rms_delta_active': float(perturb(rnd)),
            'blindness_ratio': float(blind / perturb(rnd))}


@torch.no_grad()
def lift_kernel_pairs(p, W, n=4096):
    """Causal test of a *lift-space* kernel direction. For planted pair (i,j)
    take z with only coords i,j nonzero: (a, c) vs (a·t, c/t). The lift differs
    only in the two diagonal entries (a²-a²t², c²-c²/t²) — a direction predicted
    to lie in ker(W) — while the off-diagonal z_i z_j is identical. Prediction:
    identical outputs. Control: change the product instead."""
    d = W.shape[1]
    g = torch.Generator(device=DEV).manual_seed(1)
    i, j = PAIRS[0]
    a = torch.randn(n, generator=g, device=DEV, dtype=W.dtype)
    c = torch.randn(n, generator=g, device=DEV, dtype=W.dtype)
    t = 1.5

    def build(ai, ci):
        z = torch.zeros(n, d, device=DEV, dtype=W.dtype)
        z[:, i], z[:, j] = ai, ci
        return forward(p, z @ W.T)

    y0 = build(a, c)
    y_same = build(a * t, c / t)            # same product, different magnitudes
    y_diff = build(a * t, c)                # product scaled by t
    scale = y0.abs().mean().clamp_min(1e-30)
    return {'rel_change_same_product': float((y_same - y0).abs().mean() / scale),
            'rel_change_scaled_product': float((y_diff - y0).abs().mean() / scale),
            'predicted_change_same_product': 0.0,
            'predicted_change_scaled_product': t - 1.0}


def xcoord_analysis(p, W, x_all, n_dist, thresh=1e-3):
    """The realistic analysis position: the forms are seen in x coordinates and
    the embedding W is unknown. Under an anisotropic embedding a RAW magnitude
    threshold misreads which directions are small; the Λ metric whitened by Σ_x
    is the correct one. Blind directions of the whitened family map back as
    δ_x = Σ_x^{1/2} δ_w; the planted blind subspace in x is W·(distractor axes)."""
    d = W.shape[1]
    Qx = interaction(p)
    Sigma_x = input_moment(x_all)
    S = sqrtm_psd(Sigma_x)
    Qx_w = whiten(Qx, Sigma_x)
    planted_blind_x = W[:, K_BITS:] if n_dist else torch.zeros(d, 0, device=DEV, dtype=Qx.dtype)
    Nx_raw = blind_subspace(Qx, thresh)[0]
    Nx_wht = S @ blind_subspace(Qx_w, thresh)[0]
    return {
        'eff_rank_raw': effective_rank(Qx, thresh=thresh)[0],
        'eff_rank_whitened': effective_rank(Qx_w, thresh=thresh)[0],
        'blind_dim_raw': int(Nx_raw.shape[1]), 'blind_dim_whitened': int(Nx_wht.shape[1]),
        'blind_overlap_raw': subspace_overlap(Nx_raw, planted_blind_x) if n_dist else float('nan'),
        'blind_overlap_whitened': subspace_overlap(Nx_wht, planted_blind_x) if n_dist else float('nan'),
        'cond_Sigma_x': float(torch.linalg.cond(Sigma_x)),
    }


def diag_mass(Qz):
    dg = torch.diagonal(Qz, dim1=1, dim2=2).abs().sum()
    return float(dg / Qz.abs().sum().clamp_min(1e-30))


def planted_mass(Qz):
    """Fraction of |Q| mass on the planted (i,j) entries."""
    tot = Qz.abs().sum()
    inside = sum(Qz[c, i, j].abs() + Qz[c, j, i].abs() for c, (i, j) in enumerate(PAIRS))
    return float(inside / tot.clamp_min(1e-30))


def weight_level_stat(p, W):
    """A deliberately BASIS-READING statistic: how concentrated the rows of L
    (pulled back to z coordinates) are on the planted bit coordinates. This is
    the thing the gauge null must kill."""
    Lz = p['L'] @ W                                        # (h, d)
    mass = Lz[:, :K_BITS].abs().sum() / Lz.abs().sum().clamp_min(1e-30)
    top2 = Lz.abs().topk(2, dim=1).values.sum(1) / Lz.abs().sum(1).clamp_min(1e-30)
    return {'L_mass_on_bits': float(mass), 'L_row_top2_concentration': float(top2.mean())}


# ------------------------------------------------------------------ one run

def run(arm='graded', n_dist=4, kappa=1.0, noise=0.0, seed=0, shuffle_labels=False,
        train_steps=STEPS, verbose=True):
    d = K_BITS + n_dist
    W = make_embedding(d, seed=100 + seed, kappa=kappa)
    gen, _ = sampler(W, n_dist, arm, noise, seed=seed)

    def data_fn():
        x, y, _ = gen(BATCH)
        if shuffle_labels:
            y = y[torch.randperm(y.shape[0], device=DEV)]
        return x, y

    p = init_params(d, H, len(PAIRS), seed=seed, device=DEV)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
    p, hist, _ = train(p, data_fn, train_steps, lr=3e-3, checkpoint_every=train_steps // 4,
                       log_every=0, cosine=True)

    xe, ye, _ = gen(16384)
    fvu = float(((forward(p, xe) - ye) ** 2).mean() / ye.var())

    Qz = forms_in_z(p, W)
    Qtrue = planted_forms(d)
    x_all, _, z_all = gen(16384)
    Sigma_z = input_moment(z_all)

    r_raw, sv_raw = effective_rank(Qz, thresh=1e-3)
    r_wht, sv_wht = effective_rank(whiten(Qz, Sigma_z), thresh=1e-3)
    N_z, sv_blind = blind_subspace(Qz, thresh=1e-3)

    # The realistic analysis position: you see the model in x coordinates and do
    # NOT know the embedding W. Under an anisotropic embedding the raw metric
    # misreads what is small; the Λ metric whitened by Σ_x is the correct one.
    xcoord = xcoord_analysis(p, W, x_all, n_dist)
    planted_blind = torch.zeros(d, n_dist, device=DEV, dtype=Qz.dtype)
    for a in range(n_dist):
        planted_blind[K_BITS + a, a] = 1.0

    out = {
        'arm': arm, 'n_dist': n_dist, 'kappa': kappa, 'noise': noise, 'seed': seed,
        'shuffle_labels': shuffle_labels, 'fvu': fvu,
        'eff_rank_raw': r_raw, 'eff_rank_whitened': r_wht, 'planted_rank': len(PAIRS),
        'sv_raw': [float(v) for v in sv_raw[:8]],
        'blind_dim': int(N_z.shape[1]), 'planted_blind_dim': n_dist,
        'blind_overlap': subspace_overlap(N_z, planted_blind) if n_dist else float('nan'),
        'Q_cos_planted': [round(lam_cos(Qz[c:c + 1], Qtrue[c:c + 1]), 6) for c in range(len(PAIRS))],
        'Q_relerr_planted': lam_relerr(Qtrue, Qz * 1.0),
        'planted_entry_mass': planted_mass(Qz), 'diag_mass': diag_mass(Qz),
        'causal': causal_blindness(p, W, N_z, gen),
        'lift_pairs': lift_kernel_pairs(p, W),
        'weight_stat': weight_level_stat(p, W),
        'xcoord': xcoord,
    }
    if verbose:
        print(f"  [{arm} nd={n_dist} k={kappa} nz={noise} s={seed}"
              f"{' SHUF' if shuffle_labels else ''}] fvu {fvu:.2e} rank {r_raw}/{r_wht} "
              f"blind {out['blind_dim']}/{n_dist} ovl {out['blind_overlap']:.4f} "
              f"cosQ {min(out['Q_cos_planted']):.4f} diag {out['diag_mass']:.3f} "
              f"blindratio {out['causal']['blindness_ratio']:.2e}")
    return out, p, W, Qz


# ------------------------------------------------------------------ main

def main():
    t0 = time.time()
    res = {'config': {'k_bits': K_BITS, 'pairs': PAIRS, 'h': H, 'steps': STEPS,
                      'batch': BATCH, 'device': DEV}}

    print('== main arms (3 seeds each) ==')
    res['main'] = []
    for arm in ('pure', 'graded'):
        for seed in range(3):
            o, _, _, _ = run(arm=arm, n_dist=4, seed=seed)
            res['main'].append(o)

    print('== NULL 1: random weights (untrained) ==')
    res['null_random'] = []
    for seed in range(3):
        d = K_BITS + 4
        W = make_embedding(d, seed=100 + seed)
        gen, _ = sampler(W, 4, 'graded', seed=seed)
        p = init_params(d, H, len(PAIRS), seed=seed, device=DEV)
        p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
        Qz = forms_in_z(p, W)
        N_z, _ = blind_subspace(Qz, thresh=1e-3)
        pb = torch.zeros(d, 4, device=DEV, dtype=Qz.dtype)
        for a in range(4):
            pb[K_BITS + a, a] = 1.0
        o = {'seed': seed, 'eff_rank_raw': effective_rank(Qz, 1e-3)[0],
             'blind_dim': int(N_z.shape[1]),
             'blind_overlap': subspace_overlap(N_z, pb),
             'planted_entry_mass': planted_mass(Qz), 'diag_mass': diag_mass(Qz),
             'Q_cos_planted': [round(lam_cos(Qz[c:c + 1], planted_forms(d)[c:c + 1]), 6)
                               for c in range(len(PAIRS))],
             'weight_stat': weight_level_stat(p, W)}
        print(f"  [random s={seed}] rank {o['eff_rank_raw']} blind {o['blind_dim']} "
              f"planted_mass {o['planted_entry_mass']:.4f} cosQ {max(o['Q_cos_planted']):.3f}")
        res['null_random'].append(o)

    print('== NULL 2a: gauge scramble of the EXACT sparse solution ==')
    # The trained model is already dense in the privileged basis (h=32 >> the 4
    # units an exact solution needs), so there is no basis-reading finding in it
    # for the gauge null to kill. The hand-coded minimal solution is the object
    # where a weight reading DOES work — scramble that.
    res['null_gauge_handcoded'] = []
    for nd in (0, 4):
        d = K_BITS + nd
        W = make_embedding(d, seed=100)
        p_hc = handcoded(d, W)
        gen, _ = sampler(W, nd, 'graded', seed=0)
        xe, ye, _ = gen(8192)
        fvu_hc = float(((forward(p_hc, xe) - ye) ** 2).mean() / ye.var())
        Qz_hc = forms_in_z(p_hc, W)
        pg, resid = gauge_refactor(p_hc, seed=4242)
        Qz_g = forms_in_z(pg, W)
        o = {'n_dist': nd, 'fvu_handcoded': fvu_hc, 'refactor_residual': resid,
             'Q_relerr_hc_vs_gauge': lam_relerr(Qz_hc, Qz_g),
             'planted_entry_mass_hc': planted_mass(Qz_hc),
             'planted_entry_mass_gauged': planted_mass(Qz_g),
             'blind_dim_hc': int(blind_subspace(Qz_hc, 1e-3)[0].shape[1]),
             'blind_dim_gauged': int(blind_subspace(Qz_g, 1e-3)[0].shape[1]),
             'weight_stat_hc': weight_level_stat(p_hc, W),
             'weight_stat_gauged': weight_level_stat(pg, W)}
        print(f"  [nd={nd}] fvu {fvu_hc:.1e} residual {resid:.1e} ΔQ "
              f"{o['Q_relerr_hc_vs_gauge']:.1e} | INVARIANT: planted mass "
              f"{o['planted_entry_mass_hc']:.4f}->{o['planted_entry_mass_gauged']:.4f}, blind "
              f"{o['blind_dim_hc']}->{o['blind_dim_gauged']} | DIES: L top2 "
              f"{o['weight_stat_hc']['L_row_top2_concentration']:.3f}->"
              f"{o['weight_stat_gauged']['L_row_top2_concentration']:.3f}")
        res['null_gauge_handcoded'].append(o)

    print('== NULL 2b: gauge scramble of the trained model ==')
    res['null_gauge'] = []
    for seed in range(2):
        o_t, p, W, Qz = run(arm='graded', n_dist=4, seed=seed, verbose=False)
        pg, resid = gauge_refactor(p, seed=999 + seed)
        Qz_g = forms_in_z(pg, W)
        o = {'seed': seed, 'refactor_residual': resid,
             'Q_relerr_trained_vs_gauge': lam_relerr(Qz, Qz_g),
             'trained': {'eff_rank': o_t['eff_rank_raw'], 'blind_dim': o_t['blind_dim'],
                         'blind_overlap': o_t['blind_overlap'],
                         'planted_entry_mass': o_t['planted_entry_mass'],
                         'weight_stat': o_t['weight_stat']},
             'gauged': {'eff_rank': effective_rank(Qz_g, 1e-3)[0],
                        'blind_dim': int(blind_subspace(Qz_g, 1e-3)[0].shape[1]),
                        'planted_entry_mass': planted_mass(Qz_g),
                        'weight_stat': weight_level_stat(pg, W)}}
        print(f"  [gauge s={seed}] residual {resid:.1e}  Q identical: "
              f"{o['Q_relerr_trained_vs_gauge']:.1e}  L-mass-on-bits "
              f"{o['trained']['weight_stat']['L_mass_on_bits']:.3f} -> "
              f"{o['gauged']['weight_stat']['L_mass_on_bits']:.3f}  top2 "
              f"{o['trained']['weight_stat']['L_row_top2_concentration']:.3f} -> "
              f"{o['gauged']['weight_stat']['L_row_top2_concentration']:.3f}")
        res['null_gauge'].append(o)

    print('== NULL 3: task shuffle (permuted labels) ==')
    res['null_shuffle'] = []
    for seed in range(2):
        o, _, _, _ = run(arm='graded', n_dist=4, seed=seed, shuffle_labels=True)
        res['null_shuffle'].append(o)

    print('== knob: distractor count ==')
    res['knob_ndist'] = [run(arm='graded', n_dist=nd, seed=0)[0] for nd in (0, 4, 8, 16)]

    print('== knob: embedding anisotropy, EXACT model (pure measurement question) ==')
    res['knob_kappa_exact'] = []
    for kk in (1.0, 10.0, 100.0, 1000.0):
        d = K_BITS + 4
        W = make_embedding(d, seed=100, kappa=kk)
        p_hc = handcoded(d, W)
        gen, _ = sampler(W, 4, 'graded', seed=0)
        xe, ye, _ = gen(16384)
        o = {'kappa': kk,
             'fvu': float(((forward(p_hc, xe) - ye) ** 2).mean() / ye.var()),
             **xcoord_analysis(p_hc, W, xe, 4)}
        print(f"  kappa {kk:6.0f} condΣ {o['cond_Sigma_x']:.1e} | raw: rank "
              f"{o['eff_rank_raw']} blind {o['blind_dim_raw']}/4 ovl {o['blind_overlap_raw']:.3f}"
              f" | whitened: rank {o['eff_rank_whitened']} blind {o['blind_dim_whitened']}/4 "
              f"ovl {o['blind_overlap_whitened']:.3f}")
        res['knob_kappa_exact'].append(o)

    print('== knob: embedding anisotropy, TRAINED (optimisation is also affected) ==')
    res['knob_kappa'] = [run(arm='graded', n_dist=4, kappa=kk, seed=0)[0]
                         for kk in (1.0, 10.0, 100.0)]

    print('== knob: label noise ==')
    res['knob_noise'] = [run(arm='graded', n_dist=4, noise=nz, seed=0)[0]
                         for nz in (0.0, 0.1, 0.3)]

    res['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a1_results.json'
    with open(path, 'w') as fh:
        json.dump(res, fh, indent=1)
    print(f'\nwrote {path}  ({res["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
