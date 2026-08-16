"""A3 — planted low-rank teacher: CP calibration.

DGP.  y_i = Σ_k c_ik (a_kᵀx)(b_kᵀx) with known {a_k, b_k}, true rank K.
Train a bilinear student, then fit a partially-symmetric CP to the student's
interaction forms and ask whether the planted directions come back.

The gauge. Q_i = Σ_k c_ik·sym(a_k b_kᵀ), so the identifiable object per component
is the SYMMETRIC form S_k = sym(a_k b_kᵀ), invariant to the scale gauge
(a,b) → (ca, b/c), to a joint sign flip, and to swapping a↔b. Recovery is
therefore scored on {S_k} (matched by Hungarian assignment), not on raw factors.

This experiment exists to bound the others: it says where a CP component may be
read as a variable and where it may not. Two arms are run separately, because
they fail for different reasons:
  solver  — fit CP to the EXACT teacher forms. Isolates the fitter.
  learned — fit CP to a trained student. Adds estimation error on top.
"""

import json
import math
import sys
import time

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import (init_params, forward, interaction, train, fit_cp, lam_relerr,
                       lam_cos, gauge_refactor, effective_rank)

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

D = 16
M = 8
STEPS = 8000
BATCH = 1024


# ------------------------------------------------------------------ teacher

def teacher(K, seed, rho=0.0):
    """Planted directions; rho mixes in a shared direction, correlating them."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(K, D, generator=g)
    b = torch.randn(K, D, generator=g)
    if rho > 0:
        w1 = torch.randn(1, D, generator=g)
        w2 = torch.randn(1, D, generator=g)
        a = (1 - rho) * a + rho * w1 * math.sqrt(D) / w1.norm() * a.norm(dim=1, keepdim=True)
        b = (1 - rho) * b + rho * w2 * math.sqrt(D) / w2.norm() * b.norm(dim=1, keepdim=True)
    a = (a / a.norm(dim=1, keepdim=True)).to(DEV)
    b = (b / b.norm(dim=1, keepdim=True)).to(DEV)
    c = (torch.randn(M, K, generator=g) / math.sqrt(K)).to(DEV)
    return {'a': a, 'b': b, 'c': c}


def teacher_forms(T):
    S = torch.einsum('ka,kb->kab', T['a'], T['b'])
    S = 0.5 * (S + S.transpose(1, 2))
    return torch.einsum('ik,kab->iab', T['c'], S), S


def make_data(T, noise=0.0, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def gen(n):
        x = torch.randn(n, D, generator=g, device=DEV, dtype=torch.get_default_dtype())
        y = ((x @ T['a'].T) * (x @ T['b'].T)) @ T['c'].T
        if noise:
            y = y + noise * y.std() * torch.randn(y.shape, generator=g, device=DEV, dtype=y.dtype)
        return x, y

    return gen


# ------------------------------------------------------------------ scoring

def component_forms(p):
    S = torch.einsum('ka,kb->kab', p['L'], p['R'])
    S = 0.5 * (S + S.transpose(1, 2))
    n = S.flatten(1).norm(dim=1).clamp_min(1e-30)
    return S / n[:, None, None]


def match_components(S_hat, S_true):
    """Hungarian matching on |cosine| between symmetric component forms."""
    A = S_hat.flatten(1)
    B = S_true.flatten(1)
    A = A / A.norm(dim=1, keepdim=True).clamp_min(1e-30)
    B = B / B.norm(dim=1, keepdim=True).clamp_min(1e-30)
    C = (A @ B.T).abs()
    r, c = linear_sum_assignment((-C).cpu().numpy())
    cos = C[r, c]
    return {'mean_cos': float(cos.mean()), 'median_cos': float(cos.median()),
            'frac_above_0.9': float((cos > 0.9).to(cos.dtype).mean()),
            'frac_above_0.99': float((cos > 0.99).to(cos.dtype).mean()),
            'per_component': [round(float(v), 4) for v in cos]}


def subspace_recovery(p, T):
    """Do the recovered factor directions span the planted 2K-dim subspace?"""
    Vh = torch.cat([p['L'], p['R']], 0).T
    Vt = torch.cat([T['a'], T['b']], 0).T
    Qh = torch.linalg.qr(Vh)[0]
    Qt = torch.linalg.qr(Vt)[0]
    return float(((Qh.T @ Qt) ** 2).sum() / Qt.shape[1])


# ------------------------------------------------------------------ runs

def run_solver(K, seed=0, rho=0.0, fit_rank=None, steps=6000):
    """Arm 1: CP fitted to the EXACT teacher forms. Pure fitter calibration."""
    T = teacher(K, seed, rho)
    Q, S_true = teacher_forms(T)
    r = fit_rank or K
    ph, err = fit_cp(Q, r, steps=steps, lr=3e-2, seed=seed)
    S_hat = component_forms(ph)
    mt = match_components(S_hat, S_true / S_true.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None])
    return {'arm': 'solver', 'K': K, 'fit_rank': r, 'rho': rho, 'seed': seed,
            'cp_relerr': err, 'subspace_recovery': subspace_recovery(ph, T), **mt}


def run_learned(K, h, seed=0, rho=0.0, noise=0.0, steps=STEPS, fit_rank=None):
    """Arm 2: train a student, then fit CP to the student's forms."""
    T = teacher(K, seed, rho)
    Q_true, S_true = teacher_forms(T)
    gen = make_data(T, noise, seed)
    p = init_params(D, h, M, seed=seed + 50, device=DEV)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
    p, _, _ = train(p, lambda: gen(BATCH), steps, lr=3e-3, cosine=True)
    xe, ye = gen(16384)
    fvu = float(((forward(p, xe) - ye) ** 2).mean() / ye.var())
    Q = interaction(p)
    r = fit_rank or K
    ph, err = fit_cp(Q, r, steps=6000, lr=3e-2, seed=seed)
    S_hat = component_forms(ph)
    mt = match_components(S_hat, S_true / S_true.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None])
    return {'arm': 'learned', 'K': K, 'h': h, 'fit_rank': r, 'rho': rho, 'noise': noise,
            'seed': seed, 'fvu': fvu,
            'form_relerr_vs_teacher': lam_relerr(Q_true, Q), 'cp_relerr': err,
            'eff_rank': effective_rank(Q, 1e-3)[0],
            'subspace_recovery': subspace_recovery(ph, T), **mt}, p, T, S_true


def show(r):
    extra = f" fvu {r['fvu']:.2e} ΔQ {r['form_relerr_vs_teacher']:.3f}" if 'fvu' in r else ''
    print(f"  [{r['arm']} K={r['K']}{' h=' + str(r['h']) if 'h' in r else ''} rho={r['rho']} "
          f"nz={r.get('noise', 0)}]{extra} cp-err {r['cp_relerr']:.2e} | matched cos "
          f"mean {r['mean_cos']:.3f} med {r['median_cos']:.3f} | >0.9 {r['frac_above_0.9']:.2f} "
          f">0.99 {r['frac_above_0.99']:.2f} | subspace {r['subspace_recovery']:.3f}")


def main():
    t0 = time.time()
    out = {'config': {'d': D, 'm': M, 'steps': STEPS, 'batch': BATCH, 'device': DEV},
           'dim_sym2': D * (D + 1) // 2}

    print('== ARM 1 (solver calibration): CP fitted to exact teacher forms ==')
    out['solver'] = []
    for K in (2, 4, 8, 12, 16, 24, 32, 48):
        for seed in range(2):
            r = run_solver(K, seed)
            out['solver'].append(r)
            if seed == 0:
                show(r)

    print('\n== ARM 1b: correlated planted directions ==')
    out['solver_rho'] = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        r = run_solver(8, 0, rho=rho)
        out['solver_rho'].append(r)
        show(r)

    print('\n== ARM 2 (learned): train a student, then fit CP ==')
    out['learned'] = []
    for K in (2, 4, 8, 16, 24):
        for h in (K, 4 * K, 64):
            r, p, T, S = run_learned(K, h)
            out['learned'].append(r)
            show(r)

    print('\n== ARM 2b: label noise ==')
    out['learned_noise'] = []
    for nz in (0.0, 0.05, 0.2, 0.5):
        r, _, _, _ = run_learned(8, 32, noise=nz)
        out['learned_noise'].append(r)
        show(r)

    print('\n== ARM 2c: correlated planted directions, learned ==')
    out['learned_rho'] = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        r, _, _, _ = run_learned(8, 32, rho=rho)
        out['learned_rho'].append(r)
        show(r)

    print('\n== ARM 2d: over/under-ranked fits (K=8 truth) ==')
    out['rank_mismatch'] = []
    for fr in (4, 6, 8, 10, 16):
        r, _, _, _ = run_learned(8, 32, fit_rank=fr)
        out['rank_mismatch'].append(r)
        show(r)

    print('\n== NULL 1: random weights (no planted structure to find) ==')
    out['null_random'] = []
    for seed in range(3):
        T = teacher(8, seed)
        _, S_true = teacher_forms(T)
        p = init_params(D, 32, M, seed=900 + seed, device=DEV)
        p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
        ph, err = fit_cp(interaction(p), 8, steps=6000, lr=3e-2, seed=seed)
        mt = match_components(component_forms(ph),
                              S_true / S_true.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None])
        o = {'seed': seed, 'cp_relerr': err, 'subspace_recovery': subspace_recovery(ph, T), **mt}
        print(f"  [random {seed}] cp-err {err:.2e} matched cos mean {mt['mean_cos']:.3f} "
              f">0.9 {mt['frac_above_0.9']:.2f} subspace {o['subspace_recovery']:.3f}")
        out['null_random'].append(o)

    print('\n== NULL 2: gauge scramble (component recovery must be unchanged) ==')
    out['null_gauge'] = []
    for seed in range(2):
        r, p, T, S_true = run_learned(8, 32, seed=seed)
        pg, resid = gauge_refactor(p, seed=321 + seed)
        ph, err = fit_cp(interaction(pg), 8, steps=6000, lr=3e-2, seed=seed)
        mt = match_components(component_forms(ph),
                              S_true / S_true.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None])
        o = {'seed': seed, 'refactor_residual': resid,
             'mean_cos_trained': r['mean_cos'], 'mean_cos_gauged': mt['mean_cos'],
             'frac09_trained': r['frac_above_0.9'], 'frac09_gauged': mt['frac_above_0.9'],
             'hidden_width_trained': 32, 'hidden_width_gauged': int(pg['L'].shape[0])}
        print(f"  [gauge {seed}] residual {resid:.1e} | h {32} -> {o['hidden_width_gauged']} | "
              f"matched cos {r['mean_cos']:.3f} -> {mt['mean_cos']:.3f}, >0.9 "
              f"{r['frac_above_0.9']:.2f} -> {mt['frac_above_0.9']:.2f}")
        out['null_gauge'].append(o)

    print('\n== NULL 3: task shuffle (student trained on permuted targets) ==')
    out['null_taskshuffle'] = []
    for seed in range(2):
        T = teacher(8, seed)
        _, S_true = teacher_forms(T)
        gen = make_data(T, seed=seed)

        def data_fn():
            x, y = gen(BATCH)
            return x, y[torch.randperm(y.shape[0], device=DEV)]

        p = init_params(D, 32, M, seed=seed + 50, device=DEV)
        p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}
        p, _, _ = train(p, data_fn, STEPS, lr=3e-3, cosine=True)
        ph, err = fit_cp(interaction(p), 8, steps=6000, lr=3e-2, seed=seed)
        mt = match_components(component_forms(ph),
                              S_true / S_true.flatten(1).norm(dim=1).clamp_min(1e-30)[:, None, None])
        print(f"  [shuffled {seed}] cp-err {err:.2e} matched cos mean {mt['mean_cos']:.3f} "
              f">0.9 {mt['frac_above_0.9']:.2f}")
        out['null_taskshuffle'].append({'seed': seed, 'cp_relerr': err, **mt})

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a3_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
