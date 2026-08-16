"""A4 — planted quotient DGP: the stratification and the linearization band.

DGP.  Inputs x = U(m + s⊙z) in R^d: a planted mean m and per-coordinate scale s.
The teacher is a sum of rank-2 components

    y_i = Σ_j γ_j c_ij (a_jᵀx)(b_jᵀx)

with a 3×3 design over TWO independent axes, because the doc's prediction names
only one of them:

  γ (gain)            1.0 / 0.1 / 0.01   — how much of the function the component is
  ρ (mean-to-fluctuation along a_j,b_j)  0 / 2 / 10 — how CURVED it is on-distribution

plus exactly-collapsed input directions (never read at all).

Why two axes. Linearising a component replaces (aᵀx)(bᵀx) by its tangent at the
input mean; the residual is (aᵀδ)(bᵀδ) with δ the fluctuation. So how well a
component linearises is set by ρ, NOT by γ. The doc predicts the linearisation
band sits at "intermediate singular value" (the γ axis). This design lets the two
hypotheses disagree, and measures which one is right.

Interventions. Per component, three surgeries — keep (2d+m parameters), linearise
(d+m: one tangent covector), prune (0) — with the exact Pareto frontier over all
3^J assignments obtained by dynamic programming once additivity is verified.
"""

import json
import math
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bq_common import init_params, forward, interaction, train, row_space_kernel

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_dtype(torch.float64)

D = 48                 # input dim (of which the last N_DEAD are never read)
N_DEAD = 8
M = 6                  # outputs
H = 192
STEPS = 15000
BATCH = 1024
GAMMAS = (1.0, 0.1, 0.01)
RHOS = (0.0, 2.0, 10.0)
PER_CELL = 2
J = len(GAMMAS) * len(RHOS) * PER_CELL


# ------------------------------------------------------------------ DGP

def make_teacher(seed=0):
    """Components read DISJOINT pairs from a random orthonormal frame of the live
    subspace. Two consequences that matter: (a) the realised mean-to-fluctuation
    ratio of component j is exactly rho_j (with a shared mean vector over
    non-orthogonal directions it is not — the cross terms swamp it); (b) the
    component forms are Sym²-orthogonal, so surgery errors are exactly additive
    and the Pareto frontier below is exact rather than approximate."""
    g = torch.Generator().manual_seed(seed)
    live = D - N_DEAD
    U = torch.linalg.qr(torch.randn(live, live, generator=g))[0]
    comps, idx = [], 0
    for gi, gam in enumerate(GAMMAS):
        for ri, rho in enumerate(RHOS):
            for k in range(PER_CELL):
                a, b = torch.zeros(D), torch.zeros(D)
                a[:live], b[:live] = U[:, 2 * idx], U[:, 2 * idx + 1]
                idx += 1
                comps.append({'a': a.to(DEV), 'b': b.to(DEV), 'gamma': gam, 'rho': rho,
                              'cell': (gi, ri)})
    assert 2 * idx <= live, f'need {2 * idx} live dims, have {live}'
    c = (torch.randn(M, len(comps), generator=g) / math.sqrt(len(comps))).to(DEV)
    mu = torch.zeros(D, device=DEV)
    for cp in comps:                      # a_i . mu = rho_i exactly (orthonormal frame)
        mu = mu + cp['rho'] * (cp['a'] + cp['b'])
    return {'comps': comps, 'c': c, 'mu': mu}


def sampler(T, seed=0, noise=0.0):
    g = torch.Generator(device=DEV).manual_seed(seed)

    def gen(n):
        x = T['mu'] + torch.randn(n, D, generator=g, device=DEV, dtype=torch.get_default_dtype())
        feats = torch.stack([cp['gamma'] * (x @ cp['a']) * (x @ cp['b']) for cp in T['comps']], 1)
        y = feats @ T['c'].T
        if noise:
            y = y + noise * y.std() * torch.randn(y.shape, generator=g, device=DEV, dtype=y.dtype)
        return x, y

    return gen


def lift(x):
    return torch.cat([torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype), x], 1)


def measured_rho(T, gen):
    """The realised mean-to-fluctuation ratio per component (the planted rho is a
    construction knob; several components share the mean vector, so report what
    each component actually sees)."""
    x, _ = gen(20000)
    out = []
    for cp in T['comps']:
        pa, pb = x @ cp['a'], x @ cp['b']
        out.append(float(min(abs(pa.mean()) / pa.std(), abs(pb.mean()) / pb.std())))
    return out


# ------------------------------------------------- component decomposition

def component_pieces(Qt, T):
    """Least-squares decomposition of the student's quadratic part onto the
    planted component forms: Q_xx ≈ Σ_j β_ij S_j. Returns (beta (M,J), residual)."""
    S = torch.stack([0.5 * (torch.outer(cp['a'], cp['b']) + torch.outer(cp['b'], cp['a']))
                     for cp in T['comps']])
    iu, ju = torch.triu_indices(D, D)
    sc = torch.where(iu == ju, 1.0, math.sqrt(2.0)).to(Qt)
    A = (S[:, iu, ju] * sc).T                            # (dim Sym2, J)
    Qxx = Qt[:, 1:, 1:]
    Y = (Qxx[:, iu, ju] * sc).T                          # (dim Sym2, M)
    beta = torch.linalg.lstsq(A, Y).solution.T           # (M, J)
    resid = Y - A @ beta.T
    return beta, S, float((resid ** 2).sum() / (Y ** 2).sum())


def surgery_forms(beta, S, T, mu, j, mode):
    """The (M, D+1, D+1) lifted form contributed by component j under `mode`."""
    out = torch.zeros(M, D + 1, D + 1, device=S.device, dtype=S.dtype)
    Sj = S[j]
    if mode == 'prune':
        return out
    if mode == 'keep':
        out[:, 1:, 1:] = beta[:, j][:, None, None] * Sj
        return out
    if mode == 'linearize':
        # x^T S x  ->  2 mu^T S x - mu^T S mu   (tangent at the input mean)
        v = Sj @ mu
        q = float(mu @ Sj @ mu)
        out[:, 0, 0] = -beta[:, j] * q
        out[:, 0, 1:] = beta[:, j][:, None] * v[None, :]
        out[:, 1:, 0] = beta[:, j][:, None] * v[None, :]
        return out
    raise ValueError(mode)


PARAM_COST = {'keep': 2 * D + M, 'linearize': D + M, 'prune': 0}


# ------------------------------------------------------------------ analysis

@torch.no_grad()
def fn_err(Qa, Qb, xt, ref):
    """Empirical functional error ‖y_a - y_b‖² / ‖y_ref‖². Sample-based on
    purpose: the inputs have a nonzero mean, for which the zero-mean Wick
    formula behind the closed-form Λ metric is silently wrong."""
    ya = torch.einsum('ni,mij,nj->nm', xt, Qa, xt)
    yb = torch.einsum('ni,mij,nj->nm', xt, Qb, xt)
    return float(((ya - yb) ** 2).sum() / ref)


def dp_frontier(costs, errs, budgets):
    """Best-error-at-budget over all mode assignments, by DP on the additive
    (per-component) error model. Returns the assignment too, so the frontier can
    be RE-MEASURED exactly rather than trusted."""
    best = {0: (0.0, [])}
    for j in range(len(errs)):
        nxt = {}
        for c0, (e0, asg) in best.items():
            for mi in range(len(errs[j])):
                c, e = c0 + costs[j][mi], e0 + errs[j][mi]
                if c not in nxt or e < nxt[c][0]:
                    nxt[c] = (e, asg + [mi])
        best = nxt
    pts = sorted(best.items())
    res = []
    for B in budgets:
        feas = [(e, asg, c) for c, (e, asg) in pts if c <= B]
        if not feas:
            res.append({'budget': B, 'err_dp': float('nan'), 'assignment': None, 'cost': None})
        else:
            e, asg, c = min(feas)
            res.append({'budget': B, 'err_dp': e, 'assignment': asg, 'cost': c})
    return res


def main():
    t0 = time.time()
    out = {'config': {'d': D, 'n_dead': N_DEAD, 'm': M, 'h': H, 'steps': STEPS,
                      'gammas': GAMMAS, 'rhos': RHOS, 'per_cell': PER_CELL, 'device': DEV}}
    T = make_teacher(0)
    gen = sampler(T, seed=0)
    out['measured_rho'] = measured_rho(T, gen)

    print('== train student ==')
    p = init_params(D + 1, H, M, seed=1, device=DEV)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}

    def data_fn():
        x, y = gen(BATCH)
        return lift(x), y

    p, _, _ = train(p, data_fn, STEPS, lr=2e-3, cosine=True)
    xe, ye = gen(32768)
    xt = lift(xe)
    fvu = float(((forward(p, xt) - ye) ** 2).mean() / ye.var())
    print(f'  student FVU {fvu:.3e}')
    out['fvu'] = fvu

    Qt = interaction(p)                                   # (M, D+1, D+1)
    ref = float((torch.einsum('ni,mij,nj->nm', xt, Qt, xt) ** 2).sum())

    # ---- prediction (i): exactly-collapsed directions are in the kernel
    print('== prediction (i): planted collapse == kernel ==')
    Qxx = Qt[:, 1:, 1:]
    dead_ax = torch.zeros(D, N_DEAD, device=DEV, dtype=Qt.dtype)
    for i in range(N_DEAD):
        dead_ax[D - N_DEAD + i, i] = 1.0
    sens = torch.linalg.svdvals(Qxx.reshape(M * D, D))
    g = torch.Generator(device=DEV).manual_seed(5)
    with torch.no_grad():
        pert_dead = torch.zeros(xe.shape[0], D, device=DEV, dtype=xe.dtype)
        pert_dead[:, D - N_DEAD:] = torch.randn(xe.shape[0], N_DEAD, generator=g,
                                                device=DEV, dtype=xe.dtype)
        pert_live = torch.zeros_like(pert_dead)
        pert_live[:, :D - N_DEAD] = torch.randn(xe.shape[0], D - N_DEAD, generator=g,
                                                device=DEV, dtype=xe.dtype)
        pert_live = pert_live / pert_live.norm(dim=1, keepdim=True) * pert_dead.norm(dim=1, keepdim=True)
        y0 = forward(p, xt)
        dd = float(((forward(p, lift(xe + pert_dead)) - y0) ** 2).mean().sqrt())
        dl = float(((forward(p, lift(xe + pert_live)) - y0) ** 2).mean().sqrt())
    rows, ker, sv = row_space_kernel(Qxx, 1e-6)
    dead_in_ker = float(((ker.reshape(ker.shape[0], -1) @
                          torch.stack([torch.outer(dead_ax[:, i], dead_ax[:, i]).flatten()
                                       for i in range(N_DEAD)]).T) ** 2).sum() /
                        max(N_DEAD, 1))
    out['collapse'] = {'rms_out': float((y0 ** 2).mean().sqrt()),
                       'rms_delta_dead_perturbation': dd,
                       'rms_delta_live_perturbation': dl,
                       'blindness_ratio': dd / dl,
                       'lift_rank': int(rows.shape[0]), 'lift_kernel_dim': int(ker.shape[0]),
                       'dead_diag_lift_captured_by_kernel': dead_in_ker}
    print(f"  equivalent-input perturbation (dead dirs): rms Δy {dd:.3e} vs live {dl:.3e} "
          f"-> blindness ratio {dd / dl:.2e}")
    print(f"  lift map rank {rows.shape[0]} / kernel {ker.shape[0]} of "
          f"{D * (D + 1) // 2}; dead lift directions captured by kernel {dead_in_ker:.6f}")

    # ---- prediction (ii): contraction spectrum reproduces the planted gains
    print('== prediction (ii): contraction spectrum vs planted gain profile ==')
    beta, S, resid = component_pieces(Qt, T)
    gam_hat = []
    for j, cp in enumerate(T['comps']):
        gam_hat.append(float((beta[:, j] / T['c'][:, j]).median()))
    out['recovery'] = {'lstsq_residual': resid,
                       'planted_gamma': [cp['gamma'] for cp in T['comps']],
                       'recovered_gamma': gam_hat}
    for gi, gam in enumerate(GAMMAS):
        got = [gam_hat[j] for j, cp in enumerate(T['comps']) if cp['cell'][0] == gi]
        print(f"  planted gamma {gam:6.3f}: recovered {[round(v, 5) for v in got]}")
    print(f'  component decomposition residual {resid:.3e}')

    # ---- prediction (iii) + the intervention
    print('== stratification and the three surgeries (per component) ==')
    modes = ['keep', 'linearize', 'prune']
    Qmode = [[surgery_forms(beta, S, T, T['mu'], j, md) for md in modes] for j in range(J)]
    # Surgery is applied to the student in place: swap component j's "keep" form
    # for its linearised or pruned form and leave everything else (including the
    # part of the student outside the component span) untouched.
    errs, costs, per_comp = [], [], []
    for j in range(J):
        ej, cj = [], []
        for mi, md in enumerate(modes):
            Qtry = Qt - Qmode[j][0] + Qmode[j][mi]
            ej.append(fn_err(Qt, Qtry, xt, ref))
            cj.append(PARAM_COST[md])
        errs.append(ej)
        costs.append(cj)
        cp = T['comps'][j]
        per_comp.append({'j': j, 'gamma': cp['gamma'], 'rho': cp['rho'],
                         'measured_rho': out['measured_rho'][j],
                         'err_keep': ej[0], 'err_linearize': ej[1], 'err_prune': ej[2],
                         'curvature_ratio': ej[1] / max(ej[2], 1e-300)})
        print(f"  comp {j:2d} gamma {cp['gamma']:5.2f} rho {cp['rho']:4.1f} (measured "
              f"{out['measured_rho'][j]:5.2f}): prune err {ej[2]:.3e} | linearize err "
              f"{ej[1]:.3e} | ratio lin/prune {ej[1] / max(ej[2], 1e-300):.3e}")
    out['per_component'] = per_comp

    # additivity check: does the sum of single-component errors predict a combination?
    import random
    random.seed(0)
    checks = []
    for t in range(6):
        pick = [random.randrange(3) for _ in range(J)]
        Qtry = Qt.clone()
        pred = 0.0
        for j in range(J):
            Qtry = Qtry - Qmode[j][0] + Qmode[j][pick[j]]
            pred += errs[j][pick[j]]
        checks.append({'assignment': pick, 'measured': fn_err(Qt, Qtry, xt, ref), 'predicted': pred})
    out['additivity_check'] = checks
    mx = max(abs(c['measured'] - c['predicted']) / max(c['measured'], 1e-30) for c in checks)
    print(f'  additivity of surgery errors: max relative deviation {mx:.3f} over 6 random assignments')

    print('== exact Pareto frontier: keep/prune vs keep/prune/linearize ==')
    budgets = sorted(set([0] + [int(v) for v in torch.linspace(0, J * PARAM_COST['keep'], 25)]))
    fro_kp = dp_frontier([[c[0], c[2]] for c in costs], [[e[0], e[2]] for e in errs], budgets)
    fro_all = dp_frontier(costs, errs, budgets)

    def measure(asg, mode_map):
        if asg is None:
            return float('nan')
        Qtry = Qt.clone()
        for j, mi in enumerate(asg):
            Qtry = Qtry - Qmode[j][0] + Qmode[j][mode_map[mi]]
        return fn_err(Qt, Qtry, xt, ref)

    for r in fro_kp:
        r['err'] = measure(r['assignment'], {0: 0, 1: 2})
    for r in fro_all:
        r['err'] = measure(r['assignment'], {0: 0, 1: 1, 2: 2})
    out['frontier'] = {'budgets': budgets, 'keep_prune': fro_kp, 'keep_prune_linearize': fro_all}
    wins = sum(1 for a_, b_ in zip(fro_kp, fro_all) if b_['err'] < a_['err'] * 0.999)
    dpmax = max(abs(r['err'] - r['err_dp']) / max(r['err'], 1e-30) for r in fro_all
                if r['assignment'] is not None)
    print(f'  DP (additive) vs exactly re-measured error: max relative deviation {dpmax:.3e}')
    print(f'  linearization strictly improves the frontier at {wins}/{len(budgets)} budgets')
    for a_, b_ in list(zip(fro_kp, fro_all))[::3]:
        print(f"    budget {a_['budget']:5d}: keep/prune {a_['err']:.4e} | +linearize "
              f"{b_['err']:.4e} | improvement x{a_['err'] / max(b_['err'], 1e-300):.2f}")

    # which axis predicts linearization benefit?
    print('== which axis governs the linearization band? ==')
    by_gamma, by_rho = {}, {}
    for r in per_comp:
        by_gamma.setdefault(r['gamma'], []).append(r['curvature_ratio'])
        by_rho.setdefault(r['rho'], []).append(r['curvature_ratio'])
    lg = [math.log10(max(r['curvature_ratio'], 1e-12)) for r in per_comp]
    cg = [math.log10(r['gamma']) for r in per_comp]
    cr = [math.log10(1 + r['measured_rho']) for r in per_comp]

    def corr(u, v):
        mu_, mv = sum(u) / len(u), sum(v) / len(v)
        du = [a_ - mu_ for a_ in u]
        dv = [a_ - mv for a_ in v]
        return sum(a_ * b_ for a_, b_ in zip(du, dv)) / math.sqrt(
            sum(a_ * a_ for a_ in du) * sum(b_ * b_ for b_ in dv) + 1e-300)
    out['band_axis_corr'] = {'log_ratio_vs_log_gamma': corr(lg, cg),
                             'log_ratio_vs_log_measured_rho': corr(lg, cr)}
    print(f"  correlation of log(lin/prune ratio) with log gamma: "
          f"{out['band_axis_corr']['log_ratio_vs_log_gamma']:+.3f}  |  with log(1+measured rho): "
          f"{out['band_axis_corr']['log_ratio_vs_log_measured_rho']:+.3f}")
    out['band_axis'] = {
        'by_gamma': {str(k): sum(v) / len(v) for k, v in by_gamma.items()},
        'by_rho': {str(k): sum(v) / len(v) for k, v in by_rho.items()}}
    print('  mean lin/prune error ratio by GAMMA (doc\'s predicted axis): '
          + ', '.join(f'{k}: {sum(v) / len(v):.3e}' for k, v in sorted(by_gamma.items())))
    print('  mean lin/prune error ratio by RHO   (curvature axis):        '
          + ', '.join(f'{k}: {sum(v) / len(v):.3e}' for k, v in sorted(by_rho.items())))

    # where the optimal assignment places each component, as the budget moves
    print('== the band moves with the budget (which components get linearized) ==')
    sel = []
    for B in budgets[::3]:
        bestv, besta = float('inf'), None
        stack = [(0, 0.0, [])]
        for j in range(J):
            nxt = {}
            for c0, e0, asg in stack:
                for mi in range(3):
                    c, e = c0 + costs[j][mi], e0 + errs[j][mi]
                    if c <= B and ((c not in nxt) or e < nxt[c][1]):
                        nxt[c] = (c, e, asg + [mi])
            stack = list(nxt.values())
        for c, e, asg in stack:
            if e < bestv:
                bestv, besta = e, asg
        if besta:
            sel.append({'budget': B, 'n_keep': besta.count(0), 'n_linearize': besta.count(1),
                        'n_prune': besta.count(2), 'err': bestv,
                        'linearized_cells': [(T['comps'][j]['gamma'], T['comps'][j]['rho'])
                                             for j in range(J) if besta[j] == 1]})
            print(f"    budget {B:5d}: keep {sel[-1]['n_keep']:2d} linearize "
                  f"{sel[-1]['n_linearize']:2d} prune {sel[-1]['n_prune']:2d} err {bestv:.3e} "
                  f"| linearized (gamma,rho) {sel[-1]['linearized_cells']}")
    out['band_vs_budget'] = sel

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a4_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
