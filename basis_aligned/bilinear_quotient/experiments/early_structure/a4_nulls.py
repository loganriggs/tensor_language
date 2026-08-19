"""A4 gap-closing: the four nulls, and the oracle-free version.

Two gaps were recorded against A4 in RESULTS.md:

  1. it shipped without the four nulls;
  2. it decomposed the student into components using the TEACHER's own directions,
     so the surgery analysis was handed the answer.

Both are closed here. The blind arm recovers the components with the CP fitter
that A3 calibrated — A4 has 18 components in 48 input dimensions, i.e. K/d = 0.38,
comfortably inside the regime A3 certified (exact recovery to K/d ≈ 1.5).

The axis claim is re-tested with predictors computed blind: each recovered
component's own size, and its own mean-to-fluctuation ratio measured from the
data through the RECOVERED directions rather than the planted ones.
"""

import json
import math
import sys
import time

import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
import a4_quotient as A4
from bq_common import init_params, forward, interaction, train, fit_cp, gauge_refactor

DEV = A4.DEV
torch.set_default_dtype(torch.float64)
D, M, J = A4.D, A4.M, A4.J


def train_student(T, seed=1, shuffle=False, steps=A4.STEPS):
    gen = A4.sampler(T, seed=0)
    p = init_params(D + 1, A4.H, M, seed=seed, device=DEV)
    p = {k: v.to(torch.get_default_dtype()) for k, v in p.items()}

    def data_fn():
        x, y = gen(A4.BATCH)
        if shuffle:
            y = y[torch.randperm(y.shape[0], device=DEV)]
        return A4.lift(x), y

    p, _, _ = train(p, data_fn, steps, lr=2e-3, cosine=True)
    return p, gen


def blind_components(Qt, rank=J, seed=0, steps=6000):
    """Recover components from the student's quadratic part with no reference to
    the teacher. Returns (S_hat (rank,D,D) unit-norm, beta (M,rank))."""
    Qxx = Qt[:, 1:, 1:]
    ph, err = fit_cp(Qxx, rank, steps=steps, lr=3e-2, seed=seed)
    S = 0.5 * (torch.einsum('ka,kb->kab', ph['L'], ph['R'])
               + torch.einsum('kb,ka->kab', ph['L'], ph['R']))
    n = S.flatten(1).norm(dim=1).clamp_min(1e-30)
    beta = ph['D'] * n[None, :]
    return S / n[:, None, None], beta, err, ph


def blind_predictors(S, ph, xe):
    """Per component, computed WITHOUT the planted labels:
       size  = the component's share of the model's output variance
       rho   = mean-to-fluctuation of the input along the recovered directions."""
    mu = xe.mean(0)
    out = []
    for j in range(S.shape[0]):
        a, b = ph['L'][j], ph['R'][j]
        pa, pb = xe @ a, xe @ b
        rho = float(min(abs(pa.mean()) / pa.std().clamp_min(1e-30),
                        abs(pb.mean()) / pb.std().clamp_min(1e-30)))
        f = torch.einsum('ni,ij,nj->n', xe, S[j], xe)
        out.append({'rho_hat': rho, 'var_hat': float(f.var())})
    return out


def surgery_table(Qt, S, beta, mu, xt, ref):
    """Per component: error of keeping / straightening / deleting it."""
    modes = ['keep', 'linearize', 'prune']
    Qmode = []
    for j in range(S.shape[0]):
        row = []
        for md in modes:
            out = torch.zeros(M, D + 1, D + 1, device=S.device, dtype=S.dtype)
            if md == 'keep':
                out[:, 1:, 1:] = beta[:, j][:, None, None] * S[j]
            elif md == 'linearize':
                v = S[j] @ mu
                q = float(mu @ S[j] @ mu)
                out[:, 0, 0] = -beta[:, j] * q
                out[:, 0, 1:] = beta[:, j][:, None] * v[None, :]
                out[:, 1:, 0] = beta[:, j][:, None] * v[None, :]
            row.append(out)
        Qmode.append(row)
    errs, costs = [], []
    for j in range(S.shape[0]):
        ej = []
        for mi in range(3):
            ej.append(A4.fn_err(Qt, Qt - Qmode[j][0] + Qmode[j][mi], xt, ref))
        errs.append(ej)
        costs.append([A4.PARAM_COST[m] for m in modes])
    return errs, costs, Qmode


def corr(u, v):
    mu_, mv = sum(u) / len(u), sum(v) / len(v)
    du, dv = [a - mu_ for a in u], [b - mv for b in v]
    return sum(a * b for a, b in zip(du, dv)) / math.sqrt(
        sum(a * a for a in du) * sum(b * b for b in dv) + 1e-300)


def analyse(p, T, gen, label, blind=True, rank=J, verbose=True):
    xe, ye = gen(32768)
    xt = A4.lift(xe)
    Qt = interaction(p)
    ref = float((torch.einsum('ni,mij,nj->nm', xt, Qt, xt) ** 2).sum())
    fvu = float(((forward(p, xt) - ye) ** 2).mean() / ye.var())
    mu = T['mu']

    if blind:
        S, beta, cperr, ph = blind_components(Qt, rank=rank)
        pred = blind_predictors(S, ph, xe)
        # match to planted purely for SCORING
        Sp = torch.stack([0.5 * (torch.outer(c['a'], c['b']) + torch.outer(c['b'], c['a']))
                          for c in T['comps']])
        Sp = Sp / Sp.flatten(1).norm(dim=1)[:, None, None]
        A_ = S.flatten(1) / S.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-30)
        Bm = Sp.flatten(1) / Sp.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-30)
        Cm = (A_ @ Bm.T).abs()
        ri, ci = linear_sum_assignment((-Cm).cpu().numpy())
        match_cos = [float(Cm[a, b]) for a, b in zip(ri, ci)]
        planted_of = {int(a): int(b) for a, b in zip(ri, ci)}
    else:
        beta, Sp_, lsq = A4.component_pieces(Qt, T)
        S = Sp_ / Sp_.flatten(1).norm(dim=1)[:, None, None]
        beta = beta * Sp_.flatten(1).norm(dim=1)[None, :]
        cperr, match_cos = lsq, [1.0] * J
        planted_of = {j: j for j in range(J)}
        pred = [{'rho_hat': A4.measured_rho(T, gen)[j], 'var_hat': float('nan')}
                for j in range(J)]

    errs, costs, _ = surgery_table(Qt, S, beta, mu, xt, ref)
    ratio = [e[1] / max(e[2], 1e-300) for e in errs]
    lg = [math.log10(max(r, 1e-12)) for r in ratio]
    sizes = [max(e[2], 1e-300) for e in errs]          # prune error = the piece's size
    c_size = corr(lg, [math.log10(s) for s in sizes])
    c_rho = corr(lg, [math.log10(1 + pred[j]['rho_hat']) for j in range(len(errs))])

    budgets = sorted(set(int(v) for v in torch.linspace(0, len(errs) * A4.PARAM_COST['keep'], 25)))
    fkp = A4.dp_frontier([[c[0], c[2]] for c in costs], [[e[0], e[2]] for e in errs], budgets)
    fal = A4.dp_frontier(costs, errs, budgets)
    wins = sum(1 for a, b in zip(fkp, fal)
               if b['err_dp'] < a['err_dp'] * 0.999)
    gain = [a['err_dp'] / max(b['err_dp'], 1e-300) for a, b in zip(fkp, fal)
            if a['err_dp'] == a['err_dp'] and a['err_dp'] > 1e-30]

    res = {'label': label, 'blind': blind, 'fvu': fvu, 'cp_or_lsq_err': cperr,
           'mean_match_cos': sum(match_cos) / len(match_cos),
           'frac_match_above_0.9': sum(1 for c in match_cos if c > 0.9) / len(match_cos),
           'corr_log_ratio_vs_log_size': c_size,
           'corr_log_ratio_vs_log_rho': c_rho,
           'linearize_wins_at': wins, 'n_budgets': len(budgets),
           'median_gain': sorted(gain)[len(gain) // 2] if gain else float('nan'),
           'per_component': [{'j': j, 'planted': planted_of.get(j),
                              'rho_hat': pred[j]['rho_hat'], 'ratio': ratio[j],
                              'err_keep': errs[j][0], 'err_lin': errs[j][1],
                              'err_prune': errs[j][2]} for j in range(len(errs))]}
    if verbose:
        print(f"  [{label}] fvu {fvu:.2e} | component recovery: mean cos "
              f"{res['mean_match_cos']:.3f}, {res['frac_match_above_0.9']*100:.0f}% above 0.9")
        print(f"    axis: corr(log straighten/delete, log size) {c_size:+.3f} | "
              f"vs log(1+rho) {c_rho:+.3f}")
        print(f"    straightening wins at {wins}/{len(budgets)} budgets, "
              f"median gain x{res['median_gain']:.2f}")
    return res


def main():
    t0 = time.time()
    out = {'note': 'closes the two gaps recorded against A4: missing nulls, oracle '
                   'component directions', 'runs': []}
    T = A4.make_teacher(0)

    print('== reference: the oracle decomposition from a4_quotient.py ==')
    p, gen = train_student(T, seed=1)
    out['runs'].append(analyse(p, T, gen, 'oracle-directions', blind=False))

    print('\n== BLIND: components recovered by the CP fitter A3 calibrated ==')
    for rank in (J, J + 4):
        out['runs'].append(analyse(p, T, gen, f'blind-rank{rank}', blind=True, rank=rank))

    print('\n== NULL 1: random weights ==')
    pr = init_params(D + 1, A4.H, M, seed=777, device=DEV)
    pr = {k: v.to(torch.get_default_dtype()) for k, v in pr.items()}
    out['null_random'] = analyse(pr, T, gen, 'null-random', blind=True)

    print('\n== NULL 2: gauge scramble (every measurement must be unchanged) ==')
    pg, resid = gauge_refactor(p, seed=404)
    rg = analyse(pg, T, gen, 'null-gauge', blind=False)
    rg['refactor_residual'] = resid
    base = out['runs'][0]
    print(f"    refactor residual {resid:.1e} | hidden width {A4.H} -> {pg['L'].shape[0]} | "
          f"axis corr {base['corr_log_ratio_vs_log_rho']:+.3f} -> "
          f"{rg['corr_log_ratio_vs_log_rho']:+.3f} | wins "
          f"{base['linearize_wins_at']} -> {rg['linearize_wins_at']}")
    out['null_gauge'] = rg

    print('\n== NULL 3: task shuffle ==')
    ps, gens = train_student(T, seed=1, shuffle=True)
    out['null_taskshuffle'] = analyse(ps, T, gens, 'null-taskshuffle', blind=True)

    print('\n== NULL 4: reader shuffle ==')
    # permuting which OUTPUT reads which component leaves the component set — and
    # therefore every surgery decision — untouched. Vacuous here by construction;
    # measured rather than asserted.
    Qt = interaction(p)
    g = torch.Generator().manual_seed(9)
    perm = torch.randperm(M, generator=g).to(DEV)

    class PermP(dict):
        pass
    pp = {k: v.clone() for k, v in p.items()}
    pp['D'] = p['D'][perm]
    rr = analyse(pp, T, gen, 'null-readershuffle', blind=False)
    out['null_readershuffle'] = rr
    print(f"    component set unchanged by construction; axis corr "
          f"{base['corr_log_ratio_vs_log_rho']:+.3f} -> {rr['corr_log_ratio_vs_log_rho']:+.3f}")

    out['runtime_s'] = time.time() - t0
    path = '/workspace/tensor_language/basis_aligned/bilinear_quotient/a4_nulls_results.json'
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {path} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
