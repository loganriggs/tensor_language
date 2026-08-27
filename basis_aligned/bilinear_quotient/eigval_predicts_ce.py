# eigval_predicts_ce: IF THE SEPARATION GAP DOES NOT PREDICT CAUSAL COST, DOES THE
# RAW EIGENVALUE MAGNITUDE?
#
# §1644 refuted §1643's bridge: the separation gap does not predict how much ablating a
# class's slice costs (rho +.056, p .43 on twelve classes). §1645 then showed the
# n_positions confound was a SCALE effect -- normalising CE rise by the class's own
# baseline CE cuts the confound from -.580 to -.098 and raises the gap signal to +.231,
# but at permutation p .235 that is still nothing.
#
# §1645 named "more classes at the corrected currency" as the next step. Reaching p<.05
# on a true rho near .23 needs roughly fifty classes -- about a hundred GPU-minutes to
# establish a relationship explaining ~5% of rank variance. Before paying that, a
# cheaper and sharper question: the CE rise is REAL (11/12 positive) and varies THIRTY-
# FOLD across classes in relative terms (.0016 to .0537). WHAT PREDICTS IT?
#
# The obvious candidate costs nothing. The slice is the |lambda|-ordered top-2
# eigenpair of the class-projected quadratic form at mlp11, and those EIGENVALUES are
# computed from weights alone -- no forward pass, no rows, no seeds. If |lambda|
# predicts causal cost better than the separation statistic does, then twenty-two runs
# of separation machinery were measuring a worse signal than one eigendecomposition.
#
# Reuses the twelve classes and their measured CE rises from gap_ce_12class_results.json
# (§1644/§1645) so nothing is re-measured and the comparison is exact. The only new
# computation is twelve eigendecompositions.
#
# Registered predictions:
#   pred_a |lambda| PREDICTS BETTER THAN THE GAP: |rho(|lambda1|, relative CE rise)| >
#          .231, the gap's value at the corrected currency.
#   pred_b AND IT CLEARS SIGNIFICANCE, which the gap did not: sampled permutation
#          p < .05.
#   pred_c THE EFFECT IS IN THE MAGNITUDE, NOT THE SPREAD: |rho(|lambda1|, rel rise)|
#          exceeds |rho(|lambda1/lambda2|, rel rise)|, i.e. the size of the leading
#          eigenvalue matters more than how dominant it is over the second.
import json, time, sys, re, random, torch
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152
SITE = 11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'eigval_predicts_ce_results.json'
SRC = PT + 'gap_ce_12class_results.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
GAP_RHO_AT_CORRECTED_CURRENCY = 0.2308     # §1645


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def eigs(mask_v):
    """The top-RANK |lambda| eigenvalues of the class-projected quadratic at mlp11.
    Weights only -- no forward pass, no rows."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, _ = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:RANK]
    return [float(lam[i]) for i in o]


PATS = {'question': r'^\?$| \?$', 'to': r'^ to$', 'period': r'^\.$|^ \.$',
        'and': r'^ and$', 'comma': r'^,$|^ ,$', 'the': r'^ the$',
        'is': r'^ is$', 'at': r'^ at$', 'with': r'^ with$', 'by': r'^ by$',
        'of': r'^ of$', 'in': r'^ in$'}


@torch.no_grad()
def main():
    import os
    t0 = time.time()
    src = json.load(open(SRC))['cells']
    ks = [k for k in PATS if k in src]
    assert len(ks) == 12, f'expected 12 reused classes, got {len(ks)}'
    print(f'reusing {len(ks)} measured CE rises from gap_ce_12class (§1644/§1645); '
          f'computing {len(ks)} eigendecompositions, no forward passes', flush=True)

    rel = {k: src[k]['ce_rise'] / src[k]['base_ce'] for k in ks}
    gap = {k: src[k]['mean_gap'] for k in ks}
    e = {k: eigs(rx(PATS[k])) for k in ks}
    l1 = {k: abs(e[k][0]) for k in ks}
    ratio = {k: abs(e[k][0]) / max(abs(e[k][1]), 1e-12) for k in ks}

    n = len(ks)

    def rk(x):
        o = sorted(x, key=lambda z: -x[z]); return [o.index(z) + 1 for z in ks]

    def rho(a, b):
        return 1 - 6 * sum((a[i] - b[i]) ** 2 for i in range(n)) / (n * (n * n - 1))

    rr = rk(rel)
    rho_l1 = rho(rk(l1), rr)
    rho_ratio = rho(rk(ratio), rr)
    rho_gap = rho(rk(gap), rr)

    random.seed(20260827)
    NPERM = 200000
    basep = list(range(1, n + 1))
    obs = abs(rho_l1)
    hits = sum(1 for _ in range(NPERM)
               if abs(rho(rk(l1), random.sample(basep, n))) >= obs - 1e-12)
    pval = hits / NPERM

    pa = abs(rho_l1) > GAP_RHO_AT_CORRECTED_CURRENCY
    pb = pval < 0.05
    pc = abs(rho_l1) > abs(rho_ratio)

    print(f'\n  {"class":9s} {"|lam1|":>10s} {"|lam1/lam2|":>12s} {"rel CE rise":>12s} '
          f'{"gap":>9s}', flush=True)
    for k in sorted(ks, key=lambda z: -l1[z]):
        print(f'  {k:9s} {l1[k]:10.2f} {ratio[k]:12.3f} {rel[k]:12.5f} {gap[k]:+9.4f}',
              flush=True)
    print(f'\n  rho(|lam1|,      rel CE rise) = {rho_l1:+.4f}   two-sided perm p = {pval:.5f}',
          flush=True)
    print(f'  rho(|lam1/lam2|, rel CE rise) = {rho_ratio:+.4f}', flush=True)
    print(f'  rho(gap,         rel CE rise) = {rho_gap:+.4f}   (§1645, p .235)', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_classes': n,
                      'reused_from': 'gap_ce_12class_results.json (§1644/§1645)',
                      'new_computation': 'eigendecompositions only; no forward passes',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)'},
           'per_class': {k: {'lam1': round(e[k][0], 3), 'lam2': round(e[k][1], 3),
                             'abs_lam1': round(l1[k], 3), 'ratio': round(ratio[k], 4),
                             'rel_ce_rise': round(rel[k], 5), 'gap': gap[k]} for k in ks},
           'rho_lam1_vs_rel_rise': round(rho_l1, 4),
           'rho_ratio_vs_rel_rise': round(rho_ratio, 4),
           'rho_gap_vs_rel_rise': round(rho_gap, 4),
           'permutation_p_two_sided': pval, 'n_permutations': NPERM,
           'predictions': {'pred_a_lam1_beats_gap': bool(pa),
                           'pred_b_perm_p_lt_05': bool(pb),
                           'pred_c_magnitude_beats_ratio': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
