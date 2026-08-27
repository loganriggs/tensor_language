# ratio_zero_ablation: IS THE RATIO RESULT AN ARTIFACT OF MEAN-ABLATION? — a third
# independent axis: same hypothesis, DIFFERENT INTERVENTION.
#
# The eigenvalue ratio |lam1/lam2| has now been tested out-of-sample three times
# (S1647 mlp11 function words rho +.678; S1648 mlp11 type-spanning +.511; S1649 mlp14
# +.573), with the strongest legitimate pool -- class-disjoint, across sites -- at
# rho +.532, permutation p .0085 on n=24. Caution 1 remains undischarged: S1614
# reported rho .6727 at p .0192 and S1616 refuted it, so this ledger has been fooled
# in this numeric neighbourhood before.
#
# More classes or more sites buy more of the same evidence. This varies a THIRD axis
# that no previous run touched: the INTERVENTION. Every CE rise in S1643-S1649 came
# from MEAN-ablation -- replacing the slice coordinates of the MLP input with their
# global mean. If the ratio->cost relationship is an artifact of that particular
# intervention it should vanish under a different one; if it is a property of the
# quadratic form it should survive.
#
# ZERO-ablation is the cleanest single-variable change: the slice coordinates are set
# to zero rather than to their global mean. It is a harsher intervention -- it removes
# the class-independent component as well as the class-conditional deviation -- so the
# costs should be larger in absolute terms, which pred_c checks as a manipulation
# check rather than as a claim.
#
# Uses S1648's twelve TYPE-SPANNING classes at mlp11. Those were chosen because
# S1647's set has already been used twice (S1647 and S1649) and reusing it a third
# time would make any pooled statement harder to audit, not easier.
#
# Registered predictions:
#   pred_a THE RELATIONSHIP SURVIVES A DIFFERENT INTERVENTION: rho(|lam1/lam2|,
#          relative CE rise under ZERO-ablation) >= +.30.
#   pred_b AND IS AT LEAST AS STRONG AS UNDER MEAN-ABLATION: rho >= +.5105, the value
#          S1648 measured on these exact twelve classes with the mean intervention.
#   pred_c MANIPULATION CHECK -- zero-ablation is harsher: the MEAN relative CE rise
#          across the twelve classes exceeds the mean under mean-ablation (S1648's
#          twelve gave +.00676).
import json, time, sys, os, re, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152; T = 256
SITE = 11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ratio_zero_ablation_results.json'
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
NROWS = 480
MEANABL_RHO = 0.5105            # §1648, these exact twelve classes, MEAN-ablation
MEANABL_MEANRISE = 0.00676      # §1648, mean relative rise over the same twelve

# twelve TYPE-SPANNING classes; none has ever had its CE rise measured
PATS = {'exclaim': r'^!$|^ !$', 'semicolon': r'^;$|^ ;$', 'colon': r'^:$|^ :$',
        'quote': r'^"$|^ "$', 'dash': r'^-$|^ -$', 'digit': r'^ ?[0-9]+$',
        'cap': r'^ [A-Z][a-z]+$', 'we': r'^ (we|We)$', 'you': r'^ (you|You)$',
        'it': r'^ (it|It)$', 'an': r'^ an$', 'my': r'^ my$'}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def slice_and_eigs(mask_v):
    """Top-RANK |lambda| eigenpair of the class-projected quadratic. Weights only."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:RANK]
    return V[:, o].contiguous(), [float(lam[i]) for i in o]


def mk_pre_hook(V2, mu):
    """ZERO-ablation: project the slice OUT entirely. mu is accepted for signature
    compatibility with the mean-ablation lineage and is deliberately unused."""
    def pre(mod, args):
        f = args[0].float()
        p = f @ V2
        return ((f - p @ V2.T).to(args[0].dtype),) + tuple(args[1:])
    return pre


@torch.no_grad()
def ce_pass(rows, mask_v, V2, mu):
    """Mean CE on the class's own positions. mu=None captures the global slice mean."""
    cap = {}
    if mu is None:
        def h(mod, args):
            f = args[0].float().reshape(-1, D)
            cap['s'] = cap.get('s', torch.zeros(RANK, device=DEV)) + (f @ V2).sum(0)
            cap['n'] = cap.get('n', 0) + f.shape[0]
            return None
        handle = H[SITE].mlp.register_forward_pre_hook(h)
    else:
        handle = H[SITE].mlp.register_forward_pre_hook(mk_pre_hook(V2, mu))
    tot, npos = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                 reduction='none').reshape(tg.shape)
            pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
            tot += float(ce[pm].sum()); npos += int(pm.sum())
    finally:
        handle.remove()
    return tot / max(npos, 1), npos, ((cap['s'] / max(cap['n'], 1)) if cap else None)


@torch.no_grad()
def main():
    import hashlib
    t0 = time.time()
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    rows = raw[:NROWS, :T + 1].contiguous()
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'ZERO-ABLATION: {len(PATS)} type-spanning classes at mlp{SITE}. {NROWS} canonical '
          f'rows (receipt {rh}). Single registered hypothesis: |lam1/lam2|.', flush=True)

    per = {}
    for c, pat in PATS.items():
        mask_v = rx(pat)
        V2, ev = slice_and_eigs(mask_v)
        base, npos, mu = ce_pass(rows, mask_v, V2, None)
        abl, _, _ = ce_pass(rows, mask_v, V2, mu)
        rise = abl - base
        rel = rise / base if base > 0 else 0.0
        ratio = abs(ev[0]) / max(abs(ev[1]), 1e-12)
        per[c] = {'lam1': round(ev[0], 3), 'lam2': round(ev[1], 3),
                  'ratio': round(ratio, 4), 'base_ce': round(base, 5),
                  'ce_rise': round(rise, 5), 'rel_ce_rise': round(rel, 5),
                  'n_positions': npos}
        print(f'  {c:6s} ratio {ratio:6.3f} | n={npos:5d} | CE {base:.4f} -> {abl:.4f} '
              f'| rel rise {rel:+.5f}', flush=True)

    ks = list(per); n = len(ks)
    ratio = {k: per[k]['ratio'] for k in ks}
    rel = {k: per[k]['rel_ce_rise'] for k in ks}

    def rk(x):
        o = sorted(x, key=lambda z: -x[z]); return [o.index(z) + 1 for z in ks]

    def rho(a, b):
        return 1 - 6 * sum((a[i] - b[i]) ** 2 for i in range(n)) / (n * (n * n - 1))

    r = rho(rk(ratio), rk(rel))
    random.seed(20260827)
    NPERM = 200000
    basep = list(range(1, n + 1))
    hits = sum(1 for _ in range(NPERM)
               if abs(rho(rk(ratio), random.sample(basep, n))) >= abs(r) - 1e-12)
    pval = hits / NPERM
    npos_pos = sum(1 for k in ks if per[k]['ce_rise'] > 0)

    mean_rise = sum(rel.values()) / len(rel)
    pa = r >= 0.30
    pb = r >= MEANABL_RHO
    pc = mean_rise > MEANABL_MEANRISE

    print(f'\n  HELD-OUT rho(|lam1/lam2|, relative CE rise) = {r:+.4f}', flush=True)
    print(f'  two-sided sampled permutation p ({NPERM}) = {pval:.5f}', flush=True)
    print(f'  same twelve under MEAN-ablation (§1648) = {MEANABL_RHO:+.4f}', flush=True)
    print(f'  mean relative rise: zero-abl {mean_rise:+.5f} vs mean-abl {MEANABL_MEANRISE:+.5f}',
          flush=True)
    print(f'  classes with positive CE rise: {npos_pos}/{n}  (§1644 had 11/12)', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'classes': 'TYPE-SPANNING held out -- punctuation, digits, capitalised, function words; no CE rise ever measured for any',
                      'single_registered_hypothesis': 'abs(lam1/lam2) predicts relative CE rise',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)',
                      'meanablation_rho_same_classes': MEANABL_RHO, 'intervention': 'ZERO-ablation'},
           'per_class': per, 'heldout_rho': round(r, 4),
           'permutation_p_two_sided': pval, 'n_permutations': NPERM,
           'n_positive_rise': npos_pos,
           'predictions': {'pred_a_rho_ge_030_zeroablation': bool(pa),
                           'pred_b_rho_ge_meanablation': bool(pb),
                           'pred_c_zero_costs_more': bool(pc)},
           'mean_relative_rise': round(mean_rise, 5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
