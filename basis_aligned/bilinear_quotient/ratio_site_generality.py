# ratio_site_generality: DOES THE EIGENVALUE RATIO PREDICT CAUSAL COST AT A SITE IT
# WAS NEVER TESTED AT? — generality in the SITE dimension, not more classes.
#
# S1647 and S1648 tested |lam1/lam2| on twenty-four classes held out from the
# selection, pooling to rho +.4157 at permutation p .0452. Both were at mlp11. The
# obvious next move is a third set of classes, but that buys more of the same evidence
# and the well-populated classes are running out.
#
# A different and arguably stronger test: the SAME hypothesis at a DIFFERENT SITE.
# Every measurement in this arc -- separation, gap, CE rise, ratio -- has been at
# mlp11. If the ratio predicts causal cost at mlp14 as well, that is evidence of a
# different KIND than a third class set, because the slice basis, the eigenvalues, the
# ablation target and the CE effect are all recomputed from a different layer's
# weights and activations. Nothing about mlp14 was used to select the hypothesis.
#
# Reuses S1647's twelve function words, which gave the cleanest signal at mlp11
# (rho +.6783). Their CLASSES have been used, but their ratio and CE rise AT MLP14
# have never been measured, so the hypothesis has never been fitted at this site.
#
# mlp14 is a deliberate choice: S1635 found it the one site where the separation gap
# ran POSITIVE across classes while its neighbours ran negative, and S1632 corroborated
# that with a second class. If the ratio is a real property of the quadratic form
# rather than an artifact of mlp11, it should not care.
#
# Registered predictions:
#   pred_a THE RATIO GENERALISES ACROSS SITES: rho(|lam1/lam2|, relative CE rise) >=
#          +.30 at mlp14.
#   pred_b AND CLEARS SIGNIFICANCE: two-sided sampled permutation p < .05.
#   pred_c ABLATION STILL COSTS AT THE NEW SITE: at least 10 of 12 classes show a
#          POSITIVE CE rise when their own mlp14 slice is mean-ablated.
import json, time, sys, os, re, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152; T = 256
SITE = 14                       # NEW SITE -- every prior measurement was mlp11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ratio_site_generality_results.json'
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
NROWS = 480
MLP11_RHO = 0.6783              # §1647, same twelve classes, at mlp11

# S1647's twelve function words -- never measured at mlp14
PATS = {'for': r'^ for$', 'on': r'^ on$', 'that': r'^ that$', 'as': r'^ as$',
        'was': r'^ was$', 'but': r'^ but$', 'not': r'^ not$', 'this': r'^ this$',
        'which': r'^ which$', 'or': r'^ or$', 'had': r'^ had$', 'they': r'^ they$'}


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
    def pre(mod, args):
        f = args[0].float()
        p = f @ V2
        return ((f - (p - mu) @ V2.T).to(args[0].dtype),) + tuple(args[1:])
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
    print(f'SITE GENERALITY: mlp{SITE}, {len(PATS)} classes never measured here. {NROWS} canonical '
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

    pa = r >= 0.30
    pb = pval < 0.05
    pc = npos_pos >= 10

    print(f'\n  HELD-OUT rho(|lam1/lam2|, relative CE rise) = {r:+.4f}', flush=True)
    print(f'  two-sided sampled permutation p ({NPERM}) = {pval:.5f}', flush=True)
    print(f'  same twelve classes at mlp11 gave {MLP11_RHO:+.4f} (§1647, p .019)', flush=True)
    print(f'  classes with positive CE rise: {npos_pos}/{n}  (§1644 had 11/12)', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'classes': "S1647's twelve function words, never measured at this site",
                      'single_registered_hypothesis': 'abs(lam1/lam2) predicts relative CE rise',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)',
                      'same_classes_at_mlp11_rho': MLP11_RHO},
           'per_class': per, 'heldout_rho': round(r, 4),
           'permutation_p_two_sided': pval, 'n_permutations': NPERM,
           'n_positive_rise': npos_pos,
           'predictions': {'pred_a_rho_ge_030_at_mlp14': bool(pa),
                           'pred_b_perm_p_lt_05': bool(pb),
                           'pred_c_10of12_positive_rise_mlp14': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
