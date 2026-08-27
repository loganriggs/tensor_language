# band_linear_compare: THE PROGRAM-FAMILY TABLE, ALL BANDS, ONE PROTOCOL
#
# §1667 gave the middle band's family ladder: a per-token table reaches 21.73%, a
# full-rank linear map of the module input reaches 62.33%, and 37.67% of the band is
# irreducibly quadratic. That is one row of a table. This fills in the others.
#
# The point is not more numbers -- it is that "what kind of program is this module" only
# becomes answerable when the SAME families are priced at the SAME grain. §1665 and §1666
# together cost one wasted comparison to learn that; every arm here is a joint
# substitution over its band, scored the same way, against the same optimal-constant
# stake.
#
# bilin18's MLPs are pure bilinear, so the full-rank linear ceiling has a precise meaning:
# what it MISSES is the model's quadratic form doing work that no linear function of the
# same input can imitate. The complement, 1 - (full-rank linear ceiling), is therefore a
# direct price on the bilinearity, band by band.
#
# Registered predictions:
#   pred_a THE FRONT BAND IS ALSO MORE LINEAR THAN TABULAR: its full-rank linear ceiling
#          exceeds its 76.45% token ceiling by >= 5 points. If a token table beats a linear
#          map of the input at the front, the front band is genuinely a lookup rather than
#          a cheap computation that merely correlates with the token.
#   pred_b BILINEARITY EARNS ITS KEEP WITH DEPTH: the front band's quadratic remainder
#          (1 - full-rank linear) is smaller than the middle band's 37.67%.
#   pred_c CONTROL -- the middle band is re-run here as a replication: its full-rank
#          linear ceiling lands within 1 point of §1667's 62.33%. If the control drifts,
#          the cross-band comparison is not trustworthy whatever it shows.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
BANDS = {'front': list(range(0, 4)), 'middle': list(range(4, 16)),
         'late': list(range(16, 18)), 'all18': list(range(0, 18))}
BAND = BANDS['front']          # rebound per band in main()
RANKS = [1, 8, 32, 64, 256, 1152]
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'band_linear_compare_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
TABLE_CEILINGS = {'front': 0.7645, 'middle': 0.2173, 'late': 0.5102, 'all18': 0.3427}
S1667_MIDDLE_FULL_LINEAR = 0.6233
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = list(hooks)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            if score is not None:
                lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
                score(lg, bb[:, 1:].to(DEV), idx)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def fit_linear(rows, BAND):
    """Normal equations per site, from each module's own input and output."""
    A = {L: torch.zeros(D, D, device=DEV, dtype=torch.float64) for L in BAND}
    B = {L: torch.zeros(D, D, device=DEV, dtype=torch.float64) for L in BAND}
    n = {'v': 0}

    def mk(L):
        def hook(mod, args, out):
            x = args[0].reshape(-1, D).double()
            y = out.reshape(-1, D).double()
            A[L] += x.T @ x
            B[L] += x.T @ y
            if L == BAND[0]:
                n['v'] += x.shape[0]
            return None
        return hook
    sweep(rows, hooks=[H[L].mlp.register_forward_hook(mk(L)) for L in BAND])
    assert n['v'] > 0, 'no fit positions accumulated'
    W = {}
    for L in BAND:
        a = A[L] / n['v']
        reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
        W[L] = torch.linalg.solve(a + reg, B[L] / n['v']).float()
    return W, n['v']


@torch.no_grad()
def truncate(W, r):
    if r >= D:
        return W
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    return ((U[:, :r] * S[:r]) @ Vh[:r]).float()


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce(rows, K, BAND, mode, Wt=None, seen=None):
    """mode: live | const | linear. Scored on all positions and, if seen is given, on
    covered positions too -- both from the same forward pass."""
    hooks = []
    for L in BAND:
        if mode == 'const':
            hooks.append(H[L].mlp.register_forward_hook(
                (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
                    K[f'mlp{L}'].to(DEV).float())))
        elif mode == 'linear':
            def mk(L):
                def hook(mod, args, out):
                    return (args[0].reshape(-1, D) @ Wt[L]).reshape(out.shape).to(out.dtype)
                return hook
            hooks.append(H[L].mlp.register_forward_hook(mk(L)))
    acc = {'ta': 0.0, 'na': 0, 'tc': 0.0, 'nc': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        acc['ta'] += float(e.sum()); acc['na'] += e.numel()
        if seen is not None:
            cov = seen[idx[:, 64:]]
            acc['tc'] += float(e[cov].sum()); acc['nc'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return (acc['ta'] / max(acc['na'], 1),
            acc['tc'] / max(acc['nc'], 1) if seen is not None else None)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    seen = seen_mask(fit)
    print(f'BAND LINEAR COMPARE | ranks {RANKS} | ridge {RIDGE} | joint substitution per '
          f'band, covered-position scoring | fit skip1200, eval skip7000', flush=True)

    out = {}
    for name, band in BANDS.items():
        W, nfit = fit_linear(fit, band)
        cla, clc = ce(ev, K, band, 'live', seen=seen)
        cca, ccc = ce(ev, K, band, 'const', seen=seen)
        sc = ccc - clc
        curve = {}
        for r in RANKS:
            Wt = {L: truncate(W[L], r) for L in band}
            ta, tc = ce(ev, K, band, 'linear', Wt=Wt, seen=seen)
            curve[r] = round((ccc - tc) / sc if sc > 1e-6 else float('nan'), 5)
        del W
        torch.cuda.empty_cache()
        full = curve[RANKS[-1]]
        tbl = TABLE_CEILINGS[name]
        out[name] = {'sites': band, 'joint_stake': round(sc, 5), 'linear_curve': curve,
                     'full_rank_linear': full, 'token_table_ceiling': tbl,
                     'linear_minus_table': round(full - tbl, 5),
                     'quadratic_remainder': round(1.0 - full, 5)}
        print(f'  {name:7s} stake {sc:7.4f} | table {tbl:6.2%} | linear ' +
              '  '.join(f'r{r}={curve[r]:.1%}' for r in RANKS) +
              f' | QUADRATIC REMAINDER {1.0 - full:6.2%}', flush=True)

    pa = (out['front']['full_rank_linear'] - TABLE_CEILINGS['front']) >= 0.05
    pb = out['front']['quadratic_remainder'] < out['middle']['quadratic_remainder']
    pc = abs(out['middle']['full_rank_linear'] - S1667_MIDDLE_FULL_LINEAR) <= 0.01

    print(f'\n  PROGRAM-FAMILY TABLE (joint ceiling of the best member of each family)',
          flush=True)
    print(f'  {"band":8s} {"token":>8s} {"linear r64":>11s} {"linear full":>12s} '
          f'{"quadratic":>10s}', flush=True)
    for name in BANDS:
        r = out[name]
        print(f'  {name:8s} {r["token_table_ceiling"]:8.2%} {r["linear_curve"][64]:11.2%} '
              f'{r["full_rank_linear"]:12.2%} {r["quadratic_remainder"]:10.2%}', flush=True)
    print(f'\n  front linear beats its token table by '
          f'{out["front"]["linear_minus_table"]:+.2%} -> {pa}', flush=True)
    print(f'  quadratic remainder front {out["front"]["quadratic_remainder"]:.2%} vs middle '
          f'{out["middle"]["quadratic_remainder"]:.2%} -> bilinearity grows with depth {pb}',
          flush=True)
    print(f'  CONTROL middle full-rank linear {out["middle"]["full_rank_linear"]:.2%} vs '
          f'§1667 {S1667_MIDDLE_FULL_LINEAR:.2%} -> replicates {pc}', flush=True)

    res = {'config': {'bands': BANDS, 'ranks': RANKS, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'fit': 'least squares, each module OWN input to OWN output, float64 normal equations',
                      'scoring': 'covered positions only, joint substitution per band',
                      'token_table_comparators': TABLE_CEILINGS,
                      'quadratic_remainder_meaning': 'bilin18 MLPs are pure bilinear, so 1 - (full-rank '
                                                     'linear ceiling) prices what the quadratic form does '
                                                     'that no linear map of the same input can imitate'},
           'bands': out,
           'predictions': {'pred_a_front_linear_beats_table_ge_5pts': bool(pa),
                           'pred_b_bilinearity_grows_with_depth': bool(pb),
                           'pred_c_middle_control_replicates': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
