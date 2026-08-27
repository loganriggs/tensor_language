# mid_band_linear_program: IF THE MIDDLE BAND IS NOT A TOKEN TABLE, IS IT A LINEAR MAP?
#
# §1666 settled, with protocols matched and the instrument checked at 100%, that the
# middle twelve MLPs are 2.645 nats of computation that a per-token table reproduces only
# 21.73% of, against 76.45% for the front four. So the middle band is where bilin18 stops
# being a lookup. The next question is what it is instead, and the cheapest non-trivial
# family to try is the one directly above a table: a LINEAR MAP OF THE MODULE'S INPUT.
#
# That family is the right one to test next for a specific reason. bilin18's MLPs are
# PURE BILINEAR -- each output is a quadratic form in its input, with no elementwise
# nonlinearity anywhere. A least-squares linear map therefore isolates exactly the part of
# the module that is not quadratic. What the linear map fails to recover is the model's
# actual bilinearity doing work, not an artifact of a badly chosen basis.
#
# METHOD, for mlp4..mlp15 jointly (the §1666 band, same protocol so the numbers compare):
#   accumulate the normal equations A = sum x x^T, B = sum x y^T over the fit rows from
#   each module's own input x and output y; solve W = (A + lambda I)^-1 B; truncate W to
#   rank r by SVD; substitute y_hat = x W at every site simultaneously and measure the
#   joint ceiling against the same optimal-constant stake §1665 used.
#
# A linear map has NO COVERAGE GAP -- it is defined at every position, unlike a token
# table. So the §1661 hybrid hook is unnecessary here, and that is itself worth checking:
# the run reports the joint ceiling under BOTH all-position and covered-position scoring.
# If the two agree, the token-table comparison at 21.73% is not being helped or hurt by
# its coverage restriction. If they disagree, that gap is a property of the eval set and
# I want it visible rather than assumed away.
#
# Registered predictions:
#   pred_a THE MIDDLE BAND IS MORE LINEAR THAN IT IS TABULAR: the rank-64 linear map's
#          joint ceiling exceeds the token table's 21.73% by >= 20 percentage points.
#   pred_b BUT THE QUADRATIC PART IS DOING REAL WORK: even the FULL-rank linear map falls
#          short of 85%. If a linear map reproduces the band, the bilinearity is decorative
#          in the middle of this model, which would be a large and surprising claim.
#   pred_c MANIPULATION CHECK -- the rank axis is informative: rank 1 falls at least 20
#          points below full rank. A saturated curve would mean the rank readings carry no
#          information whichever way pred_a came out.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
BAND = list(range(4, 16))
RANKS = [1, 8, 32, 64, 256, 1152]
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mid_band_linear_program_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1666_TABLE = {'middle_joint_ceiling': 0.2173, 'front_joint_ceiling': 0.7645,
               'middle_joint_stake': 2.6453}
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
def fit_linear(rows):
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
def ce(rows, K, mode, Wt=None, seen=None):
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
    print(f'MID BAND LINEAR PROGRAM | sites mlp{BAND[0]}-mlp{BAND[-1]} | ranks {RANKS} | '
          f'ridge {RIDGE} | fit skip1200, eval skip7000', flush=True)

    W, nfit = fit_linear(fit)
    print(f'  normal equations from {nfit} fit positions per site', flush=True)

    cla, clc = ce(ev, K, 'live', seen=seen)
    cca, ccc = ce(ev, K, 'const', seen=seen)
    sa, sc = cca - cla, ccc - clc
    print(f'  joint stake: all-positions {sa:.4f} | covered-only {sc:.4f}  '
          f'(§1665 covered {S1666_TABLE["middle_joint_stake"]:.4f})', flush=True)

    curve = {}
    for r in RANKS:
        Wt = {L: truncate(W[L], r) for L in BAND}
        ta, tc = ce(ev, K, 'linear', Wt=Wt, seen=seen)
        ka = (cca - ta) / sa if sa > 1e-6 else float('nan')
        kc = (ccc - tc) / sc if sc > 1e-6 else float('nan')
        curve[r] = {'ceiling_all': round(ka, 5), 'ceiling_covered': round(kc, 5)}
        print(f'    rank {r:5d}: joint ceiling  all {ka:7.2%} | covered {kc:7.2%}', flush=True)

    full = curve[RANKS[-1]]['ceiling_covered']
    r64 = curve[64]['ceiling_covered']
    r1 = curve[1]['ceiling_covered']
    scoring_gap = max(abs(v['ceiling_all'] - v['ceiling_covered']) for v in curve.values())

    pa = (r64 - S1666_TABLE['middle_joint_ceiling']) >= 0.20
    pb = full < 0.85
    pc = (full - r1) >= 0.20

    print(f'\n  rank-64 LINEAR {r64:.2%}  vs  TOKEN TABLE {S1666_TABLE["middle_joint_ceiling"]:.2%} '
          f'(§1666, same band, same stake)  -> more linear than tabular {pa}', flush=True)
    print(f'  FULL-rank linear {full:.2%}  -> quadratic part still essential {pb}', flush=True)
    print(f'  largest all-vs-covered scoring gap across the curve: {scoring_gap:.2%} '
          f'(a linear map has no coverage gap; this bounds what the table comparison '
          f'inherits from its own)', flush=True)

    res = {'config': {'band': BAND, 'ranks': RANKS, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'fit': 'least squares from each module OWN input to its OWN output, '
                             'normal equations in float64',
                      'ablation': 'optimal constants, opt_ablation_consts_all.pt',
                      'why_linear': 'bilin18 MLPs are pure bilinear -- a linear map isolates '
                                    'exactly the non-quadratic part',
                      's1666_table_comparator': S1666_TABLE},
           'stake': {'all_positions': round(sa, 5), 'covered_only': round(sc, 5)},
           'curve': curve, 'max_scoring_gap': round(scoring_gap, 5),
           'predictions': {'pred_a_linear_beats_table_ge_20pts': bool(pa),
                           'pred_b_quadratic_essential_full_lt_85': bool(pb),
                           'pred_c_rank_axis_informative': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
