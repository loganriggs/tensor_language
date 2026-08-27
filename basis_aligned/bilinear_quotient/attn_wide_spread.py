# attn_wide_spread: IS ATTENTION'S REMAINING THIRD LONG-RANGE, OR OUT OF POSITIONAL REACH?
#
# §1686 left 31.95% of attention's output write outside a five-position linear description,
# and found that beyond lag 1 the dependence is DIFFUSE -- a spread window (1,2,4,8) at 68.05%
# beats a contiguous one (1,2,3,4) at 66.71% with the same slot count. Sampling context widely
# beats sampling it densely.
#
# That leaves two very different possibilities for the remaining third, and they are
# distinguishable: either it is genuinely LONG-RANGE, in which case widening the spread keeps
# buying ground, or it is out of reach of ANY fixed-position description -- content-dependent
# routing, which is what attention is for -- in which case the curve plateaus however many
# positions are added.
#
# ARMS, geometric spreads plus two controls:
#   1,2,4,8            §1686's best, CONTROL, must reproduce 68.05%
#   1,2,4,8,16
#   1,2,4,8,16,32
#   1,2,4,8,16,32,64   reaching a quarter of the 256-token window
#   2,4,8,16,32,64     SECOND CONTROL: the same spread with LAG 1 REMOVED. §1685 showed lag 1
#                      is worth 32.6 points on its own; if the wide arms are buying long-range
#                      structure rather than re-deriving lag 1 through correlated neighbours,
#                      dropping it should cost most of that.
#
# Registered predictions:
#   pred_a THE REMAINING THIRD IS AT LEAST PARTLY LONG-RANGE: (1,2,4,8,16,32,64) exceeds
#          §1686's 68.05% by >= 3 percentage points.
#   pred_b BUT IT PLATEAUS: the gain from width 5 to width 7 is smaller than the +11.79 that
#          (1,2,4,8) bought over (1,). If the curve climbs as fast at width 7 as at width 4,
#          the positional description has not saturated and no claim about an unreachable
#          residue is warranted.
#   pred_c LAG 1 IS NOT RECOVERABLE FROM ITS NEIGHBOURS: dropping it from the six-slot spread
#          costs >= 10 points.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
LAGSETS = [(1, 2, 4, 8), (1, 2, 4, 8, 16), (1, 2, 4, 8, 16, 32),
           (1, 2, 4, 8, 16, 32, 64), (2, 4, 8, 16, 32, 64)]
S1686_SPREAD4 = 0.6805
S1686_LAG1_STEP = 0.1179
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_wide_spread_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
MLP_PROGRAM = {'linear_full_rank': 0.6081, 'rank128': 0.5412, 'stake': 4.3301}
STATE = {}
SEENREF = {}
LAGSTATE = {'lags': ()}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def out_y(out):
    return out[0] if isinstance(out, tuple) else out


def repack(out, y):
    """Substitute the output write, pass v1 and anything else through UNCHANGED."""
    return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y


def const_hook(const):
    def hook(mod, args, out):
        y = out_y(out)
        return repack(out, const.to(y.dtype).expand_as(y))
    return hook


def lagged(x, lags):
    """[x_t, x_{t-l} for l in lags], zero-padded at the start. Empty lags -> x_t alone."""
    if not lags:
        return x.reshape(-1, D)
    parts = [x] + [torch.cat([torch.zeros_like(x[:, :l]), x[:, :-l]], dim=1) for l in lags]
    return torch.cat(parts, dim=-1).reshape(-1, D * (1 + len(lags)))


def linear_hook(W, seen):
    def hook(mod, args, out):
        y = out_y(out)
        sub = (lagged(args[0], LAGSTATE['lags']) @ W).reshape(y.shape).to(y.dtype)
        return repack(out, torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y))
    return hook


def install(prog):
    return [H[L].attn.register_forward_hook(linear_hook(W, SEENREF['m']))
            for L, W in prog.items()]


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
def fit_site(rows, L, prog):
    din = D * (1 + len(LAGSTATE['lags']))
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = torch.zeros(din, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = lagged(args[0], LAGSTATE['lags']).double()
        y = out_y(out).reshape(-1, D).double()
        A.add_(x.T @ x); B.add_(x.T @ y); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[L].attn.register_forward_hook(collect)])
    assert n['v'] > 0, f'attn{L}: no fit positions accumulated'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(din, device=DEV, dtype=torch.float64)
    return torch.linalg.solve(a + reg, B / n['v']).float()


@torch.no_grad()
def compile_stack(rows):
    prog = {}
    for L in ALL18:
        prog[L] = fit_site(rows, L, prog)
    return prog


@torch.no_grad()
def seen_mask(rows):
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        t = rows[i:i + 8, :-1].to(DEV).reshape(-1)
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    return c > 0


@torch.no_grad()
def ce(rows, seen, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def main():
    t0 = time.time()
    ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    mask_rows = load(MASK_ROWS)
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_ROWS)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].attn.register_forward_hook(
        const_hook(K[f'attn{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'ATTN WIDE SPREAD | is the remaining third long-range, or out of positional reach? | '
          f'stake {st:.4f} nats | control (1,2,4,8) must reproduce {S1686_SPREAD4:.2%}',
          flush=True)

    curve = {}
    for lags in LAGSETS:
        LAGSTATE['lags'] = lags
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        key = ','.join(str(l) for l in lags)
        curve[key] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    lags {key:22s} ({D * (1 + len(lags)):5d}-dim): CEILING {curve[key]:8.2%}',
              flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = list(curve.values())
    assert len(set(vals)) > 1, f'every lag set identical -- the lag input is a no-op: {vals}'

    c4 = curve['1,2,4,8']; c5 = curve['1,2,4,8,16']
    c6 = curve['1,2,4,8,16,32']; c7 = curve['1,2,4,8,16,32,64']
    no1 = curve['2,4,8,16,32,64']

    pa = (c7 - S1686_SPREAD4) >= 0.03
    pb = (c7 - c5) < S1686_LAG1_STEP
    pc = (c6 - no1) >= 0.10
    ctrl = abs(c4 - S1686_SPREAD4) <= 0.01

    print(f'\n  widest (7 slots) {c7:.2%} vs §1686 four-slot {S1686_SPREAD4:.2%} '
          f'({c7 - S1686_SPREAD4:+.2%}) -> partly long-range {pa}', flush=True)
    print(f'  width 5 -> width 7: {c7 - c5:+.2%} vs the §1686 step {S1686_LAG1_STEP:+.2%} '
          f'-> plateauing {pb}', flush=True)
    print(f'  DROP LAG 1 from the six-slot spread: {c6:.2%} -> {no1:.2%} ({no1 - c6:+.2%}) '
          f'-> lag 1 not recoverable from neighbours {pc}', flush=True)
    print(f'  CONTROL (1,2,4,8) reproduces §1686: {ctrl}', flush=True)
    print(f'  remaining outside the widest positional description: {1.0 - c7:.2%}', flush=True)

    res = {'config': {'sites': ALL18, 'lagsets': [list(l) for l in LAGSETS], 'ridge': RIDGE,
                      'target': 'attention OUTPUT WRITE only; v1 passed through unchanged',
                      'family': 'least squares from [x_t, x_(t-l) for l in lags] to the write',
                      'controls': ['(1,2,4,8) must reproduce §1686 68.05%',
                                   '(2,4,8,16,32,64) is the same spread with LAG 1 REMOVED'],
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1686_spread4': S1686_SPREAD4, 's1686_lag1_step': S1686_LAG1_STEP},
           'stake': round(st, 5), 'curve': curve,
           'widest_minus_spread4': round(c7 - S1686_SPREAD4, 5),
           'width5_to_width7': round(c7 - c5, 5),
           'cost_of_dropping_lag1': round(no1 - c6, 5),
           'remaining_outside_widest': round(1.0 - c7, 5),
           'control_reproduces': bool(ctrl),
           'predictions': {'pred_a_partly_long_range_ge_3pts': bool(pa),
                           'pred_b_plateauing': bool(pb),
                           'pred_c_lag1_not_recoverable_ge_10pts': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
