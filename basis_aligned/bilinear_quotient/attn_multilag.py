# attn_multilag: HOW MANY RECENT POSITIONS DOES ATTENTION'S WRITE NEED?
#
# §1685 found that one previous position takes attention's output write from 16.38% to
# 56.26%, with 32.6 of those 39.9 points specific to lag 1 rather than to having a second
# slot. 43.74% still requires something wider than two positions. This asks how much wider.
#
# ARMS -- sets of lags, each a least-squares map from the concatenated slots, compiled
# bottom-up with the mask pinned:
#   ()            x_t alone                   -- CONTROL, must reproduce 16.38%
#   (1,)          x_t, x_{t-1}                -- CONTROL, must reproduce 56.26%
#   (1,2)         two positions back
#   (1,2,3,4)     four recent positions
#   (1,2,4,8)     THE CONTROL THAT MATTERS. Same slot count as (1,2,3,4) but spread. If a
#                 spread window does as well, what attention needs is "some history", not
#                 "the immediately preceding tokens", and the contiguity reading is wrong.
#
# Two controls at the bottom of the ladder rather than one, because every claim here is a
# difference against them and §1684 showed how easily an arm can be degenerate without
# looking it.
#
# Registered predictions:
#   pred_a DIMINISHING RETURNS: (1,2,3,4) beats (1,) by >= 5 points but by LESS than the
#          +39.88 that (1,) bought over (). The first position back is the cliff.
#   pred_b FOUR RECENT POSITIONS GET MOST OF IT: (1,2,3,4) exceeds 70%.
#   pred_c CONTIGUITY MATTERS: (1,2,3,4) beats the spread (1,2,4,8) by >= 3 points. If not,
#          the window is about having history rather than recency and §1685's lag-1 reading
#          needs qualifying.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
LAGSETS = [(), (1,), (1, 2), (1, 2, 3, 4), (1, 2, 4, 8)]
S1682_LAG0 = 0.1638
S1685_LAG1 = 0.5626
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_multilag_results.json'
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
    print(f'ATTN MULTILAG | how many recent positions does the write need? | sets {LAGSETS} | '
          f'stake {st:.4f} nats', flush=True)
    print(f'  controls: () must reproduce {S1682_LAG0:.2%}, (1,) must reproduce '
          f'{S1685_LAG1:.2%}', flush=True)

    curve = {}
    for lags in LAGSETS:
        LAGSTATE['lags'] = lags
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        key = ','.join(str(l) for l in lags) if lags else 'none'
        curve[key] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    lags {key:10s} ({D * (1 + len(lags)):5d}-dim input): '
              f'CEILING {curve[key]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = list(curve.values())
    assert len(set(vals)) > 1, f'every lag set identical -- the lag input is a no-op: {vals}'

    c0, c1 = curve['none'], curve['1']
    c4, cs = curve['1,2,3,4'], curve['1,2,4,8']
    first_step = c1 - c0

    pa = ((c4 - c1) >= 0.05) and ((c4 - c1) < first_step)
    pb = c4 > 0.70
    pc = (c4 - cs) >= 0.03
    ctrl = (abs(c0 - S1682_LAG0) <= 0.01) and (abs(c1 - S1685_LAG1) <= 0.01)

    print(f'\n  first step (none -> 1): {first_step:+.2%}', flush=True)
    print(f'  three more recent (1 -> 1,2,3,4): {c4 - c1:+.2%} -> diminishing returns {pa}',
          flush=True)
    print(f'  four recent positions reach {c4:.2%} -> above 70% {pb}', flush=True)
    print(f'  contiguous (1,2,3,4) {c4:.2%} vs spread (1,2,4,8) {cs:.2%} '
          f'({c4 - cs:+.2%}) -> recency not just history {pc}', flush=True)
    print(f'  CONTROLS reproduce §1682 and §1685: {ctrl}', flush=True)
    print(f'  remaining after four recent positions: {1.0 - c4:.2%}', flush=True)

    res = {'config': {'sites': ALL18, 'lagsets': [list(l) for l in LAGSETS], 'ridge': RIDGE,
                      'target': 'attention OUTPUT WRITE only; v1 passed through unchanged',
                      'family': 'least squares from [x_t, x_(t-l) for l in lags] to the write',
                      'control': '(1,2,4,8) has the same slot count as (1,2,3,4) but spread, so '
                                 'contiguity is separated from mere history',
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1682_lag0': S1682_LAG0, 's1685_lag1': S1685_LAG1},
           'stake': round(st, 5), 'curve': curve,
           'first_step': round(first_step, 5), 'three_more': round(c4 - c1, 5),
           'contiguous_minus_spread': round(c4 - cs, 5),
           'remaining_after_four': round(1.0 - c4, 5),
           'controls_reproduce': bool(ctrl),
           'predictions': {'pred_a_diminishing_returns': bool(pa),
                           'pred_b_four_positions_gt_70': bool(pb),
                           'pred_c_contiguity_matters_ge_3pts': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
