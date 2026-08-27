# attn_lag_family: HOW MUCH OF ATTENTION IS JUST THE PREVIOUS TOKEN?
#
# §1682 fitted a POSITION-WISE linear map to each attention output write and got 16.38%,
# leaving the write 83.6% non-local. That is a floor set by the family, not by attention:
# a map that can only see position t cannot express anything attention does across positions,
# so 16.38% is what survives of a cross-position module under a strictly local description.
#
# The cheapest family that can see across positions is one extra slot. §843 found attn0
# writes the PREVIOUS token's identity into the stream, and previous-token heads are the
# best-attested structure in this ledger, so lag 1 is the specific hypothesis rather than an
# arbitrary widening.
#
# ARMS, each a least-squares map compiled bottom-up with the coverage mask pinned:
#   lag0        y = x_t W                    -- §1682's family, reproduced as a control
#   lag1        y = [x_t, x_{t-1}] W         -- the hypothesis
#   lag8        y = [x_t, x_{t-8}] W         -- THE CONTROL THAT MATTERS. Same parameter
#               count, same architecture, a position with no privileged relation. Without it,
#               any gain from lag1 could just be "a second slot helps".
# Position 0 has no predecessor; the lagged slot is zero-padded there, and those positions are
# outside the scored window (>= 64) in any case.
#
# Registered predictions:
#   pred_a THE PREVIOUS TOKEN CARRIES REAL WEIGHT: lag1 exceeds lag0's 16.38% by >= 10
#          percentage points.
#   pred_b ATTENTION IS NOT MERELY PREVIOUS-TOKEN COPYING: lag1 stays below 50%. If two
#          positions reproduce half of attention, the non-locality in §1682 is much shallower
#          than that number suggests.
#   pred_c IT IS LAG 1 SPECIFICALLY, NOT JUST A SECOND SLOT: lag1 exceeds lag8 by >= 5
#          points. If lag8 does as well, the gain is extra parameters and the previous-token
#          reading is wrong.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
LAGS = [0, 1, 8]
S1682_LAG0 = 0.1638
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_lag_family_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
MLP_PROGRAM = {'linear_full_rank': 0.6081, 'rank128': 0.5412, 'stake': 4.3301}
STATE = {}
SEENREF = {}
LAGSTATE = {'lag': 0}


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


def lagged(x, lag):
    """[x_t, x_{t-lag}] with zero padding at the start; lag 0 returns x_t alone."""
    if lag == 0:
        return x.reshape(-1, D)
    p = torch.cat([torch.zeros_like(x[:, :lag]), x[:, :-lag]], dim=1)
    return torch.cat([x, p], dim=-1).reshape(-1, x.shape[-1] * 2)


def linear_hook(W, seen):
    def hook(mod, args, out):
        y = out_y(out)
        sub = (lagged(args[0], LAGSTATE['lag']) @ W).reshape(y.shape).to(y.dtype)
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
    din = D if LAGSTATE['lag'] == 0 else 2 * D
    A = torch.zeros(din, din, device=DEV, dtype=torch.float64)
    B = torch.zeros(din, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = lagged(args[0], LAGSTATE['lag']).double()
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
    print(f'ATTN LAG FAMILY | y = [x_t, x_(t-lag)] W on 18 attention output writes, compiled '
          f'bottom-up | lags {LAGS} | stake {st:.4f} nats', flush=True)
    print(f'  §1682 position-wise (lag 0) reference: {S1682_LAG0:.2%}', flush=True)

    curve = {}
    for lag in LAGS:
        LAGSTATE['lag'] = lag
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        curve[lag] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    lag {lag:2d}: CEILING {curve[lag]:8.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = [curve[l] for l in LAGS]
    assert len(set(vals)) > 1, f'every lag identical -- the lag is a no-op: {vals}'

    pa = (curve[1] - S1682_LAG0) >= 0.10
    pb = curve[1] < 0.50
    pc = (curve[1] - curve[8]) >= 0.05

    print(f'\n  lag1 {curve[1]:.2%} vs lag0 {curve[0]:.2%} ({curve[1] - curve[0]:+.2%}) '
          f'-> previous token carries real weight {pa}', flush=True)
    print(f'  lag1 vs lag8 {curve[8]:.2%} ({curve[1] - curve[8]:+.2%}) -> lag 1 specifically, '
          f'not just a second slot {pc}', flush=True)
    print(f'  lag1 below 50% -> not merely previous-token copying {pb}', flush=True)
    print(f'  remaining non-local share after lag 1: {1.0 - curve[1]:.2%}', flush=True)

    res = {'config': {'sites': ALL18, 'lags': LAGS, 'ridge': RIDGE,
                      'target': 'attention OUTPUT WRITE only; v1 passed through unchanged (§1682, §1684)',
                      'family': 'least squares from [x_t, x_(t-lag)] to the module output write',
                      'control': 'lag 8 has the same parameter count as lag 1 with no privileged relation',
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80 (§1676)',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      'padding': 'lagged slot zero-padded at the start; scored window is >= 64 anyway',
                      's1682_lag0': S1682_LAG0},
           'stake': round(st, 5), 'curve': curve,
           'lag1_minus_lag0': round(curve[1] - curve[0], 5),
           'lag1_minus_lag8': round(curve[1] - curve[8], 5),
           'remaining_nonlocal': round(1.0 - curve[1], 5),
           'predictions': {'pred_a_prev_token_ge_10pts': bool(pa),
                           'pred_b_not_merely_prev_token_lt_50': bool(pb),
                           'pred_c_lag1_specific_ge_5pts': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
