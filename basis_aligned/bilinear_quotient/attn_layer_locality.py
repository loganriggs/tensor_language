# attn_layer_locality: WHICH ATTENTION LAYERS ARE PREVIOUS-TOKEN MACHINES?
#
# §1685-§1687 priced attention's output write as a whole: 16.4% from the current position,
# +39.9% from the previous one (irreplaceable -- worth 27.9 points even against a six-slot
# spread), +13.8% from a handful of further positions where it barely matters which, and
# ~29.9% outside ANY fixed-position linear description.
#
# Those are eighteen modules averaged into one number. §843 located previous-token writing
# specifically at attn0, and the front/middle/late regimes on the MLP side turned out to be
# real per-site properties rather than band artifacts (§1672). So the obvious question is
# whether the lag-1 story is a property of attention or of a few early layers.
#
# METHOD, the §1671 exempt-one pattern: for each layer L, RECOMPILE the whole lag-1 program
# with L exempted and left live, and read off the gain over the full program. Recompiling per
# exemption rather than dropping one hook is not optional -- every map above L was fitted with
# L substituted, and un-substituting it would put them off-distribution and measure LESSONS 28
# instead of the layer.
#
# A layer with a LARGE gain is one the lag-1 program describes badly; a layer with a gain near
# zero is one it describes well. So the prediction is about where the gains are small.
#
# CONTROL: the no-exemption arm must reproduce §1685's 56.26%. Every number here is a
# difference against it.
#
# Registered predictions:
#   pred_a EARLY ATTENTION IS THE BEST DESCRIBED: the mean exemption gain over attn0-3 is
#          below the mean over attn4-17. §843 puts previous-token writing at attn0, and a
#          lag-1 family is exactly the right description for that.
#   pred_b THE SHORTFALL IS CONCENTRATED, unlike the three diffuse results this arc has
#          already produced (§1663, §1671, §1686): the worst single layer accounts for >= 20%
#          of the program's 43.74-point shortfall, against 5.6% under a uniform split.
#   pred_c CONTROL -- the no-exemption arm lands within 1 point of §1685's 56.26%.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
S1685_LAG1 = 0.5626
S1682_LAG0 = 0.1638
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'attn_layer_locality_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
MASK_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
MLP_PROGRAM = {'linear_full_rank': 0.6081, 'rank128': 0.5412, 'stake': 4.3301}
STATE = {}
SEENREF = {}
LAGSTATE = {'lag': 1}
EXEMPT = {'L': None}


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
        if L == EXEMPT['L']:
            continue
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
    print(f'ATTN LAYER LOCALITY | lag-1 program, RECOMPILED per exemption (§1671) | '
          f'stake {st:.4f} nats | control must reproduce §1685 {S1685_LAG1:.2%}', flush=True)

    def ceiling_for(exempt):
        EXEMPT['L'] = exempt
        prog = compile_stack(fit)
        ct = ce(ev, seen, hooks=install(prog))
        del prog
        torch.cuda.empty_cache()
        return (cc - ct) / st if st > 1e-6 else float('nan')

    base = ceiling_for(None)
    shortfall = 1.0 - base
    print(f'  full lag-1 program: CEILING {base:7.2%} | shortfall {shortfall:.2%}', flush=True)

    gains = {}
    for L in ALL18:
        c = ceiling_for(L)
        gains[f'attn{L}'] = {'ceiling_exempt': round(c, 5), 'gain': round(c - base, 5),
                             'share_of_shortfall': round((c - base) / shortfall, 4)
                             if shortfall > 1e-9 else None}
        print(f'    exempt attn{L:<2d} ceiling {c:7.2%}  gain {c - base:+7.2%}  = '
              f'{(c - base) / shortfall:6.2%} of the shortfall', flush=True)

    vals = [g['gain'] for g in gains.values()]
    assert len(set(vals)) > 1, f'every layer identical -- the exemption is a no-op: {vals}'

    early = sum(gains[f'attn{L}']['gain'] for L in range(0, 4)) / 4
    late = sum(gains[f'attn{L}']['gain'] for L in range(4, 18)) / 14
    rank = sorted(gains.items(), key=lambda kv: -kv[1]['gain'])
    top, topv = rank[0]

    pa = early < late
    pb = topv['share_of_shortfall'] >= 0.20
    pc = abs(base - S1685_LAG1) <= 0.01

    print(f'\n  mean exemption gain: attn0-3 {early:+.2%} | attn4-17 {late:+.2%} '
          f'-> early best described {pa}', flush=True)
    print(f'  worst layer {top} at {topv["gain"]:+.2%} = {topv["share_of_shortfall"]:.2%} of '
          f'the shortfall (uniform would be {1 / 18:.2%}) -> concentrated {pb}', flush=True)
    print(f'  top three: ' + ',  '.join(f'{k} {v["gain"]:+.2%}' for k, v in rank[:3]), flush=True)
    print(f'  best three: ' + ',  '.join(f'{k} {v["gain"]:+.2%}' for k, v in rank[-3:]), flush=True)
    print(f'  CONTROL {base:.2%} vs §1685 {S1685_LAG1:.2%} -> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'lag': 1, 'ridge': RIDGE,
                      'target': 'attention OUTPUT WRITE only; v1 passed through unchanged',
                      'method': 'exempt-one, RECOMPILING the whole stack per exemption so the maps '
                                'above the exempt layer are never applied off-distribution (LESSONS 28)',
                      'compilation': 'bottom-up (§1669)', 'coverage': 'mask pinned to n96_skip80',
                      'fit_rows': 'fineweb_n480_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      's1685_lag1': S1685_LAG1, 's1682_lag0': S1682_LAG0},
           'stake': round(st, 5), 'base_ceiling': round(base, 5),
           'shortfall': round(shortfall, 5), 'layers': gains,
           'mean_gain_early_attn0_3': round(early, 5),
           'mean_gain_rest_attn4_17': round(late, 5),
           'ranking': [k for k, _ in rank],
           'predictions': {'pred_a_early_best_described': bool(pa),
                           'pred_b_shortfall_concentrated_ge_20pct': bool(pb),
                           'pred_c_control_reproduces_s1685': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
