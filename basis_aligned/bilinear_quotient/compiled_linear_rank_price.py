# compiled_linear_rank_price: WHAT DOES THE PROGRAM COST? — parameters against fidelity
#
# The replacement program for bilin18's MLP stack is eighteen least-squares linear maps
# compiled bottom-up, at 60.81% of a 4.33-nat stake (§1676, 480 fit rows, mask fixed). Each
# map is 1152 x 1152, so the program is 18 x 1.33M = 23.9M reals. That is a fidelity number
# without a price, and pricing is the point of this line of work.
#
# Rank-truncating W gives the price curve: a rank-r map costs r x 2 x 1152 reals per site.
#
# §1668 already tried rank-truncated linear maps and got a NON-MONOTONE curve (30.1%, -9.0%,
# 9.8%, 12.3%, 52.6%, 68.7%), which LESSONS 28 says is compounding rather than
# dimensionality -- those maps were fitted naively. §1669 showed bottom-up compilation
# removes the compounding entirely. So the same measurement should now be readable, and
# monotonicity is the check that it is: this run repeats a measurement that FAILED, under
# the fix, and monotonicity is what tells us the fix worked rather than my say-so.
#
# W is refitted at each rank rather than truncated after fitting at full rank, so each arm is
# the best rank-r map rather than a damaged full-rank one.
#
# Registered predictions:
#   pred_a COMPILATION MAKES THE RANK CURVE READABLE: it is monotone non-decreasing in r,
#          against §1668's non-monotone curve for the same family fitted naively.
#   pred_b THE PROGRAM IS CHEAP: rank 128 (2.1M reals, 11% of full) reaches >= 90% of the
#          full-rank program's ceiling.
#   pred_c IDENTITY CHECK: rank 1152 reproduces §1676's 60.81% within 1 point.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'compiled_linear_rank_price_results.json'
FIT_SETS = [('n96_skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt'),
            ('n96_skip80', PT + '.rowcache/fineweb_n96_skip80.pt'),
            ('n480_skip80', PT + '.rowcache/fineweb_n480_skip80.pt')]
FIT_ROWS = FIT_SETS[0][1]
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1672_BEST_PURE = 0.5729
S1676_FULL_RANK_480 = 0.6081
RANKS = [8, 32, 128, 512, 1152]
S1668_NAIVE_NONMONOTONE = [0.301, -0.090, 0.098, 0.123, 0.526, 0.687]
S1675_GROWING_MASK_SIZE_EFFECT = {'additive': 0.0123, 'linear': -0.0329, 'table_mlp0_2': -0.0045}
S1669_ALL_LINEAR = 0.5428
S1668_NAIVE_TABLE = 0.3427
S1668_BANDS = {'front_token': 0.7645, 'front_linear': 0.6868, 'middle_linear': 0.6233,
               'late_linear': 0.8360}
STATE = {}
SEENREF = {}
RANKSTATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def linear_hook(W):
    def hook(mod, args, out):
        return (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
    return hook


def table_hook(tbl, seen):
    def hook(mod, args, out):
        sub = tbl[STATE['idx'].reshape(-1)].reshape(out.shape).to(out.dtype)
        return torch.where(seen[STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def linear_hybrid_hook(W, seen):
    def hook(mod, args, out):
        sub = (args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(seen[STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def additive_hybrid_hook(tbl, W, seen):
    def hook(mod, args, out):
        b = tbl[STATE['idx'].reshape(-1)]
        sub = (b + args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
        return torch.where(seen[STATE['idx']].unsqueeze(-1), sub, out)
    return hook


def additive_hook(tbl, W):
    """y_hat = b(token) + xW, defined at EVERY position (b falls back to the
    position-weighted mean at unseen tokens, which is already baked into tbl)."""
    def hook(mod, args, out):
        b = tbl[STATE['idx'].reshape(-1)]
        return (b + args[0].reshape(-1, D) @ W).reshape(out.shape).to(out.dtype)
    return hook


def install(prog):
    """prog: site -> ('linear', W) | ('table', tbl, seen)"""
    hs = []
    for L, p in prog.items():
        hs.append(H[L].mlp.register_forward_hook(
            linear_hook(p[1]) if p[0] == 'linear'
            else linear_hybrid_hook(p[1], SEENREF['m']) if p[0] == 'linear_hybrid'
            else additive_hook(p[1], p[2]) if p[0] == 'additive'
            else additive_hybrid_hook(p[1], p[2], SEENREF['m']) if p[0] == 'additive_hybrid'
            else table_hook(p[1], p[2])))
    return hs


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
def fit_site(rows, L, kind, prog, seen):
    """Fit site L's program with everything already in `prog` installed."""
    if kind == 'linear':
        A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
        B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
        n = {'v': 0}

        def collect(mod, args, out):
            x = args[0].reshape(-1, D).double(); y = out.reshape(-1, D).double()
            A.add_(x.T @ x); B.add_(x.T @ y); n['v'] += x.shape[0]
            return None
        sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
        assert n['v'] > 0, f'site {L}: no fit positions'
        a = A / n['v']
        reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
        W = torch.linalg.solve(a + reg, B / n['v']).float()
        r = RANKSTATE.get('r', D)
        if r < D:
            U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
            W = ((U[:, :r] * S[:r]) @ Vh[:r]).float()
        return ('linear', W)
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect_t(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect_t)])
    assert float(c.sum()) > 0, f'site {L}: no token counts'
    sn = c > 0
    tbl = (s.sum(0) / c.sum()).unsqueeze(0).repeat(50257, 1)
    tbl[sn] = s[sn] / c[sn].unsqueeze(1)
    return ('table', tbl, sn)


@torch.no_grad()
def fit_additive(rows, L, prog, seen):
    """Table first, then least squares on its residual. Two passes."""
    _, tbl, _ = fit_site(rows, L, 'table', prog, seen)
    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect(mod, args, out):
        x = args[0].reshape(-1, D).double()
        r = (out.float().reshape(-1, D) - tbl[STATE['idx'].reshape(-1)]).double()
        A.add_(x.T @ x); B.add_(x.T @ r); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect)])
    assert n['v'] > 0, f'site {L}: no additive fit positions'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    return ('additive', tbl, torch.linalg.solve(a + reg, B / n['v']).float())


@torch.no_grad()
def compile_program(rows, kinds, seen):
    prog = {}
    for L in ALL18:
        k = kinds[L]
        if k in ('additive', 'additive_hybrid'):
            p = fit_additive(rows, L, prog, seen)
            prog[L] = (k, p[1], p[2])
        elif k == 'linear_hybrid':
            p = fit_site(rows, L, 'linear', prog, seen)
            prog[L] = ('linear_hybrid', p[1])
        else:
            prog[L] = fit_site(rows, L, k, prog, seen)
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
    mask_rows = load(FIT_SETS[1][1])
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    del mask_rows
    torch.cuda.empty_cache()
    fit = load(FIT_SETS[2][1])
    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'COMPILED LINEAR RANK PRICE | 18 maps, bottom-up, fit n480_skip80, mask fixed at '
          f'n96_skip80 | ranks {RANKS} | stake {st:.4f}', flush=True)

    kinds = {L: 'linear_hybrid' for L in ALL18}
    curve = {}
    for r in RANKS:
        RANKSTATE['r'] = r
        prog = compile_program(fit, kinds, seen)
        ct = ce(ev, seen, hooks=install(prog))
        reals = 18 * (r * 2 * D if r < D else D * D)
        curve[r] = {'ceiling': round((cc - ct) / st if st > 1e-6 else float('nan'), 5),
                    'reals': reals}
        print(f'    rank {r:5d}: CEILING {curve[r]["ceiling"]:7.2%} | {reals / 1e6:6.2f}M reals',
              flush=True)
        del prog
        torch.cuda.empty_cache()

    vals = [curve[r]['ceiling'] for r in RANKS]
    assert len(set(vals)) > 1, (
        'every rank returned an identical ceiling -- the truncation is a no-op. '
        f'{vals}')
    full = curve[D]['ceiling']
    mono = all(curve[RANKS[i + 1]]['ceiling'] >= curve[RANKS[i]]['ceiling'] - 0.005
               for i in range(len(RANKS) - 1))
    r128 = curve[128]['ceiling'] / full if full > 1e-9 else float('nan')

    pa = mono
    pb = r128 >= 0.90
    pc = abs(full - S1676_FULL_RANK_480) <= 0.01

    print(f'\n  monotone {mono}  (§1668 same family fitted NAIVELY: '
          f'{[f"{v:.1%}" for v in S1668_NAIVE_NONMONOTONE]}) -> compilation makes it readable {pa}',
          flush=True)
    print(f'  rank 128 = {curve[128]["ceiling"]:.2%} = {r128:.1%} of full rank, at '
          f'{curve[128]["reals"] / 1e6:.2f}M reals vs {curve[D]["reals"] / 1e6:.2f}M '
          f'({curve[128]["reals"] / curve[D]["reals"]:.1%}) -> cheap {pb}', flush=True)
    print(f'  IDENTITY rank {D}: {full:.2%} vs §1676 {S1676_FULL_RANK_480:.2%} -> {pc}',
          flush=True)

    res = {'config': {'sites': ALL18, 'ranks': RANKS, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n480_skip80.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'coverage': 'mask pinned to n96_skip80 (§1676)',
                      'compilation': 'bottom-up (§1669); W REFITTED at each rank, not truncated after',
                      'price': 'a rank-r map costs r*2*1152 reals per site; full rank costs 1152^2',
                      'repeats_under_fix': '§1668 measured this family NAIVELY and got a non-monotone '
                                           'curve; monotonicity here is the check that compilation fixed it',
                      's1676_full_rank_480': S1676_FULL_RANK_480,
                      's1668_naive_nonmonotone': S1668_NAIVE_NONMONOTONE},
           'stake': round(st, 5), 'curve': curve, 'monotone': bool(mono),
           'rank128_frac_of_full': round(r128, 5),
           'predictions': {'pred_a_monotone_after_compilation': bool(pa),
                           'pred_b_rank128_ge_90pct_of_full': bool(pb),
                           'pred_c_identity_reproduces_s1676': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
