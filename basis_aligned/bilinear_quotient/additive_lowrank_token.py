# additive_lowrank_token: CURING THE ADDITIVE FAMILY'S ESTIMATION PROBLEM BY TRUNCATING b
#
# §1676 confirmed §1673's hypothesis with the coverage mask held fixed: the additive program
# y = b(token) + xW is ESTIMATION-LIMITED. Five times the fit data buys it +5.29 points,
# twice what the linear family gains (+2.64) and twenty-five times what the pure table gains
# (+0.21). Its problem is not expressiveness, it is that b has 6009 x 1152 free parameters
# fitted from ~4 positions per token.
#
# More data is the obvious cure and there is no more data cached. The other cure is to stop
# asking for so many parameters. §1664 measured the front MLPs' token tables at rank 64 of
# 1152 recovering 91.7-97.5% of the full table, so most of what b needs to say fits in a
# small subspace. A rank-r b costs r x (6009 + 1152) reals instead of 6009 x 1152 -- at
# r = 64, about ten times fewer.
#
# If the diagnosis is right, truncating b should RAISE the additive program at the small fit
# size, and the optimum should be INTERIOR -- a rank below full. That interior optimum is
# the discriminating prediction: more capacity monotonically helping would mean the family
# was capacity-limited after all and §1676 measured something else.
#
# METHOD: at each site, compiled bottom-up, fit b as in §1673 (per-token mean, then W by
# least squares on the residual y - b). Truncate b to rank r by SVD, centred on the
# position-weighted mean with rows scaled by sqrt(token count) so the truncation minimises
# POSITION-weighted error (§1664's protocol). Then REFIT W on the residual of the TRUNCATED
# b -- fitting W against the full b and then truncating b would leave W correcting a term
# that is no longer there.
#
# Everything runs at the SMALL fit set (n96_skip80) with the fixed n96_skip80 mask, so the
# comparators are §1676's own small-data column: additive 53.79%, linear 58.17%, table 56.95%.
#
# Registered predictions:
#   pred_a TRUNCATION CURES IT: the best rank-truncated additive beats full-rank additive
#          (53.79%) by >= 3 points at the same fit size.
#   pred_b AND IT OVERTAKES THE LINEAR FAMILY at that fit size, beating 58.17%.
#   pred_c THE OPTIMUM IS INTERIOR: the best rank is not the largest one tried. If capacity
#          helps monotonically, the family was not estimation-limited and §1676's reading
#          needs revisiting.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'additive_lowrank_token_results.json'
FIT_SETS = [('n96_skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt'),
            ('n96_skip80', PT + '.rowcache/fineweb_n96_skip80.pt'),
            ('n480_skip80', PT + '.rowcache/fineweb_n480_skip80.pt')]
FIT_ROWS = FIT_SETS[0][1]
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1672_BEST_PURE = 0.5729
S1676_SMALL = {'linear': 0.5817, 'table_mlp0_2': 0.5695, 'additive': 0.5379}
TOKEN_RANKS = [8, 32, 64, 256, 1152]
S1675_GROWING_MASK_SIZE_EFFECT = {'additive': 0.0123, 'linear': -0.0329, 'table_mlp0_2': -0.0045}
S1669_ALL_LINEAR = 0.5428
S1668_NAIVE_TABLE = 0.3427
S1668_BANDS = {'front_token': 0.7645, 'front_linear': 0.6868, 'middle_linear': 0.6233,
               'late_linear': 0.8360}
STATE = {}
SEENREF = {}


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
        return ('linear', torch.linalg.solve(a + reg, B / n['v']).float())
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
def truncate_table(tbl, cnt, mean, r):
    """Rank-r approximation of the seen rows, minimising POSITION-weighted error (§1664)."""
    sn = cnt > 0
    if r >= D:
        return tbl
    rows_c = tbl[sn] - mean.unsqueeze(0)
    w = cnt[sn].sqrt().unsqueeze(1)
    U, S, Vh = torch.linalg.svd(rows_c * w, full_matrices=False)
    k = min(r, S.numel())
    out = tbl.clone()
    out[sn] = ((U[:, :k] * S[:k]) @ Vh[:k] / w) + mean.unsqueeze(0)
    return out


@torch.no_grad()
def fit_additive_rank(rows, L, prog, seen, rank):
    """Table, truncated to `rank`, then W refitted on the TRUNCATED residual."""
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
    mean = s.sum(0) / c.sum()
    tbl = mean.unsqueeze(0).repeat(50257, 1)
    tbl[sn] = s[sn] / c[sn].unsqueeze(1)
    tbl = truncate_table(tbl, c, mean, rank)

    A = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    B = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'v': 0}

    def collect_w(mod, args, out):
        x = args[0].reshape(-1, D).double()
        r = (out.float().reshape(-1, D) - tbl[STATE['idx'].reshape(-1)]).double()
        A.add_(x.T @ x); B.add_(x.T @ r); n['v'] += x.shape[0]
        return None
    sweep(rows, hooks=install(prog) + [H[L].mlp.register_forward_hook(collect_w)])
    assert n['v'] > 0, f'site {L}: no additive fit positions'
    a = A / n['v']
    reg = RIDGE * torch.diag(a).mean() * torch.eye(D, device=DEV, dtype=torch.float64)
    return ('additive_hybrid', tbl, torch.linalg.solve(a + reg, B / n['v']).float())


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
def compile_rank(rows, seen, rank):
    prog = {}
    for L in ALL18:
        prog[L] = fit_additive_rank(rows, L, prog, seen, rank)
    return prog


@torch.no_grad()
def main():
    t0 = time.time()
    ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    fit = load(FIT_SETS[1][1])
    seen = seen_mask(fit)
    SEENREF['m'] = seen
    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'ADDITIVE, LOW-RANK TOKEN TERM | fit n96_skip80 ({fit.shape[0]} rows, '
          f'{int(seen.sum())} tokens) | ranks {TOKEN_RANKS} | stake {st:.4f}', flush=True)
    print(f'  comparators at this fit size (§1676): additive {S1676_SMALL["additive"]:.2%} | '
          f'linear {S1676_SMALL["linear"]:.2%} | table {S1676_SMALL["table_mlp0_2"]:.2%}',
          flush=True)

    curve = {}
    for r in TOKEN_RANKS:
        prog = compile_rank(fit, seen, r)
        ct = ce(ev, seen, hooks=install(prog))
        curve[r] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
        print(f'    b rank {r:5d}: CEILING {curve[r]:7.2%}   '
              f'({curve[r] - S1676_SMALL["additive"]:+.2%} vs full-rank additive)', flush=True)
        del prog
        torch.cuda.empty_cache()

    best_r = max(curve, key=lambda r: curve[r])
    best = curve[best_r]

    pa = (best - S1676_SMALL['additive']) >= 0.03
    pb = best > S1676_SMALL['linear']
    pc = best_r != max(TOKEN_RANKS)

    print(f'\n  BEST rank {best_r} at {best:.2%}', flush=True)
    print(f'    vs full-rank additive {S1676_SMALL["additive"]:.2%} '
          f'({best - S1676_SMALL["additive"]:+.2%}) -> truncation cures it {pa}', flush=True)
    print(f'    vs linear {S1676_SMALL["linear"]:.2%} '
          f'({best - S1676_SMALL["linear"]:+.2%}) -> overtakes linear {pb}', flush=True)
    print(f'    optimum interior (not rank {max(TOKEN_RANKS)}) -> {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ranks': TOKEN_RANKS, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n96_skip80.pt', 'eval_rows': 'fineweb_n192_skip7000.pt',
                      'program': 'y_hat = b_rank_r(token) + xW, W REFITTED on the truncated residual',
                      'truncation': 'SVD centred on the position-weighted mean, rows scaled by '
                                    'sqrt(count) so it minimises POSITION-weighted error (§1664)',
                      'compilation': 'bottom-up (§1669)',
                      'coverage': 'hybrid, mask fixed at n96_skip80',
                      'hypothesis_under_test': '§1676 -- the additive family is estimation-limited, '
                                               'so reducing b\'s parameter count should help at small data',
                      's1676_small_fit_comparators': S1676_SMALL},
           'stake': round(st, 5), 'curve': curve, 'best_rank': best_r, 'best_ceiling': best,
           'predictions': {'pred_a_truncation_cures_ge_3pts': bool(pa),
                           'pred_b_overtakes_linear': bool(pb),
                           'pred_c_optimum_is_interior': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
