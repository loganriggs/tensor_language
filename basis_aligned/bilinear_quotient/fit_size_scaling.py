# fit_size_scaling: IS THE ADDITIVE FAMILY ESTIMATION-LIMITED? — testing §1673's hypothesis
#
# §1673 found the additive program y = b(token) + xW at 54.35% under matched coverage,
# below both of its own special cases (all-linear 57.99%, tables-at-mlp0-2 57.29%). A
# strictly richer family cannot be less expressive, so the loss must come from estimating
# it. The hypothesis I registered there, explicitly as a hypothesis and not a finding:
#
#   the additive arm fits a per-token table at ALL EIGHTEEN sites -- 6009 tokens x 1152
#   dims from 24576 fit positions, about four positions per token. At mlp0-2 the token
#   really does determine the output (ceilings 90/96/77) so that estimator is sound. At the
#   other fifteen sites the token explains almost nothing, b is close to noise, and W is
#   then fitted on the residual of a noisy b.
#
# If that is right, more fit data must help the additive family substantially more than it
# helps the linear family, whose 1152x1152 map is already well determined by 24576
# positions. If additive does NOT improve with 5x the data, the hypothesis is wrong and the
# loss is something else -- and §1673's explanation has to be withdrawn.
#
# DESIGN, with the offset confound controlled rather than ignored. The large cache
# (n480_skip80) starts at a different document offset than the fit set used all afternoon
# (n96_skip1200), so size and offset would otherwise vary together:
#     n96_skip1200  -> n96_skip80    isolates OFFSET at fixed size  (the control)
#     n96_skip80    -> n480_skip80   isolates SIZE at fixed offset  (the test)
# Eval is fineweb_n192_skip7000 throughout, disjoint from all three.
# Every arm uses the hybrid coverage policy so family is the only other variable, and the
# coverage mask is refitted per fit-set -- a larger fit set covers more tokens, which is
# part of what "more data" buys and must not be held artificially fixed.
#
# Registered predictions:
#   pred_a THE ADDITIVE FAMILY IS ESTIMATION-LIMITED: its ceiling at n480 exceeds its
#          ceiling at n96 (same offset) by >= 5 percentage points.
#   pred_b AND MORE SO THAN THE LINEAR FAMILY: additive's n96->n480 gain exceeds linear's.
#          If linear gains as much, the run is measuring fit-set size in general and says
#          nothing about the additive family specifically.
#   pred_c CONTROL -- offset is not the variable: at fixed n96, every family's ceiling moves
#          by <= 2 points between skip1200 and skip80.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fit_size_scaling_results.json'
FIT_SETS = [('n96_skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt'),
            ('n96_skip80', PT + '.rowcache/fineweb_n96_skip80.pt'),
            ('n480_skip80', PT + '.rowcache/fineweb_n480_skip80.pt')]
FIT_ROWS = FIT_SETS[0][1]
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1672_BEST_PURE = 0.5729
S1673_HYBRID = {'linear': 0.5799, 'table_mlp0_2': 0.5729, 'additive': 0.5435}
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
    print(f'FIT SIZE SCALING | testing §1673\'s estimation hypothesis | ridge {RIDGE} | '
          f'eval fineweb_n192_skip7000 (disjoint from every fit set)', flush=True)

    arms = {
        'linear': {L: 'linear_hybrid' for L in ALL18},
        'table_mlp0_2': {L: ('table' if L < 3 else 'linear') for L in ALL18},
        'additive': {L: 'additive_hybrid' for L in ALL18},
    }
    out = {}
    for fname, fpath in FIT_SETS:
        fit = load(fpath)
        seen = seen_mask(fit)
        SEENREF['m'] = seen
        ntok = int(seen.sum())
        cl = ce(ev, seen)
        cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
            (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
                K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
        st = cc - cl
        row = {'fit_rows': tuple(fit.shape), 'tokens_covered': ntok, 'stake': round(st, 5)}
        print(f'  {fname:14s} {fit.shape[0]:4d} rows | {ntok} tokens covered | '
              f'stake {st:.4f}', flush=True)
        for name, kinds in arms.items():
            prog = compile_program(fit, kinds, seen)
            ct = ce(ev, seen, hooks=install(prog))
            row[name] = round((cc - ct) / st if st > 1e-6 else float('nan'), 5)
            print(f'      {name:14s} CEILING {row[name]:7.2%}', flush=True)
            del prog
            torch.cuda.empty_cache()
        out[fname] = row
        del fit
        torch.cuda.empty_cache()

    small, large = out['n96_skip80'], out['n480_skip80']
    d_add = large['additive'] - small['additive']
    d_lin = large['linear'] - small['linear']
    d_tab = large['table_mlp0_2'] - small['table_mlp0_2']
    offset = {k: abs(out['n96_skip80'][k] - out['n96_skip1200'][k]) for k in arms}

    pa = d_add >= 0.05
    pb = d_add > d_lin
    pc = all(v <= 0.02 for v in offset.values())

    print(f'\n  SIZE EFFECT (n96 -> n480, same offset): additive {d_add:+.2%} | '
          f'linear {d_lin:+.2%} | table {d_tab:+.2%}', flush=True)
    print(f'  additive estimation-limited {pa} | more than linear {pb}', flush=True)
    print(f'  OFFSET CONTROL (n96 skip1200 vs skip80): ' +
          '  '.join(f'{k} {v:.2%}' for k, v in offset.items()) + f'  -> {pc}', flush=True)
    print(f'  at n480 the ordering is: ' + '  '.join(
        f'{k} {large[k]:.2%}' for k in sorted(arms, key=lambda k: -large[k])), flush=True)
    print(f'  (§1673 at n96_skip1200: ' +
          '  '.join(f'{k} {v:.2%}' for k, v in S1673_HYBRID.items()) + ')', flush=True)

    res = {'config': {'fit_sets': [f for f, _ in FIT_SETS], 'eval_rows': 'fineweb_n192_skip7000.pt',
                      'ridge': RIDGE, 'compilation': 'bottom-up (§1669)',
                      'coverage': 'hybrid for every arm, mask refitted per fit set',
                      'design': 'skip1200->skip80 at n96 isolates OFFSET; n96->n480 at skip80 isolates SIZE',
                      'hypothesis_under_test': "§1673 -- the additive family is estimation-limited "
                                               "because it fits a per-token table at all 18 sites",
                      's1673_hybrid': S1673_HYBRID},
           'fit_sets': out,
           'size_effect': {'additive': round(d_add, 5), 'linear': round(d_lin, 5),
                           'table_mlp0_2': round(d_tab, 5)},
           'offset_effect': {k: round(v, 5) for k, v in offset.items()},
           'predictions': {'pred_a_additive_estimation_limited_ge_5pts': bool(pa),
                           'pred_b_more_than_linear': bool(pb),
                           'pred_c_offset_control_le_2pts': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
