# fit_size_fixed_mask: THE SAME TEST WITH THE COVERAGE MASK HELD FIXED
#
# §1675 tried to test whether the additive family is estimation-limited by refitting on 5x
# the data. The offset control passed cleanly (<=0.56 points), but the size arm is
# CONFOUNDED and I am not reading the hypothesis off it.
#
# The confound: the coverage mask is derived from the fit set, so going from 96 to 480 fit
# rows took coverage from 6009 to 16110 tokens. Under the hybrid policy that changes two
# things at once -- the programs are better estimated, AND far fewer positions are left
# LIVE, so the program has to do more of the work and gets less free help from the real
# module. The scored position set changes too. The symptom is unmistakable: the LINEAR arm
# got 3.29 points WORSE with five times the data, which no least-squares fit does. That is
# the measurement getting harder, not the fit getting worse.
#
# I chose that design deliberately, on the reasoning that wider coverage "is part of what
# more data buys". It is -- but it makes the size arm useless for the question actually
# being asked, which is about ESTIMATION alone.
#
# FIX: hold the coverage mask fixed at the n96_skip80 mask for every arm. The same
# positions are substituted, the same positions are left live, the same positions are
# scored. The only thing that varies is how much data the programs were estimated from.
#
# Registered predictions:
#   pred_a WITH THE MASK FIXED, THE ADDITIVE FAMILY IS ESTIMATION-LIMITED: its ceiling
#          improves by >= 3 points going from 96 to 480 fit rows. This is §1673's
#          hypothesis, tested cleanly. If it fails here, §1673's explanation is wrong and
#          I withdraw it rather than look for a third mechanism.
#   pred_b THE §1675 LINEAR DEGRADATION WAS THE MASK, NOT THE FIT: with the mask fixed, the
#          linear arm does not lose ground with more data (change >= -1 point), against the
#          -3.29 points it showed when the mask was allowed to grow.
#   pred_c CONTROL -- the n96 arm reproduces §1675's own n96_skip80 row within 0.5 points
#          for all three families.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'fit_size_fixed_mask_results.json'
FIT_SETS = [('n96_skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt'),
            ('n96_skip80', PT + '.rowcache/fineweb_n96_skip80.pt'),
            ('n480_skip80', PT + '.rowcache/fineweb_n480_skip80.pt')]
FIT_ROWS = FIT_SETS[0][1]
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1672_BEST_PURE = 0.5729
S1675_N96_SKIP80 = {'linear': 0.5817, 'table_mlp0_2': 0.5695, 'additive': 0.5379}
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

    # ONE mask for every arm, from the SMALL fit set at the shared offset.
    mask_rows = load(FIT_SETS[1][1])
    seen = seen_mask(mask_rows)
    SEENREF['m'] = seen
    ntok = int(seen.sum())
    del mask_rows
    torch.cuda.empty_cache()

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'FIT SIZE, MASK HELD FIXED | coverage mask from n96_skip80 for EVERY arm '
          f'({ntok} tokens) | stake {st:.4f} | eval n192_skip7000', flush=True)

    arms = {
        'linear': {L: 'linear_hybrid' for L in ALL18},
        'table_mlp0_2': {L: ('table' if L < 3 else 'linear') for L in ALL18},
        'additive': {L: 'additive_hybrid' for L in ALL18},
    }
    out = {}
    for fname, fpath in (FIT_SETS[1], FIT_SETS[2]):
        fit = load(fpath)
        row = {'fit_rows': tuple(fit.shape)}
        print(f'  fit {fname:12s} ({fit.shape[0]:4d} rows)', flush=True)
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
    delta = {k: large[k] - small[k] for k in arms}
    ctrl = {k: abs(small[k] - S1675_N96_SKIP80[k]) for k in arms}

    pa = delta['additive'] >= 0.03
    pb = delta['linear'] >= -0.01
    pc = all(v <= 0.005 for v in ctrl.values())

    print(f'\n  SIZE EFFECT with the mask FIXED (96 -> 480 rows):', flush=True)
    for k in arms:
        print(f'    {k:14s} {small[k]:7.2%} -> {large[k]:7.2%}   {delta[k]:+.2%}   '
              f'(§1675, mask growing: {S1675_GROWING_MASK_SIZE_EFFECT[k]:+.2%})', flush=True)
    print(f'  additive estimation-limited {pa} | §1675 linear drop was the mask {pb} | '
          f'control {pc}', flush=True)
    print(f'  ordering at 480 rows: ' + '  '.join(
        f'{k} {large[k]:.2%}' for k in sorted(arms, key=lambda k: -large[k])), flush=True)

    res = {'config': {'eval_rows': 'fineweb_n192_skip7000.pt', 'ridge': RIDGE,
                      'compilation': 'bottom-up (§1669)',
                      'coverage': f'HELD FIXED at the n96_skip80 mask ({ntok} tokens) for every arm -- '
                                  'same positions substituted, left live, and scored',
                      'varies': 'only the number of fit rows used to estimate the programs',
                      'fixes': '§1675, where the mask grew 6009->16110 tokens alongside the fit set',
                      'hypothesis_under_test': '§1673 -- the additive family is estimation-limited',
                      's1675_n96_skip80': S1675_N96_SKIP80,
                      's1675_growing_mask_size_effect': S1675_GROWING_MASK_SIZE_EFFECT},
           'tokens_covered': ntok, 'stake': round(st, 5), 'fit_sets': out,
           'size_effect_fixed_mask': {k: round(v, 5) for k, v in delta.items()},
           'control_vs_s1675': {k: round(v, 5) for k, v in ctrl.items()},
           'predictions': {'pred_a_additive_estimation_limited_ge_3pts': bool(pa),
                           'pred_b_s1675_linear_drop_was_the_mask': bool(pb),
                           'pred_c_control_reproduces_s1675': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
