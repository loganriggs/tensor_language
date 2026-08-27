# additive_program_family: b(token) + xW AT EVERY SITE — the family both lanes converged on
#
# §1672 found the best assignment of PURE families: token tables at mlp0-2, linear maps at
# mlp3-17, compiled bottom-up, 57.29% of bilin18's 4.3196-nat MLP stake. It also found that
# tables and linear maps are doing different work -- tables win outright at mlp0-1, linear
# wins from mlp3 on -- which is exactly the situation where neither pure family is right and
# their SUM should beat both.
#
# That sum is also the form Codex arrived at independently for their native-Down program,
# `y_hat = b(token) + c + A.B.h(z)`. Two lanes converging on one functional form from
# different directions is worth testing directly rather than noting.
#
# METHOD, per site, compiled bottom-up (§1669) with the stack below already substituted:
#   1. fit the per-token table b(token) on the module output
#   2. fit the linear map W by least squares on the RESIDUAL y - b(token)
#   3. substitute y_hat = b(token) + xW, with b falling back to the position-weighted mean
#      at tokens unseen at fit time -- NOT to zero, and NOT leaving the module live, because
#      unlike a pure table this program is defined everywhere and the §1661 hybrid hook
#      would confound the family comparison with a coverage policy
# The residual fit is the honest order: fitting W first and the table on ITS residual would
# hand the linear map the mean structure the table exists to capture.
#
# Registered predictions:
#   pred_a THE SUM BEATS THE BEST PURE ASSIGNMENT: additive at all eighteen sites exceeds
#          §1672's 57.29% by >= 3 percentage points.
#   pred_b THE TABLE EARNS ITS PLACE EVERYWHERE, not just at the front: additive exceeds
#          all-linear by more than the best pure mixed program does (>= 3.01 points), i.e.
#          a token term helps at sites where a pure table loses.
#   pred_c CONTROL -- the pure mlp0-2 arm reproduces §1672's 57.29% within 1 point. Every
#          comparison is against it.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'additive_program_family_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1672_BEST_PURE = 0.5729
S1669_ALL_LINEAR = 0.5428
S1668_NAIVE_TABLE = 0.3427
S1668_BANDS = {'front_token': 0.7645, 'front_linear': 0.6868, 'middle_linear': 0.6233,
               'late_linear': 0.8360}
STATE = {}


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
            else additive_hook(p[1], p[2]) if p[0] == 'additive'
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
        prog[L] = (fit_additive(rows, L, prog, seen) if kinds[L] == 'additive'
                   else fit_site(rows, L, kinds[L], prog, seen))
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
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    seen = seen_mask(fit)
    print(f'ADDITIVE PROGRAM FAMILY | y_hat = b(token) + xW at every site, compiled '
          f'bottom-up | ridge {RIDGE} | fit skip1200, eval skip7000', flush=True)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'  CE live {cl:.5f} | all-MLP constant {cc:.5f} | stake {st:.4f} nats', flush=True)

    arms = {
        'pure_table_mlp0_2': {L: ('table' if L < 3 else 'linear') for L in ALL18},
        'all_linear': {L: 'linear' for L in ALL18},
        'all_additive': {L: 'additive' for L in ALL18},
    }
    out = {}
    for name, kinds in arms.items():
        prog = compile_program(fit, kinds, seen)
        ct = ce(ev, seen, hooks=install(prog))
        ceil = (cc - ct) / st if st > 1e-6 else float('nan')
        out[name] = {'ceiling': round(ceil, 5), 'ce': round(ct, 5)}
        print(f'  {name:19s} CE {ct:.5f} | CEILING {ceil:7.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    ctrl = out['pure_table_mlp0_2']['ceiling']
    lin = out['all_linear']['ceiling']
    add = out['all_additive']['ceiling']

    pa = (add - ctrl) >= 0.03
    pb = (add - lin) >= (ctrl - lin)
    pc = abs(ctrl - S1672_BEST_PURE) <= 0.01

    print(f'\n  ADDITIVE {add:.2%}  vs best pure assignment {ctrl:.2%}  '
          f'({add - ctrl:+.2%}) -> sum beats both families {pa}', flush=True)
    print(f'  additive over all-linear {add - lin:+.2%}  vs  pure-mixed over all-linear '
          f'{ctrl - lin:+.2%} -> token term helps everywhere {pb}', flush=True)
    print(f'  CONTROL pure mlp0-2 {ctrl:.2%} vs §1672 {S1672_BEST_PURE:.2%} -> {pc}',
          flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'program': 'y_hat = b(token) + xW, table fitted first, W by least squares '
                                 'on the residual y - b(token)',
                      'compilation': 'bottom-up (§1669)',
                      'unseen_tokens': 'b falls back to the position-weighted mean; the additive '
                                       'program is defined at every position so no hybrid hook is used',
                      'scoring': 'covered positions only',
                      'converges_with': "Codex's native-Down form y_hat = b(token) + c + A.B.h(z)",
                      's1672_best_pure': S1672_BEST_PURE, 's1669_all_linear': S1669_ALL_LINEAR},
           'stake': round(st, 5), 'arms': out,
           'predictions': {'pred_a_additive_beats_best_pure_ge_3pts': bool(pa),
                           'pred_b_token_term_helps_everywhere': bool(pb),
                           'pred_c_control_reproduces_s1672': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
