# additive_coverage_controlled: SEPARATING FAMILY FROM COVERAGE POLICY
#
# §1673 measured the additive program y = b(token) + xW at 47.34%, below both pure families
# (tables-at-mlp0-2 57.29%, all-linear 54.28%). A strictly richer family scoring worse than
# both of its own special cases is a signal that the measurement is wrong, not that the
# family is bad -- b=0 recovers all-linear and W=0 recovers all-table.
#
# THE CONFOUND IS ONE I BUILT DELIBERATELY AND MISJUDGED. The pure-table arm uses the §1661
# hybrid hook: the table is applied only where the token was seen at fit time, and the
# module runs LIVE at the other 23.4% of positions. The additive arm substituted at EVERY
# position, with b falling back to the position-weighted mean. I chose that so the family
# comparison would not be "confounded with a coverage policy" -- and thereby confounded it
# with a coverage policy, in the direction LESSONS 27 measured at 15.9 points on mlp0 alone.
# At an uncovered position the additive program uses a b that is systematically wrong while
# W was fitted on residuals from the CORRECT b, so the two terms do not compensate.
#
# So coverage policy and family are varied SEPARATELY here:
#           substitute everywhere        hybrid (live at uncovered)
#   linear      §1673's 54.28%                  measured here
#   additive    §1673's 47.34%                  measured here
#   table       (not defined)                   §1672's 57.29%, control
# A pure table has no everywhere-variant worth running -- that is the arm LESSONS 27 already
# showed is broken -- so that cell is left empty rather than filled with a known artifact.
#
# Registered predictions:
#   pred_a THE COVERAGE POLICY EXPLAINS THE FAILURE: additive under the hybrid hook exceeds
#          additive-everywhere by >= 8 percentage points.
#   pred_b WITH COVERAGE MATCHED, THE RICHER FAMILY WINS: additive-hybrid exceeds §1672's
#          best pure assignment of 57.29%. If it does not, the additive family genuinely
#          does not help here and §1673's verdict stands for a different reason than I gave.
#   pred_c CONTROLS -- both reference points reproduce: pure tables-at-mlp0-2 within 1 point
#          of 57.29%, and all-linear-everywhere within 1 point of 54.28%.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'additive_coverage_controlled_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1672_BEST_PURE = 0.5729
S1673_ADDITIVE_EVERYWHERE = 0.4734
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
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    seen = seen_mask(fit)
    SEENREF['m'] = seen
    print(f'ADDITIVE, COVERAGE CONTROLLED | family x coverage policy varied separately | '
          f'ridge {RIDGE} | fit skip1200, eval skip7000', flush=True)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'  CE live {cl:.5f} | all-MLP constant {cc:.5f} | stake {st:.4f} nats', flush=True)

    arms = {
        'table_mlp0_2_hybrid': {L: ('table' if L < 3 else 'linear') for L in ALL18},
        'linear_everywhere': {L: 'linear' for L in ALL18},
        'linear_hybrid': {L: 'linear_hybrid' for L in ALL18},
        'additive_everywhere': {L: 'additive' for L in ALL18},
        'additive_hybrid': {L: 'additive_hybrid' for L in ALL18},
    }
    out = {}
    for name, kinds in arms.items():
        prog = compile_program(fit, kinds, seen)
        ct = ce(ev, seen, hooks=install(prog))
        ceil = (cc - ct) / st if st > 1e-6 else float('nan')
        out[name] = {'ceiling': round(ceil, 5), 'ce': round(ct, 5)}
        print(f'  {name:22s} CE {ct:.5f} | CEILING {ceil:7.2%}', flush=True)
        del prog
        torch.cuda.empty_cache()

    ah = out['additive_hybrid']['ceiling']
    ae = out['additive_everywhere']['ceiling']
    lh = out['linear_hybrid']['ceiling']
    le = out['linear_everywhere']['ceiling']
    tb = out['table_mlp0_2_hybrid']['ceiling']

    pa = (ah - ae) >= 0.08
    pb = ah > S1672_BEST_PURE
    pc = (abs(tb - S1672_BEST_PURE) <= 0.01) and (abs(le - S1669_ALL_LINEAR) <= 0.01)

    print(f'\n  COVERAGE POLICY, holding family fixed:', flush=True)
    print(f'    linear   everywhere {le:7.2%} -> hybrid {lh:7.2%}  ({lh - le:+.2%})', flush=True)
    print(f'    additive everywhere {ae:7.2%} -> hybrid {ah:7.2%}  ({ah - ae:+.2%})', flush=True)
    print(f'  FAMILY, holding coverage fixed (hybrid):', flush=True)
    print(f'    linear {lh:7.2%} | table@mlp0-2 {tb:7.2%} | additive {ah:7.2%}', flush=True)
    print(f'  coverage explains the §1673 additive failure {pa} | richer family wins {pb} | '
          f'controls hold {pc}', flush=True)

    res = {'config': {'sites': ALL18, 'ridge': RIDGE,
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'design': 'family (linear | additive | table) x coverage policy '
                                '(substitute everywhere | hybrid, module live at uncovered)',
                      'compilation': 'bottom-up (§1669)', 'scoring': 'covered positions only',
                      'why': 'a strictly richer family scoring below both of its own special cases '
                             'signals a broken measurement, not a bad family',
                      's1672_best_pure': S1672_BEST_PURE,
                      's1673_additive_everywhere': S1673_ADDITIVE_EVERYWHERE,
                      's1669_all_linear': S1669_ALL_LINEAR},
           'stake': round(st, 5), 'arms': out,
           'coverage_effect': {'linear': round(lh - le, 5), 'additive': round(ah - ae, 5)},
           'predictions': {'pred_a_coverage_explains_failure_ge_8pts': bool(pa),
                           'pred_b_additive_beats_best_pure': bool(pb),
                           'pred_c_controls_hold': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
