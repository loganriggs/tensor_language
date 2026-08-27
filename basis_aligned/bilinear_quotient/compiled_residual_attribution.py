# compiled_residual_attribution: WHERE DOES THE MISSING 43.7% OF THE COMPILED PROGRAM LIVE?
#
# §1670 built the best program this ledger has for bilin18's whole MLP stack: token tables
# at mlp0-3, least-squares linear maps at mlp4-17, compiled bottom-up. It reproduces 56.29%
# of a 4.32-nat stake. The obvious next question is which sites the other 43.71% is at,
# because that is where any further modelling effort should go.
#
# METHOD: for each site L, compile the SAME mixed program with site L EXEMPTED -- left live
# -- and measure the ceiling. The gain over the full program is how much of the residual
# that one site is responsible for.
#
# THE EXPENSIVE CHOICE, MADE DELIBERATELY. The cheap version installs the already-compiled
# program and just drops site L's hook. That is wrong here: every map above L was fitted
# with L substituted, so un-substituting L puts all of them off-distribution -- exactly the
# failure LESSONS 28 was written about, reintroduced as a measurement artifact. So each of
# the eighteen arms RECOMPILES the whole stack with L exempt. Eighteen compilations instead
# of one; it is the difference between measuring the site and measuring the compounding.
#
# CONTROL: the arm with no exemption must reproduce §1670's 56.29%. If the compilation is
# not reproducible run to run, no gain computed against it means anything.
#
# Registered predictions:
#   pred_a THE RESIDUAL IS CONCENTRATED: the single worst-modelled site accounts for >= 25%
#          of the program's total 43.71-point shortfall.
#   pred_b IT IS IN THE MIDDLE BAND: the top site by gain lies in mlp4-15, where §1668 found
#          the largest quadratic remainder (37.67%) and where a token table fails outright.
#   pred_c CONTROL -- compilation is reproducible: the no-exemption arm lands within 1 point
#          of §1670's 56.29%.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
ALL18 = list(range(0, 18))
FRONT = list(range(0, 4))
RIDGE = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'compiled_residual_attribution_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1670_MIXED = 0.5629
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


def install(prog):
    """prog: site -> ('linear', W) | ('table', tbl, seen)"""
    hs = []
    for L, p in prog.items():
        hs.append(H[L].mlp.register_forward_hook(
            linear_hook(p[1]) if p[0] == 'linear' else table_hook(p[1], p[2])))
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
def compile_program(rows, kinds, seen, exempt=None):
    prog = {}
    for L in ALL18:
        if L == exempt:
            continue
        prog[L] = fit_site(rows, L, kinds[L], prog, seen)
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
    KINDS = {L: ('table' if L in FRONT else 'linear') for L in ALL18}
    print(f'COMPILED RESIDUAL ATTRIBUTION | mixed program (table mlp0-3, linear mlp4-17), '
          f'RECOMPILED per exemption | ridge {RIDGE} | fit skip1200, eval skip7000', flush=True)

    cl = ce(ev, seen)
    cc = ce(ev, seen, hooks=[H[L].mlp.register_forward_hook(
        (lambda cst: (lambda mo, a, o: cst.to(o.dtype).expand_as(o)))(
            K[f'mlp{L}'].to(DEV).float())) for L in ALL18])
    st = cc - cl
    print(f'  CE live {cl:.5f} | all-MLP constant {cc:.5f} | stake {st:.4f} nats', flush=True)

    def ceiling_for(exempt):
        prog = compile_program(fit, KINDS, seen, exempt=exempt)
        ct = ce(ev, seen, hooks=install(prog))
        del prog
        torch.cuda.empty_cache()
        return (cc - ct) / st if st > 1e-6 else float('nan')

    base = ceiling_for(None)
    shortfall = 1.0 - base
    print(f'  full mixed program: CEILING {base:7.2%}  (§1670 {S1670_MIXED:.2%}) | '
          f'shortfall {shortfall:.2%}', flush=True)

    gains = {}
    for L in ALL18:
        c = ceiling_for(L)
        gains[f'mlp{L}'] = {'ceiling_exempt': round(c, 5), 'gain': round(c - base, 5),
                            'share_of_shortfall': round((c - base) / shortfall, 4)
                            if shortfall > 1e-9 else None,
                            'family': KINDS[L]}
        print(f'    exempt mlp{L:<2d} ({KINDS[L]:6s}) ceiling {c:7.2%}  gain {c - base:+7.2%}'
              f'  = {(c - base) / shortfall:6.2%} of the shortfall', flush=True)

    rank = sorted(gains.items(), key=lambda kv: -kv[1]['gain'])
    top, topv = rank[0]
    top_site = int(top[3:])
    gsum = sum(v['gain'] for v in gains.values())

    pa = topv['share_of_shortfall'] >= 0.25
    pb = 4 <= top_site <= 15
    pc = abs(base - S1670_MIXED) <= 0.01

    print(f'\n  WORST-MODELLED SITE: {top} ({topv["family"]}), gain {topv["gain"]:+.2%} '
          f'= {topv["share_of_shortfall"]:.2%} of the 43.7-point shortfall -> concentrated {pa}',
          flush=True)
    print(f'  top three: ' + ',  '.join(f'{k} {v["gain"]:+.2%}' for k, v in rank[:3]), flush=True)
    print(f'  in the middle band {pb} | sum of eighteen single-site gains {gsum:+.2%} '
          f'(not constrained to the shortfall -- each arm is a different compilation)', flush=True)
    print(f'  CONTROL no-exemption {base:.2%} vs §1670 {S1670_MIXED:.2%} -> reproducible {pc}',
          flush=True)

    res = {'config': {'sites': ALL18, 'front_sites': FRONT, 'ridge': RIDGE,
                      'program': 'token table at mlp0-3, linear map at mlp4-17',
                      'compilation': 'bottom-up, RECOMPILED for every exemption so the maps above '
                                     'the exempt site are never applied off-distribution (LESSONS 28)',
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'scoring': 'covered positions only',
                      's1670_mixed': S1670_MIXED},
           'stake': round(st, 5), 'base_ceiling': round(base, 5),
           'shortfall': round(shortfall, 5), 'sites': gains,
           'ranking': [k for k, _ in rank], 'sum_of_gains': round(gsum, 5),
           'predictions': {'pred_a_residual_concentrated_ge_25pct': bool(pa),
                           'pred_b_worst_site_in_middle_band': bool(pb),
                           'pred_c_control_compilation_reproducible': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
