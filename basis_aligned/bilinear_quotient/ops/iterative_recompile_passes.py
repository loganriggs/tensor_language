# ITERATIVE RECOMPILE -- interleaving fixed the sign; can iteration close the rest of the gap?
#
# §1747: all 36 `table + x W_r` maps fitted simultaneously against an all-tabled context and installed
# together came out at -0.5462 nats held out -- worse than plain tables.
# §1748: fitting them BOTTOM-UP AND INTERLEAVED (§1669) flipped that to +0.3858, a 0.932-nat swing
# from fit order alone, but recovered only 22% of +1.7460, the sum of the 36 sites measured solo.
#
# One structural reason for the shortfall is visible in the procedure itself. During a bottom-up pass,
# each site is fitted against a compiled PREFIX while everything ABOVE it is still live -- and every
# one of those sites above is subsequently compiled, changing the context the earlier map was fitted
# for. A second pass, refitting each site against the FULL current program, removes exactly that: it
# is coordinate descent over the 36 maps, with pass 1 identical to §1748.
#
# Rank 8, the efficient point (§1746, §1748: 0.581 nats per million reals against greedy's 0.0223).
#
# ROLES. Fitting uses the fit rows; both eval roles reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other so no arm is decided by
# another's outcome:
#   pred_a PASS 2 IMPROVES ON PASS 1 held out. If FALSE, the half-built-context problem is not what
#          limits the bottom-up pass, and something else -- rank, the linear class, the table -- is
#          the binding constraint. That would be the more useful answer, since it stops a whole line
#          of "just iterate more" work.
#   pred_b DIMINISHING RETURNS: the pass-3 gain is smaller than the pass-2 gain. If FALSE the process
#          is still accelerating at three passes and the run is under-budgeted -- which is the defect
#          LESSONS 31's addendum records from §1743, so it is scored rather than hand-waved.
#   pred_c IT STAYS BELOW THE SUM OF THE PARTS after three passes. If FALSE, coordinate descent fully
#          closes the composition gap and the 36 sites are effectively independent once each is
#          fitted in context -- a much stronger result than expected, and one that would make the
#          whole-model program a solved problem at 0.664M reals.
#   pred_d CONTROLS: pass 1 reproduces §1748's +0.4063 and +0.3858 within 0.005 -- the identical
#          procedure re-run by a different script -- plus table-only CE 7.35114, live CE 3.29205,
#          coverage 5419 of 50257, and every per-site fit firing on the full 24576 positions.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANK = 8
NPASS = 3
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/iterative_recompile_passes_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1747_SIMULTANEOUS = {'skip7000': -0.5235, 'skip11000': -0.5462}   # rank 8, all-36 joint
S1748_PASS1 = {'skip7000': 0.4063, 'skip11000': 0.3858}            # rank 8, one bottom-up pass
S1747_SUM_OF_PARTS = {'skip7000': 1.8057, 'skip11000': 1.7460}
COV = {}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl, seen):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def lowrank_hook(tbl, seen, W):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = sub + (args[0].reshape(-1, D).to(W.dtype) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


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
def ce(rows, hooks=()):
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        cov = COV['seen'][idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, hooks=hooks, score=score)
    return acc['t'] / acc['n']


@torch.no_grad()
def fit_tables(rows, sites):
    s = {st: torch.zeros(50257, D, device=DEV) for st in sites}
    c = torch.zeros(50257, device=DEV)
    fired = {'n': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
            t = STATE['idx'].reshape(-1)
            s[st].index_add_(0, t, y)
            if first:
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
                fired['n'] += 1
            return None
        return hook
    sweep(rows, hooks=[mod_of(*st).register_forward_hook(mk(st, j == 0))
                       for j, st in enumerate(sites)])
    assert fired['n'] > 0, 'table fit never fired'
    seen = c > 0
    out = {}
    for st in sites:
        mean = s[st].sum(0) / c.sum()
        tbl = mean.unsqueeze(0).repeat(50257, 1)
        tbl[seen] = s[st][seen] / c[seen].unsqueeze(1)
        out[st] = tbl
    return out, seen


@torch.no_grad()
def fit_one(rows, st, tables, seen, installed, rank):
    """Fit W at ONE site with the already-compiled prefix substituted and everything else LIVE."""
    xtx = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    xtr = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'k': 0}

    def hook(mod, args, out):
        y = (out[0] if isinstance(out, tuple) else out).reshape(-1, D).double()
        x = args[0].reshape(-1, D).double()
        nonlocal xtx, xtr
        xtx += x.T @ x
        xtr += x.T @ (y - tables[st][STATE['idx'].reshape(-1)].double())
        n['k'] += x.shape[0]
        return None
    hooks = [mod_of(*st).register_forward_hook(hook)]
    for s2, W2 in installed.items():
        hooks.append(mod_of(*s2).register_forward_hook(lowrank_hook(tables[s2], seen, W2)))
    sweep(rows, hooks=hooks)
    assert n['k'] > 0, f'fit at {st} never fired'
    A = xtx + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (n['k'] / D)
    U, S, Vh = torch.linalg.svd(torch.linalg.solve(A, xtr))
    return ((U[:, :rank] * S[:rank]) @ Vh[:rank]).float(), n['k']


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'ITERATIVE RECOMPILE | rank {RANK} | {NPASS} passes of coordinate descent over the 36 '
          f'sites | DISCOVERY ONLY', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    ev = {}
    base = {}
    for ename, epath, ce_ref in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        cl = ce(e)
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        tb = ce(e, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen)) for st in sites])
        base[ename] = {'live': cl, 'table_only': tb, 'stake': tb - cl}
        print(f'  {ename}: live {cl:.5f} | table-only {tb:.5f} | stake {tb - cl:.4f}', flush=True)

    def evaluate(installed):
        return {e: round(base[e]['table_only'] - ce(
            ev[e], [mod_of(*st).register_forward_hook(lowrank_hook(tables[st], seen, installed[st]))
                    for st in sites]), 5) for e in ev}

    installed, fits_ok, passes = {}, True, []
    for p in range(NPASS):
        for st in order:
            # pass 1 is exactly §1748: the compiled PREFIX is installed and everything above is live.
            # passes 2+ hold every OTHER site at its current map -- coordinate descent, where each
            # site is refitted against the full program rather than a half-built one.
            ctx = {s2: w for s2, w in installed.items() if s2 != st}
            W, nk = fit_one(fit, st, tables, seen, ctx, RANK)
            fits_ok = fits_ok and (nk == 24576)
            installed[st] = W
        rec = evaluate(installed)
        passes.append({'pass': p + 1, 'recovered': rec,
                       'frac_of_stake': {e: round(rec[e] / base[e]['stake'], 5) for e in rec}})
        print(f'\n  pass {p + 1}: ' + '  '.join(
            f'{e} {rec[e]:+.4f} ({rec[e] / base[e]["stake"]:+.2%})' for e in rec)
            + f'   [{time.time() - t0:.0f}s]', flush=True)

    ho = 'skip11000'
    g2 = passes[1]['recovered'][ho] - passes[0]['recovered'][ho]
    g3 = passes[2]['recovered'][ho] - passes[1]['recovered'][ho]
    pa = g2 > 0
    pb = g3 < g2
    pc = passes[-1]['recovered'][ho] < S1747_SUM_OF_PARTS[ho]
    pd = (abs(base['skip7000']['table_only'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == 5419 and fits_ok
          and all(abs(passes[0]['recovered'][e] - v) <= 0.005 for e, v in S1748_PASS1.items()))

    print(f'\n  pass 2 improves on pass 1 held out ({g2:+.4f}) -> {pa}', flush=True)
    print(f'  pass 3 gain {g3:+.4f} smaller than pass 2 gain {g2:+.4f} -> diminishing {pb}',
          flush=True)
    print(f'  final {passes[-1]["recovered"][ho]:+.4f} still below the sum of the parts '
          f'{S1747_SUM_OF_PARTS[ho]:+.4f} -> {pc}', flush=True)
    print(f'  pass 1 reproduces §1748 + table-only + live CE + coverage {ncov} + all fits '
          f'full-size -> control {pd}', flush=True)

    r = {'config': {'rank': RANK, 'passes': NPASS, 'ridge': RIDGE,
                    'procedure': 'pass 1 is §1748 exactly -- bottom-up, compiled prefix installed, '
                                 'everything above live. Passes 2+ are coordinate descent: each '
                                 'site is refitted with every OTHER site held at its current map, '
                                 'so it sees the full program rather than a half-built one.',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
         'passes': passes,
         'reference': {'simultaneous_S1747': S1747_SIMULTANEOUS, 'pass1_S1748': S1748_PASS1,
                       'sum_of_parts_S1747': S1747_SUM_OF_PARTS},
         'predictions': {'pred_a_pass2_improves': bool(pa),
                         'pred_b_diminishing_returns': bool(pb),
                         'pred_c_below_sum_of_parts': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
