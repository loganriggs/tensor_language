# A NON-LOCAL PROGRAM CLASS -- because §1751 showed the limit is the class, and §1747 showed exactly
# where the class fails.
#
# The state of the compilation thread: ordering is settled (§1749, one bottom-up pass is a fixed
# point, proved by an exactly-zero result), the objective is worth 43% (§1750), and capacity is not
# the constraint (§1751: rank 32 and rank 128 tie at +0.639 and rank 128 collapses to -0.0914 under
# training). What is left belongs to the program class.
#
# And the class fails in one identifiable place. `table[token] + x_t W` reads only the CURRENT
# position. §1747, held out: the median MLP closes 91.23% of its own gap and the median attention
# site closes -1.45%. An MLP's output genuinely is a function of its current-position input, so a
# linear map has something to fit; an attention module's output is a function of the whole prefix,
# and §1682 measured its write as 83.6% non-local. A current-position map has nothing to read.
#
# So give it something to read. Three variants, all fitted by the same interleaved bottom-up
# procedure, all at rank 8:
#     A   table + x_t W                                    the §1748 class, 18,432 reals per site
#     B   table + x_t W1 + x_{t-1} W2                      lag 1, 36,864 per site
#     C   table + x_t W1 + x_{t-1} W2 + prefixmean(x) W3    lag 1 and prefix mean, 55,296 per site
# The prefix mean is causal (cumulative mean up to and including t), so nothing reads the future --
# the error §1733 recorded, checked here by construction and asserted in the feature builder.
#
# Rank truncation is applied to the stacked map, so a rank-r variant with k feature blocks costs
# r*(k*1152 + 1152) reals rather than k times the rank-8 baseline.
#
# ROLES. Fitting uses the fit rows; both eval roles reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other so no arm is decided
# by another's outcome:
#   pred_a NON-LOCAL FEATURES HELP: variant C recovers more than variant A on the held-out role. If
#          FALSE, the class limit is not about locality and §1747's MLP/attention split has some
#          other cause -- which would be a surprise worth a lot, since it would mean §1682's 83.6%
#          non-locality does not translate into a program constraint.
#   pred_b THE GAIN IS CONCENTRATED IN ATTENTION: the median attention site's fraction of its own gap
#          closed rises above 20% under C, from -1.45% under A. Scored independently of pred_a, since
#          C can beat A on the joint program through the MLPs alone.
#   pred_c LAG 1 CARRIES MOST OF IT: variant B reaches at least 60% of C's improvement over A. §1707
#          found the previous position worth 39.9 points against 13.8 for everything further back,
#          so if that structure is what a program needs, one lag should do most of the work. If
#          FALSE, the prefix mean is doing something a single lag cannot, and the class needs genuine
#          aggregation rather than one more position.
#   pred_d CONTROLS: variant A reproduces §1748's +0.40631 and +0.38578 within 0.002 -- the identical
#          program and procedure rebuilt by a fourth script -- plus table-only CE 7.35114 within
#          0.005, live CE 3.29205 within 1e-3, coverage 5419 of 50257, and every per-site fit firing
#          on the full 24576 positions.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANK = 8
RIDGE = 1e-2
VARIANTS = {'A_current': 1, 'B_lag1': 2, 'C_lag1_prefixmean': 3}
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/nonlocal_program_class_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1748_A = {'skip7000': 0.40631, 'skip11000': 0.38578}
COV = {}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def features(x, nblk):
    """[x_t] , [x_t, x_{t-1}] , or [x_t, x_{t-1}, causal prefix mean]. CAUSAL BY CONSTRUCTION."""
    f = [x]
    if nblk >= 2:
        f.append(F.pad(x[:, :-1], (0, 0, 1, 0)))
    if nblk >= 3:
        t = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype).view(1, -1, 1)
        f.append(x.cumsum(1) / t)
    return torch.cat(f, dim=-1)


def assert_features_are_causal():
    """A hand-built check that no feature reads the future -- the §1733 error, prevented not hoped."""
    a = torch.zeros(1, 4, 2); a[0, 3] = 99.0            # a big value ONLY at the last position
    b = torch.zeros(1, 4, 2)
    fa, fb = features(a, 3), features(b, 3)
    assert torch.equal(fa[:, :3], fb[:, :3]), 'a feature at t<3 changed when position 3 changed'
    c = torch.zeros(1, 4, 2); c[0, 1] = 1.0
    fc = features(c, 3)
    assert float(fc[0, 2, 2]) == 1.0, f'lag-1 block wrong: {float(fc[0, 2, 2])}'
    assert abs(float(fc[0, 3, 4]) - 0.25) < 1e-6, f'prefix mean wrong: {float(fc[0, 3, 4])}'


def table_hook(tbl, seen):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def prog_hook(tbl, seen, W, nblk):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        f = features(args[0].float(), nblk).reshape(-1, nblk * D)
        sub = sub + (f @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
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
    sweep(rows, hooks=[(st, mk(st, j == 0)) for j, st in enumerate(sites)])
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
def fit_one(rows, st, tables, seen, installed, nblk):
    P = nblk * D
    xtx = torch.zeros(P, P, device=DEV, dtype=torch.float64)
    xtr = torch.zeros(P, D, device=DEV, dtype=torch.float64)
    n = {'k': 0}

    def cap(mod, args, out):
        y = (out[0] if isinstance(out, tuple) else out).reshape(-1, D).double()
        f = features(args[0].float(), nblk).reshape(-1, P).double()
        nonlocal xtx, xtr
        xtx += f.T @ f
        xtr += f.T @ (y - tables[st][STATE['idx'].reshape(-1)].double())
        n['k'] += f.shape[0]
        return None
    hooks = [(st, cap)] + [(s2, prog_hook(tables[s2], seen, W2, nblk))
                           for s2, W2 in installed.items()]
    sweep(rows, hooks=hooks)
    assert n['k'] > 0, f'fit at {st} never fired'
    A = xtx + RIDGE * torch.eye(P, device=DEV, dtype=torch.float64) * (n['k'] / P)
    U, S, Vh = torch.linalg.svd(torch.linalg.solve(A, xtr), full_matrices=False)
    return ((U[:, :RANK] * S[:RANK]) @ Vh[:RANK]).float(), n['k']


@torch.no_grad()
def main():
    t0 = time.time()
    assert_features_are_causal()
    print('  feature causality known-answer check PASSED (LESSONS 34)', flush=True)
    fit = load(FIT_ROWS)
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'NON-LOCAL PROGRAM CLASS | rank {RANK} | variants {list(VARIANTS)} | interleaved '
          f'bottom-up | DISCOVERY ONLY', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    ev, base = {}, {}
    for ename, epath, ce_ref in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        cl = ce(e)
        tb = ce(e, [(st, table_hook(tables[st], seen)) for st in sites])
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        base[ename] = {'live': cl, 'table_only': tb, 'stake': tb - cl}

    out, fits_ok = {}, True
    for vname, nblk in VARIANTS.items():
        installed = {}
        for st in order:
            W, nk = fit_one(fit, st, tables, seen, installed, nblk)
            fits_ok = fits_ok and (nk == 24576)
            installed[st] = W
        rec = {e: base[e]['table_only'] - ce(
            ev[e], [(st, prog_hook(tables[st], seen, installed[st], nblk)) for st in sites])
            for e in ev}
        per_site = {}
        for st in sites:
            nm = f'{st[0]}{st[1]}'
            nat = ce(ev['skip11000'], [(s2, table_hook(tables[s2], seen))
                                       for s2 in sites if s2 != st])
            solo = ce(ev['skip11000'], [(s2, prog_hook(tables[s2], seen, installed[s2], nblk))
                                        if s2 == st else (s2, table_hook(tables[s2], seen))
                                        for s2 in sites])
            g = base['skip11000']['table_only'] - nat
            per_site[nm] = {'gap': round(g, 5), 'kind': st[0],
                            'frac': round((base['skip11000']['table_only'] - solo) / g, 5)
                            if abs(g) > 1e-6 else None}
        aa = sorted(v['frac'] for v in per_site.values()
                    if v['kind'] == 'attn' and v['frac'] is not None)
        mm = sorted(v['frac'] for v in per_site.values()
                    if v['kind'] == 'mlp' and v['frac'] is not None)
        med = {'attn': 0.5 * (aa[len(aa) // 2 - 1] + aa[len(aa) // 2]),
               'mlp': 0.5 * (mm[len(mm) // 2 - 1] + mm[len(mm) // 2])}
        cost = round(36 * RANK * (nblk * D + D) / 1e6, 4)
        out[vname] = {'n_feature_blocks': nblk, 'cost_M': cost,
                      'recovered': {e: round(rec[e], 5) for e in rec},
                      'frac_of_stake': {e: round(rec[e] / base[e]['stake'], 5) for e in rec},
                      'median_frac': {k: round(v, 5) for k, v in med.items()},
                      'per_site': per_site}
        print(f'\n  {vname:20s} {nblk} block(s), {cost:.3f}M reals: ' + '  '.join(
            f'{e} {rec[e]:+.5f} ({rec[e] / base[e]["stake"]:+.2%})' for e in rec)
            + f'  | median frac closed  MLP {med["mlp"]:7.2%}  attention {med["attn"]:7.2%}'
              f'   [{time.time() - t0:.0f}s]', flush=True)

    ho = 'skip11000'
    A, B, C = 'A_current', 'B_lag1', 'C_lag1_prefixmean'
    pa = out[C]['recovered'][ho] > out[A]['recovered'][ho]
    pb = out[C]['median_frac']['attn'] > 0.20
    dC = out[C]['recovered'][ho] - out[A]['recovered'][ho]
    dB = out[B]['recovered'][ho] - out[A]['recovered'][ho]
    pc = (dB >= 0.60 * dC) if dC > 1e-9 else False
    pd = (all(abs(out[A]['recovered'][e] - v) <= 0.002 for e, v in S1748_A.items())
          and abs(base['skip7000']['table_only'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == 5419 and fits_ok)

    print(f'\n  C beats A held out ({out[C]["recovered"][ho]:+.5f} vs '
          f'{out[A]["recovered"][ho]:+.5f}) -> {pa}', flush=True)
    print(f'  median attention site closes >20% under C ({out[C]["median_frac"]["attn"]:.2%}, '
          f'A was {out[A]["median_frac"]["attn"]:.2%}) -> {pb}', flush=True)
    print(f'  lag-1 alone carries >=60% of the gain ({dB:+.5f} of {dC:+.5f}) -> {pc}', flush=True)
    print(f'  variant A reproduces §1748 + table-only + live CE + coverage {ncov} -> control {pd}',
          flush=True)

    r = {'config': {'rank': RANK, 'ridge': RIDGE, 'variants': VARIANTS,
                    'features': 'A: x_t | B: x_t, x_{t-1} | C: x_t, x_{t-1}, causal prefix mean. '
                                'Causality is checked by a hand-built known-answer test before any '
                                'model runs (§1733, LESSONS 34).',
                    'cost': 'rank r with k feature blocks costs r*(k*1152 + 1152) reals per site',
                    'procedure': 'interleaved bottom-up (§1669, §1748); §1749 proved one pass is a '
                                 'fixed point so no iteration is attempted',
                    'ROLE_NOTE': 'DISCOVERY ONLY.'},
         'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
         'variants': out,
         'predictions': {'pred_a_nonlocal_helps': bool(pa),
                         'pred_b_gain_in_attention': bool(pb),
                         'pred_c_lag1_carries_most': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
