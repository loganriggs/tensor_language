# HOW FAR DOWN DOES THE TABLE GO? -- extending §1755's sweep past the point where it was still
# improving, which is the defect LESSONS 31's addendum records twice.
#
# §1755: compressing each site's covered table block from full rank to 64 made the program BETTER on
# both axes -- +0.54064 against +0.38578 held out, at 15.886M reals against 225.442M. Rank 16 still
# recovered +0.46878 at 4.531M, so the registered "rank 16 breaks it" prediction failed and
# cost-efficiency was still rising monotonically at the bottom of the tested range. A sweep that ends
# while its answer is still moving cannot name a design point.
#
# So this one goes to the floor: ranks 64, 16, 8, 4, 1 and 0 -- where rank 0 is a per-site CONSTANT
# with no token dependence at all, turning the program into "optimal constant plus a rank-8 linear
# read of the site's input" and pricing at 0.083M reals for all 36 sites.
#
# Rank 0 is the interesting arm. Everything from §1662 onward has treated the per-token table as the
# base program and measured against it. If a constant plus a linear correction still recovers, the
# per-token identity -- the single most expensive part of the program by two orders of magnitude --
# is not what was carrying it.
#
# ROLES. Fitting uses the fit rows; both eval roles reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a EFFICIENCY HAS AN INTERIOR OPTIMUM: nats per million reals is NOT monotone increasing all
#          the way to rank 0. If FALSE -- if it keeps rising to a constant table -- then the whole
#          per-token apparatus is cost-dominated by something that never pays for itself and the
#          design point is rank 0.
#   pred_b FIDELITY PEAKS IN THE INTERIOR: the best held-out recovery is at neither end of the sweep.
#          Scored independently of pred_a, since the fidelity and efficiency optima need not coincide
#          and the gap between them is the actual design decision.
#   pred_c A CONSTANT TABLE STILL RECOVERS: rank 0 is above zero. If FALSE, per-token identity is
#          load-bearing after all and the compression story has a floor above the constant.
#   pred_d CONTROLS: ranks 64 and 16 reproduce §1755's +0.53433 / +0.54064 and +0.46968 / +0.46878
#          within 0.002 -- two arms of a four-arm sweep re-run in a sixth script -- plus table-only
#          CE 7.35114, live CE 3.29205, and coverage asserted at exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANK = 8
RIDGE = 1e-2
NBLK = 1                      # the §1748 class: table + x_t W
TABLE_RANKS = (64, 16, 8, 4, 1, 0)   # 0 = a per-site constant, no token dependence at all
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/table_rank_floor_sweep_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1755 = {'64': {'skip7000': 0.53433, 'skip11000': 0.54064},
         '16': {'skip7000': 0.46968, 'skip11000': 0.46878}}
NCOV = 5419
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
def compress_tables(tables, seen, r):
    """Rank-r truncate the COVERED block of each site's table.

    Storage per site: full covered block is 5419 x 1152 = 6.243M reals; rank r costs
    r * (5419 + 1152) = 6571r. Uncovered tokens keep the global mean, which is 1152 reals and is
    counted. r=None returns the tables unchanged, so that arm reproduces §1748 exactly.
    """
    if r is None:
        return tables, 36 * (NCOV * D + D)
    out = {}
    idx = seen.nonzero(as_tuple=True)[0]
    for st, tbl in tables.items():
        blk = tbl[idx].double()
        mu = blk.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(blk - mu, full_matrices=False)
        rec = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        t2 = tbl.clone()
        t2[idx] = rec
        out[st] = t2
    # r factors of length (5419 + 1152), plus the per-site mean row and the uncovered fallback row
    return out, 36 * (r * (NCOV + D) + 2 * D)


@torch.no_grad()
def main():
    t0 = time.time()
    assert_features_are_causal()
    fit = load(FIT_ROWS)
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'TABLE RANK COMPRESSION | the dominant cost term (§1754) | table ranks {TABLE_RANKS} | '
          f'linear correction rank {RANK}, interleaved bottom-up | DISCOVERY ONLY', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
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
    for r in TABLE_RANKS:
        tbl_r, tcost = compress_tables(tables, seen, r)
        tonly = {e: base[e]['table_only'] - ce(
            ev[e], [(st, table_hook(tbl_r[st], seen)) for st in sites]) for e in ev}
        installed = {}
        for st in order:
            W, nk = fit_one(fit, st, tbl_r, seen, installed, NBLK)
            fits_ok = fits_ok and (nk == 24576)
            installed[st] = W
        rec = {e: base[e]['table_only'] - ce(
            ev[e], [(st, prog_hook(tbl_r[st], seen, installed[st], NBLK)) for st in sites])
            for e in ev}
        fcost = 36 * 2 * RANK * D
        total = (tcost + fcost) / 1e6
        key = 'full' if r is None else str(r)
        out[key] = {'table_rank': r, 'table_cost_M': round(tcost / 1e6, 4),
                    'factor_cost_M': round(fcost / 1e6, 4), 'total_cost_M': round(total, 4),
                    'table_only_delta': {e: round(tonly[e], 5) for e in tonly},
                    'recovered': {e: round(rec[e], 5) for e in rec},
                    'frac_of_stake': {e: round(rec[e] / base[e]['stake'], 5) for e in rec},
                    'nats_per_Mreal': round(rec['skip11000'] / total, 6)}
        o = out[key]
        print(f'\n  table rank {str(r):5s}: cost {total:9.4f}M ({o["table_cost_M"]:.3f} tables + '
              f'{o["factor_cost_M"]:.3f} factors) | recovered ' + '  '.join(
                  f'{e} {rec[e]:+.5f}' for e in rec)
              + f' | {o["nats_per_Mreal"]:.6f} nats/M   [{time.time() - t0:.0f}s]', flush=True)
        print(f'    (table alone, before the correction: ' + '  '.join(
            f'{e} {tonly[e]:+.5f}' for e in tonly) + ')', flush=True)

    ho = 'skip11000'
    keys = [str(r) for r in TABLE_RANKS]
    eff = [out[k]['nats_per_Mreal'] for k in keys]
    fid = [out[k]['recovered'][ho] for k in keys]
    pa = not all(eff[i] < eff[i + 1] for i in range(len(eff) - 1))
    best_fid = max(range(len(fid)), key=lambda i: fid[i])
    pb = 0 < best_fid < len(fid) - 1
    pc = out['0']['recovered'][ho] > 0.0
    pd = (all(abs(out[k]['recovered'][e] - v) <= 0.002
              for k, kv in S1755.items() for e, v in kv.items())
          and abs(base['skip7000']['table_only'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == NCOV and fits_ok)

    print(f'\n  efficiency {[round(x, 4) for x in eff]} is NOT monotone to rank 0 -> interior '
          f'optimum {pa}', flush=True)
    print(f'  fidelity {[round(x, 4) for x in fid]} peaks in the interior at rank '
          f'{keys[best_fid]} -> {pb}', flush=True)
    print(f'  rank 0 -- a per-site CONSTANT, no token dependence -- still recovers '
          f'{out["0"]["recovered"][ho]:+.5f} > 0 -> {pc}', flush=True)
    print(f'  ranks 64 and 16 reproduce §1755 + table-only CE + live CE + coverage {ncov} -> '
          f'control {pd}', flush=True)

    r2 = {'config': {'correction_rank': RANK, 'table_ranks': [str(x) for x in TABLE_RANKS],
                     'costing': 'FULL program cost after §1754: tables plus factors. A full covered '
                                'table is 5419 x 1152 = 6.243M reals per site; a rank-r table is '
                                'r*(5419+1152) plus a mean row and an uncovered fallback row.',
                     'STILL_A_LOWER_BOUND': 'the hybrid hook (§1661) runs the LIVE module on the 24% '
                                            'of scored positions whose token was not covered at fit '
                                            'time, so none of these programs stands alone and none '
                                            'of these figures prices that fallback.',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
          'by_table_rank': out,
          'predictions': {'pred_a_efficiency_has_interior_optimum': bool(pa),
                          'pred_b_fidelity_peaks_in_the_interior': bool(pb),
                          'pred_c_constant_table_still_positive': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
