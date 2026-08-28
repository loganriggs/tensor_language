# THE CORRECTED GRID -- one interleaved compile per cell, which is what §1757 should have done.
#
# §1757 shared one compile across three correction ranks with the prefix built at the top rank. Four
# of its six cells became uninterpretable: the two correction-rank-8 cells that were supposed to
# reproduce §1756 came out 0.33 and 0.76 nats away. The shortcut had HELPED the low ranks once
# (§1751) and I wrote that single observation into the design as an established asymmetry; with
# compressed tables it reversed (LESSON 37).
#
# So: nine cells, three table ranks by three correction ranks, EACH WITH ITS OWN interleaved compile
# at its own correction rank. Nine compiles instead of three, for numbers that mean what they say.
#
# The one clean cell §1757 did produce is the fidelity record to beat: table rank 64 with a rank-128
# correction, +0.78535 nats held out for 25.839M reals -- 16.6x smaller than the 430.00M of native
# modules. §1756's efficiency record is 0.155565 nats per million at table rank 8, correction rank 8.
#
# ROLES. Fitting uses the fit rows; both eval roles reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a FIDELITY IS MONOTONE IN CORRECTION RANK at every table rank. §1757 could not test this --
#          its low-rank cells were mismatched. If FALSE at some table rank, the correction saturates
#          or reverses there and the grid has an interior structure worth naming.
#   pred_b SOME CELL BEATS §1757's +0.78535. Scored independently of pred_a, since a cell can win
#          without the column being monotone.
#   pred_c SOME CELL BEATS §1756's 0.155565 nats per million. If both pred_b and pred_c pass on the
#          SAME cell, that cell dominates every program in §1748-§1757 and is the design point; if
#          they pass on different cells, the two optima stay separate and the choice is a real one.
#   pred_d CONTROLS: the three correction-rank-8 cells reproduce §1756's +0.54064, +0.46883 and
#          +0.41053 within 0.002. With one compile per cell they are literally the same programs, so
#          this control CAN fail and would mean the per-cell compile is not what §1756 ran -- which
#          is exactly the failure it caught in §1757. Plus table-only CE 7.35114, live CE 3.29205,
#          and coverage asserted at exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
CORR_RANKS = (8, 32, 128)
RIDGE = 1e-2
NBLK = 1                      # the §1748 class: table + x_t W
TABLE_RANKS = (64, 16, 8)
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/table_correction_grid_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1756 = {'64': {'skip7000': 0.53434, 'skip11000': 0.54064},
         '16': {'skip7000': 0.46973, 'skip11000': 0.46883},
         '8': {'skip7000': 0.40211, 'skip11000': 0.41053}}
S1756_BEST_EFF = 0.155565
S1757_BEST_FID = 0.78535
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
    return {r: ((U[:, :r] * S[:r]) @ Vh[:r]).float() for r in CORR_RANKS}, n['k']


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
    print(f'CORRECTION RANK AT A STARVED TABLE | table ranks {TABLE_RANKS} x correction ranks '
          f'{CORR_RANKS} | interleaved bottom-up | DISCOVERY ONLY', flush=True)

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
    for tr in TABLE_RANKS:
        tbl_r, tcost = compress_tables(tables, seen, tr)
        for cr in CORR_RANKS:
            # ONE COMPILE PER CELL. §1757 shared a compile across correction ranks with the prefix
            # built at the top rank, and four of six cells became uninterpretable -- the shortcut
            # helped the low ranks in §1751 and hurt them by 0.325 nats here (LESSON 37). Each cell
            # now compiles its own prefix at its own correction rank.
            installed = {}
            for st in order:
                allr, nk = fit_one(fit, st, tbl_r, seen, installed, NBLK)
                fits_ok = fits_ok and (nk == 24576)
                installed[st] = allr[cr]
            rec = {e: base[e]['table_only'] - ce(
                ev[e], [(st, prog_hook(tbl_r[st], seen, installed[st], NBLK)) for st in sites])
                for e in ev}
            fcost = 36 * 2 * cr * D
            total = (tcost + fcost) / 1e6
            key = f't{tr}_c{cr}'
            out[key] = {'table_rank': tr, 'corr_rank': cr,
                        'table_cost_M': round(tcost / 1e6, 4),
                        'factor_cost_M': round(fcost / 1e6, 4), 'total_cost_M': round(total, 4),
                        'recovered': {e: round(rec[e], 5) for e in rec},
                        'frac_of_stake': {e: round(rec[e] / base[e]['stake'], 5) for e in rec},
                        'nats_per_Mreal': round(rec['skip11000'] / total, 6)}
            o = out[key]
            print(f'  table {tr:3d} corr {cr:3d}: cost {total:8.4f}M | recovered ' + '  '.join(
                f'{e} {rec[e]:+.5f}' for e in rec)
                + f' | {o["nats_per_Mreal"]:.6f} nats/M   [{time.time() - t0:.0f}s]', flush=True)
        del tbl_r
        torch.cuda.empty_cache()

    ho = 'skip11000'
    pa = all(out[f't{tr}_c{CORR_RANKS[i]}']['recovered'][ho]
             < out[f't{tr}_c{CORR_RANKS[i + 1]}']['recovered'][ho]
             for tr in TABLE_RANKS for i in range(len(CORR_RANKS) - 1))
    pb = max(v['recovered'][ho] for v in out.values()) > S1757_BEST_FID
    pc = max(v['nats_per_Mreal'] for v in out.values()) > S1756_BEST_EFF
    pd = (all(abs(out[f't{k}_c8']['recovered'][e] - v) <= 0.002
              for k, kv in S1756.items() for e, v in kv.items())
          and abs(base['skip7000']['table_only'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == NCOV and fits_ok)
    beff = max(out, key=lambda k: out[k]['nats_per_Mreal'])
    bfid = max(out, key=lambda k: out[k]['recovered'][ho])

    print(f'\n  fidelity monotone in correction rank at every table rank -> {pa}', flush=True)
    print(f'  best fidelity {out[bfid]["recovered"][ho]:+.5f} ({bfid}) beats §1757\'s '
          f'{S1757_BEST_FID:+.5f} -> {pb}', flush=True)
    print(f'  best efficiency {out[beff]["nats_per_Mreal"]:.6f} ({beff}) beats §1756\'s '
          f'{S1756_BEST_EFF} -> {pc}', flush=True)
    print(f'  both corr-8 cells reproduce §1756 + table-only CE + live CE + coverage {ncov} -> '
          f'control {pd}', flush=True)

    r2 = {'config': {'table_ranks': list(TABLE_RANKS), 'corr_ranks': list(CORR_RANKS),
                     'costing': 'FULL after §1754: table r*(5419+1152)+2*1152 per site, correction '
                                '2*r*1152 per site',
                     'STILL_A_LOWER_BOUND': 'the hybrid hook runs the LIVE module on the 24% of '
                                            'scored positions whose token was uncovered at fit '
                                            'time; none of these programs stands alone',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
          'cells': out, 'best_efficiency': beff, 'best_fidelity': bfid,
          'predictions': {'pred_a_fidelity_monotone_in_correction_rank': bool(pa),
                          'pred_b_beats_S1757_fidelity': bool(pb),
                          'pred_c_beats_S1756_efficiency': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
