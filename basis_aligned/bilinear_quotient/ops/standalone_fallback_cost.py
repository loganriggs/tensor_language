# WHAT THE FALLBACK IS WORTH -- pricing the caveat I have put at the top of every result.
#
# Every program in §1748-§1758 uses the hybrid hook (§1661): the per-token table applies where the
# token was seen at fit time, and the native module runs LIVE everywhere else. I have written "24% of
# scored positions fall through to the live module, so none of these programs stands alone" into six
# sections and a FINDINGS entry, and never measured what it costs.
#
# It is a pure FIDELITY question, not a fidelity/cost trade, which is why it is worth a short run: a
# standalone program costs exactly the same reals. The site's global mean row is already stored and
# already counted; the standalone arm just stops consulting the coverage mask and lets uncovered
# tokens take that mean. Nothing is added, one thing is removed -- the original 430.00M of modules.
#
# Two cells, both from §1758: the fidelity design point (table 64, correction 128, +0.78536 at
# 25.839M) and the efficiency design point (table 8, correction 8, +0.41052 at 2.639M). Each compiled
# twice, hybrid and standalone, with the compile itself run under the arm it will be deployed in --
# §1746 measured that mismatch at 6.7x and §1757 at 0.325 nats, so it is not a detail.
#
# The fallback fraction is MEASURED here per eval role rather than quoted from memory.
#
# ROLES. Fitting uses the fit rows; both eval roles reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a THE FALLBACK IS AFFORDABLE AT THE FIDELITY POINT: the standalone program loses less than
#          30% of the hybrid's held-out recovery. If FALSE, every figure in §1748-§1758 is inflated by
#          more than a third by a term I have been declaring and not pricing, and the honest frontier
#          is materially worse than published.
#   pred_b THE LOSS IS LARGER WHERE THE TABLE IS STARVED: the fraction lost at table rank 8 exceeds
#          that at table rank 64. A rank-8 table's mean row is a worse stand-in, so the fallback
#          should hurt more there. If FALSE, the loss is about the uncovered TOKENS rather than about
#          the table's quality, which is a different and more awkward problem.
#   pred_c BOTH STANDALONE PROGRAMS STAY POSITIVE -- they still beat the plain hybrid table baseline.
#          Scored independently of pred_a, since a program can lose most of its margin and remain
#          above zero.
#   pred_d CONTROLS: both HYBRID arms reproduce §1758's +0.77602 / +0.78536 and +0.40210 / +0.41052
#          within 0.002 -- an eighth script -- plus table-only CE 7.35114, live CE 3.29205, and
#          coverage asserted at exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RIDGE = 1e-2
NBLK = 1                      # the §1748 class: table + x_t W
CELLS = ((64, 128), (8, 8))   # §1758's fidelity point and efficiency point
ARMS = ('hybrid', 'standalone')
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/standalone_fallback_cost_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1758_HYBRID = {'t64_c128': {'skip7000': 0.77602, 'skip11000': 0.78536},
                't8_c8':    {'skip7000': 0.40210, 'skip11000': 0.41052}}
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


def table_hook(tbl, seen, standalone=False):
    """HYBRID (§1661): the table applies only where the token was seen at fit, and the module runs
    LIVE elsewhere -- 24% of scored positions. STANDALONE: the table applies everywhere, so an
    uncovered token gets the site's global mean row and the native module is never called.

    The standalone program costs exactly the same reals -- the mean row is already stored and already
    counted -- so this is a pure fidelity question, not a fidelity/cost trade."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        if not standalone:
            sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def prog_hook(tbl, seen, W, nblk, standalone=False):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        f = features(args[0].float(), nblk).reshape(-1, nblk * D)
        sub = sub + (f @ W).reshape(y.shape).to(y.dtype)
        if not standalone:
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
def fit_one(rows, st, tables, seen, installed, nblk, standalone=False):
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
    hooks = [(st, cap)] + [(s2, prog_hook(tables[s2], seen, W2, nblk, standalone))
                           for s2, W2 in installed.items()]
    sweep(rows, hooks=hooks)
    assert n['k'] > 0, f'fit at {st} never fired'
    A = xtx + RIDGE * torch.eye(P, device=DEV, dtype=torch.float64) * (n['k'] / P)
    U, S, Vh = torch.linalg.svd(torch.linalg.solve(A, xtr), full_matrices=False)
    return {r: ((U[:, :r] * S[:r]) @ Vh[:r]).float()
            for r in sorted({c for _, c in CELLS})}, n['k']


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
    print(f'STANDALONE FALLBACK COST | cells {CELLS} x arms {ARMS} | the 24% of positions the hybrid '
          f'hook sends to the LIVE module | DISCOVERY ONLY', flush=True)

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
        base[ename] = {'live': cl, 'table_only_hybrid': tb, 'stake': tb - cl}
        # how many scored positions the hybrid sends to the live module, measured not assumed
        nall = ncov_pos = 0
        for i2 in range(0, e.shape[0], 8):
            idx = e[i2:i2 + 8, :-1].to(DEV)
            c2 = seen[idx[:, 64:]]
            nall += c2.numel(); ncov_pos += int(c2.sum())
        base[ename]['scored_positions'] = nall
        base[ename]['covered_positions'] = ncov_pos
        base[ename]['fallback_fraction'] = round(1 - ncov_pos / nall, 5)
        print(f'  {ename}: live {cl:.5f} | hybrid table-only {tb:.5f} | '
              f'{ncov_pos}/{nall} positions covered, {base[ename]["fallback_fraction"]:.2%} fall '
              f'through to the live module', flush=True)

    out, fits_ok = {}, True
    for tr, cr in CELLS:
        tbl_r, tcost = compress_tables(tables, seen, tr)
        for arm in ARMS:
            sa = (arm == 'standalone')
            installed = {}
            for st in order:
                allr, nk = fit_one(fit, st, tbl_r, seen, installed, NBLK, sa)
                fits_ok = fits_ok and (nk == 24576)
                installed[st] = allr[cr]
            rec = {e: base[e]['table_only_hybrid'] - ce(
                ev[e], [(st, prog_hook(tbl_r[st], seen, installed[st], NBLK, sa)) for st in sites])
                for e in ev}
            total = (tcost + 36 * 2 * cr * D) / 1e6
            key = f't{tr}_c{cr}_{arm}'
            out[key] = {'table_rank': tr, 'corr_rank': cr, 'arm': arm,
                        'total_cost_M': round(total, 4),
                        'recovered': {e: round(rec[e], 5) for e in rec},
                        'frac_of_stake': {e: round(rec[e] / base[e]['stake'], 5) for e in rec},
                        'nats_per_Mreal': round(rec['skip11000'] / total, 6)}
            print(f'  table {tr:3d} corr {cr:3d} {arm:11s}: cost {total:8.4f}M | ' + '  '.join(
                f'{e} {rec[e]:+.5f}' for e in rec) + f'   [{time.time() - t0:.0f}s]', flush=True)
        del tbl_r
        torch.cuda.empty_cache()

    ho = 'skip11000'
    def loss_frac(tr, cr):
        h = out[f't{tr}_c{cr}_hybrid']['recovered'][ho]
        s = out[f't{tr}_c{cr}_standalone']['recovered'][ho]
        return (h - s) / h if h > 1e-9 else None
    lf_fid, lf_eff = loss_frac(*CELLS[0]), loss_frac(*CELLS[1])
    pa = lf_fid is not None and lf_fid < 0.30
    pb = (lf_eff is not None and lf_fid is not None and lf_eff > lf_fid)
    pc = all(out[f't{tr}_c{cr}_standalone']['recovered'][ho] > 0.0 for tr, cr in CELLS)
    pd = (all(abs(out[f'{k}_hybrid']['recovered'][e] - v) <= 0.002
              for k, kv in S1758_HYBRID.items() for e, v in kv.items())
          and abs(base['skip7000']['table_only_hybrid'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == NCOV and fits_ok)

    print(f'\n  standalone loses <30% at the fidelity point '
          f'({lf_fid if lf_fid is None else f"{lf_fid:.2%}"}) -> {pa}', flush=True)
    print(f'  the loss is larger at the starved-table efficiency point '
          f'({lf_eff if lf_eff is None else f"{lf_eff:.2%}"}) -> {pb}', flush=True)
    print(f'  both standalone programs stay positive -> {pc}', flush=True)
    print(f'  both hybrid arms reproduce §1758 + table-only CE + live CE + coverage {ncov} -> '
          f'control {pd}', flush=True)

    r2 = {'config': {'cells': [list(c) for c in CELLS], 'arms': list(ARMS),
                     'hybrid': 'table where the token was covered at fit, LIVE module elsewhere (§1661)',
                     'standalone': 'table everywhere; an uncovered token gets the site global mean '
                                   'row. The native module is never called. SAME cost -- the mean '
                                   'row is already stored and already counted.',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'baseline': {e: {k: (round(v, 5) if isinstance(v, float) else v)
                           for k, v in base[e].items()} for e in base},
          'cells': out, 'standalone_loss_fraction': {'fidelity_point': lf_fid,
                                                     'efficiency_point': lf_eff},
          'predictions': {'pred_a_standalone_loses_under_30pc': bool(pa),
                          'pred_b_loss_larger_at_starved_table': bool(pb),
                          'pred_c_standalone_stays_positive': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
