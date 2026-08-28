# STANDALONE, SCORED WHERE THE TWO ARMS ACTUALLY DIFFER
#
# §1761 set out to price the fallback -- the 24.12% / 25.41% of scored positions where the hybrid hook
# (§1661) abandons the table and runs the LIVE module -- and returned 0.00% loss in both cells. It
# could not have returned anything else. Its `ce()` scores only COVERED positions, and at a covered
# position the hybrid and standalone hooks emit the identical tensor by construction. The arms could
# differ only through propagation, and the run reported a loss fraction computed from values already
# rounded to five decimals, which is LESSON 36 for the third time.
#
# The hooks do differ -- verified directly on synthetic tensors before writing this: at an uncovered
# token the hybrid keeps the live output and the standalone takes the table's mean row.
#
# WHAT "STANDALONE" DOES AND DOES NOT MEAN HERE (Codex red team, 2026-08-28, taken in full). Both arms
# are POST-FORWARD hooks, so the native module executes at every site in both arms; the hook only
# replaces what the module returns. So this measures a program whose OUTPUT never depends on the
# native module at an uncovered position -- zero-native-OUTPUT -- and NOT a zero-native-CALL program.
# The earlier sentence "the native module is never called" is mechanically false and is withdrawn.
# Two consequences: the compute cost of the native forward is not removed by either arm, and for
# attention the `v1` value-bus element of the returned tuple is passed through UNCHANGED in both arms
# (§1682's standing scope note), so that dependency survives in the standalone arm too. The reals
# figures are storage for the substituted output path only, and always were.
#
# So: score BOTH populations in one pass, keep full precision, and let the two questions separate.
#   `all`  every scored position from 64 on. This is the honest standalone population and the one the
#          caveat has always been about.
#   `cov`  covered positions only. Every published figure uses this, the controls need it, and the
#          difference between arms here is a real if narrow quantity: how far the damage done at an
#          uncovered position PROPAGATES to a covered one through attention.
#
# ROLES. Fitting uses the fit rows; both eval roles reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a THE FALLBACK IS EXPENSIVE ON ALL POSITIONS: at the fidelity point the standalone program
#          loses at least 10% of the hybrid's held-out recovery. If FALSE, the mean row is a good
#          enough stand-in for a quarter of positions and the caveat I have attached to six sections
#          was overstated -- which would be a relief and should be said plainly.
#   pred_b §1761's NULL SURVIVES AT FULL PRECISION: on covered positions the two arms differ by less
#          than 1e-3 nats. If TRUE, damage at an uncovered position is essentially LOCAL and does not
#          propagate; if FALSE, §1761's 0.00% was rounding and there is a propagation term worth its
#          own measurement.
#   pred_c THE LOSS IS LARGER WHERE THE TABLE IS STARVED: table rank 8 loses a larger fraction than
#          table rank 64, because a rank-8 mean row is a worse stand-in. If FALSE the loss is about
#          which TOKENS are uncovered rather than about table quality.
#   pred_d CONTROLS: the covered-position hybrid arms reproduce §1758's +0.77602 / +0.78536 and
#          +0.40210 / +0.41052 within 0.002; table-only covered CE reproduces 7.35114; and the
#          ALL-position live CE reproduces **3.13704**, the value §1728's baseline assert fired on
#          when I first scored every position by mistake. That failure is now a known answer.
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
OUT = PT + 'ops/standalone_all_position_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
# all-position live CE, known from §1728's baseline assert firing at 3.13704 on skip7000
S1728_ALLPOS_LIVE = 3.13704
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
def ce_both(rows, hooks=()):
    """CE on BOTH scoring populations in one pass.

    §1761's defect: it scored only COVERED positions, where the hybrid and standalone hooks are
    identical by construction, so the arms could differ only through propagation and the run could
    not see the thing the caveat is about. `all` is the honest standalone population; `cov` is kept
    because every published figure uses it and the controls need it.
    """
    acc = {'cov': [0.0, 0], 'all': [0.0, 0]}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        cov = COV['seen'][idx[:, 64:]]
        acc['cov'][0] += float(e[cov].sum()); acc['cov'][1] += int(cov.sum())
        acc['all'][0] += float(e.sum()); acc['all'][1] += int(e.numel())
    sweep(rows, hooks=hooks, score=score)
    return {k: acc[k][0] / acc[k][1] for k in acc}


@torch.no_grad()
def ce(rows, hooks=()):
    return ce_both(rows, hooks)['cov']


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
    print(f'STANDALONE, SCORED WHERE IT DIFFERS | cells {CELLS} x arms {ARMS} | §1761 scored only '
          f'COVERED positions, where the two arms are identical by construction | DISCOVERY ONLY',
          flush=True)

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
        cl = ce_both(e)
        tb = ce_both(e, [(st, table_hook(tables[st], seen)) for st in sites])
        assert abs(cl['cov'] - ce_ref) <= 1e-2, f'{ename} covered live CE {cl["cov"]:.5f} != {ce_ref}'
        base[ename] = {'live': cl, 'table_only_hybrid': tb,
                       'stake': {k: tb[k] - cl[k] for k in cl}}
        print(f'  {ename}: live cov {cl["cov"]:.5f} all {cl["all"]:.5f} | hybrid table-only cov '
              f'{tb["cov"]:.5f} all {tb["all"]:.5f}', flush=True)

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
            rec = {}
            for e in ev:
                c1 = ce_both(ev[e], [(st, prog_hook(tbl_r[st], seen, installed[st], NBLK, sa))
                                     for st in sites])
                rec[e] = {k: base[e]['table_only_hybrid'][k] - c1[k] for k in c1}
            total = (tcost + 36 * 2 * cr * D) / 1e6
            key = f't{tr}_c{cr}_{arm}'
            # FULL precision retained: §1761 computed its loss fraction from values already rounded
            # to five decimals and reported 0.00%, which is LESSON 36 for the third time.
            out[key] = {'table_rank': tr, 'corr_rank': cr, 'arm': arm,
                        'total_cost_M': round(total, 4),
                        'recovered_full_precision': {e: dict(rec[e]) for e in rec},
                        'frac_of_stake': {e: {k: rec[e][k] / base[e]['stake'][k] for k in rec[e]}
                                          for e in rec}}
            print(f'  table {tr:3d} corr {cr:3d} {arm:11s}: cost {total:8.4f}M | ' + '  '.join(
                f'{e} cov {rec[e]["cov"]:+.6f} all {rec[e]["all"]:+.6f}' for e in rec)
                + f'   [{time.time() - t0:.0f}s]', flush=True)
        del tbl_r
        torch.cuda.empty_cache()

    ho = 'skip11000'

    def delta(tr, cr, pop):
        h = out[f't{tr}_c{cr}_hybrid']['recovered_full_precision'][ho][pop]
        s = out[f't{tr}_c{cr}_standalone']['recovered_full_precision'][ho][pop]
        return h, s, (h - s) / h if abs(h) > 1e-12 else None

    hA, sA, lA = delta(*CELLS[0], 'all')
    hB, sB, lB = delta(*CELLS[1], 'all')
    hC, sC, lC = delta(*CELLS[0], 'cov')
    pa = lA is not None and lA >= 0.10
    pb = abs(hC - sC) < 1e-3
    pc = (lA is not None and lB is not None and lB > lA)
    pd = (all(abs(out[f'{k}_hybrid']['recovered_full_precision'][e]['cov'] - v) <= 0.002
              for k, kv in S1758_HYBRID.items() for e, v in kv.items())
          and abs(base['skip7000']['table_only_hybrid']['cov'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live']['all'] - S1728_ALLPOS_LIVE) <= 1e-3
          and ncov == NCOV and fits_ok)

    print(f'\n  ALL-position: standalone loses {lA:.2%} at the fidelity point '
          f'({hA:+.6f} -> {sA:+.6f}) -> >=10% {pa}', flush=True)
    print(f'  COVERED-position difference {abs(hC - sC):.2e} nats -> §1761\'s null holds at full '
          f'precision, damage is local {pb}', flush=True)
    print(f'  loss larger at the starved table ({lB:.2%} vs {lA:.2%}) -> {pc}', flush=True)
    print(f'  §1758 covered arms + table-only CE + ALL-position live CE {base["skip7000"]["live"]["all"]:.5f} '
          f'vs §1728 {S1728_ALLPOS_LIVE} + coverage {ncov} -> control {pd}', flush=True)

    r2 = {'config': {'cells': [list(c) for c in CELLS], 'arms': list(ARMS),
                     'populations': 'cov = positions whose token was covered at fit (every published '
                                    'figure uses this); all = every scored position from 64 on, which '
                                    'is where the two arms actually differ',
                     'WHY': '§1761 scored only COVERED positions, where the hybrid and standalone '
                            'hooks are identical by construction, and reported a 0.00% difference '
                            'computed from values already rounded to five decimals.',
                     'ZERO_NATIVE_OUTPUT_NOT_ZERO_NATIVE_CALL': 'both arms are post-forward hooks, so '
                            'the native module still executes at every site; only its OUTPUT is '
                            'replaced. Attention `v1` is passed through unchanged in both arms '
                            '(§1682). Compute is not removed and the v1 dependency is not removed.',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'baseline': {e: {k: (v if not isinstance(v, dict) else {kk: round(vv, 6) for kk, vv in v.items()})
                           for k, v in base[e].items()} for e in base},
          'cells': out,
          'standalone_loss_fraction_all_positions': {'fidelity_point': lA, 'efficiency_point': lB},
          'covered_position_difference_nats': abs(hC - sC),
          'predictions': {'pred_a_allpos_loss_at_least_10pc': bool(pa),
                          'pred_b_covered_null_holds': bool(pb),
                          'pred_c_loss_larger_at_starved_table': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
