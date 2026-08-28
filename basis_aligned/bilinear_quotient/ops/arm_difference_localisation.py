# WHERE DO THE TWO ARMS ACTUALLY DIFFER? -- localising §1762's impossible zero.
#
# §1762 reported the hybrid and standalone programs differing by 0.9 nats on ALL-position scoring and
# by exactly 0.00e+00 on COVERED-position scoring, in the same forward pass. §1763 then showed that a
# perturbation at an uncovered position moves later COVERED positions by up to 0.118 nats, so the
# exact zero cannot be physical -- a quarter of positions were substituted differently and every later
# covered position could feel it.
#
# This does not re-measure anything. It runs both arms on ONE batch and prints where the per-position
# loss difference is nonzero: how many positions, how large, and how they split between covered and
# uncovered. That either finds the defect or establishes that the two arms genuinely emit identical
# tensors at covered positions, which after §1763 would itself demand an explanation.
#
# ROLES. skip7000, one batch. DIAGNOSTIC. Opens no role and produces no scientific figure.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE ARMS DIFFER SOMEWHERE: at least one position has a nonzero per-position loss
#          difference. If FALSE the two arms are producing bit-identical logits everywhere, and the
#          0.9-nat all-position gap in §1762 came from something other than the arms -- which would
#          invalidate that run entirely rather than just its covered column.
#   pred_b THE DIFFERENCE REACHES COVERED POSITIONS: at least one COVERED position differs by more
#          than 1e-9. If FALSE, §1762's zero is reproduced here at position granularity and the
#          contradiction with §1763 is real and sharper -- the most informative outcome, because it
#          would mean the substitution somehow cannot influence a covered position while an artificial
#          poke can.
#   pred_c THE UNCOVERED POSITIONS CARRY MOST OF IT: the summed absolute difference over uncovered
#          positions exceeds that over covered ones. That is what a working standalone arm should look
#          like -- large direct changes where the table replaces the module, smaller propagated ones
#          elsewhere.
#   pred_d CONTROLS: the coverage mask is 5419 of 50257 and the batch has both covered and uncovered
#          scored positions, so neither category is empty and neither prediction is vacuous.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
TABLE_RANK, CORR_RANK, NBLK, RIDGE = 64, 128, 1, 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/arm_difference_localisation_results.json'
EVAL = PT + '.rowcache/fineweb_n192_skip7000.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
STATE = {}
COV = {}


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
    return {CORR_RANK: ((U[:, :CORR_RANK] * S[:CORR_RANK]) @ Vh[:CORR_RANK]).float()}, n['k']


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
    fit = load(FIT_ROWS)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    print(f'ARM DIFFERENCE LOCALISATION | table {TABLE_RANK} corr {CORR_RANK} | coverage {ncov} | '
          f'DIAGNOSTIC', flush=True)

    tbl_r, _ = compress_tables(tables, seen, TABLE_RANK)
    progs = {}
    for arm in ('hybrid', 'standalone'):
        sa = (arm == 'standalone')
        installed = {}
        for st in order:
            allr, _ = fit_one(fit, st, tbl_r, seen, installed, NBLK, sa)
            installed[st] = allr[CORR_RANK]
        progs[arm] = installed
        print(f'  compiled {arm} ({time.time() - t0:.0f}s)', flush=True)

    ev = load(EVAL)
    bb = ev[:8]
    idx = bb[:, :-1].to(DEV).contiguous()
    tg = bb[:, 1:].to(DEV)
    STATE['idx'] = idx
    out_l = {}
    for arm in ('hybrid', 'standalone'):
        sa = (arm == 'standalone')
        hooks = [mod_of(*st).register_forward_hook(
            prog_hook(tbl_r[st], seen, progs[arm][st], NBLK, sa)) for st in sites]
        try:
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            out_l[arm] = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                         reduction='none').reshape(tg.shape).double()
        finally:
            for h in hooks:
                h.remove()

    d = (out_l['standalone'] - out_l['hybrid']).abs()[:, 64:]
    cov = seen[idx[:, 64:]]
    nz = d > 0
    stats = {'n_scored': int(d.numel()), 'n_nonzero': int(nz.sum()),
             'n_covered': int(cov.sum()), 'n_uncovered': int((~cov).sum()),
             'n_nonzero_covered': int((nz & cov).sum()),
             'n_nonzero_uncovered': int((nz & ~cov).sum()),
             'max_delta': float(d.max()),
             'max_delta_covered': float(d[cov].max()) if cov.any() else None,
             'max_delta_uncovered': float(d[~cov].max()) if (~cov).any() else None,
             'sum_abs_covered': float(d[cov].sum()) if cov.any() else 0.0,
             'sum_abs_uncovered': float(d[~cov].sum()) if (~cov).any() else 0.0}
    for k, v in stats.items():
        print(f'    {k:24s} {v}', flush=True)

    pa = stats['n_nonzero'] > 0
    pb = stats['max_delta_covered'] is not None and stats['max_delta_covered'] > 1e-9
    pc = stats['sum_abs_uncovered'] > stats['sum_abs_covered']
    pd = ncov == NCOV and stats['n_covered'] > 0 and stats['n_uncovered'] > 0

    print(f'\n  the arms differ somewhere -> {pa}', flush=True)
    print(f'  the difference reaches COVERED positions ({stats["max_delta_covered"]:.3e}) -> {pb}',
          flush=True)
    print(f'  uncovered positions carry more absolute difference '
          f'({stats["sum_abs_uncovered"]:.4f} vs {stats["sum_abs_covered"]:.4f}) -> {pc}', flush=True)
    print(f'  coverage {ncov} and both categories non-empty -> control {pd}', flush=True)

    r = {'config': {'table_rank': TABLE_RANK, 'corr_rank': CORR_RANK, 'batch': 'skip7000 rows 0-7',
                    'WHY': "S1762 reported an exactly-zero covered-position difference; S1763 showed "
                           "a poke at an uncovered position moves later covered ones by 0.118 nats, "
                           "so the zero cannot be physical.",
                    'ROLE_NOTE': 'DIAGNOSTIC. Opens no role, produces no scientific figure.'},
         'stats': stats,
         'predictions': {'pred_a_arms_differ': bool(pa),
                         'pred_b_reaches_covered': bool(pb),
                         'pred_c_uncovered_carries_more': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
