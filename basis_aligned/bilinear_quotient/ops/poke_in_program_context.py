# THE SAME POKE IN BOTH CONTEXTS -- resolving §1763 against §1764.
#
# §1763, on the LIVE model: poking attn0's output at an uncovered position by +10 per channel moves
# later COVERED positions by 0.118 nats, with a positive control confirming the instrument.
# §1764, with the compiled program installed: the hybrid and standalone arms differ at all 335
# uncovered positions by up to 14.54 nats of loss and at ZERO of 1201 covered positions.
#
# Both measurements are careful and they cannot both describe the same system. This applies the
# IDENTICAL poke in both contexts -- live model and program-installed -- so the difference between
# them is the only variable left.
#
# ROLES. skip7000, one batch. DIAGNOSTIC. Opens no role and produces no scientific figure.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a IN PROGRAM CONTEXT AN UNCOVERED POSITION STILL REACHES A LATER COVERED ONE (>1e-6). If
#          TRUE, propagation is intact with the program installed and §1764's exact partition is a
#          defect in the arm comparison itself, which is then where to look. If FALSE, the installed
#          program SUPPRESSES propagation that the live model permits -- a finding about the program,
#          not a bug, and one that would need its own mechanism.
#   pred_b THE POSITIVE CONTROL HOLDS IN PROGRAM CONTEXT: poking a COVERED position moves later
#          covered ones. If FALSE, the whole program context is somehow position-decoupled and pred_a
#          means nothing on its own.
#   pred_c THE LIVE ARM REPRODUCES §1763 (>1e-6), so the two runs agree where they overlap. If FALSE,
#          §1763 itself does not replicate and the contradiction is on that side.
#   pred_d CONTROLS: every poke changes its own position's loss, and coverage is 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/poke_in_program_context_results.json'
EVAL = PT + '.rowcache/fineweb_n192_skip7000.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
MAG = 10.0
TABLE_RANK, CORR_RANK, NBLK, RIDGE = 64, 128, 1, 1e-2
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def poke_hook(pos, mag):
    """Add a constant `mag` to every channel of this site's output at ONE position."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        y2 = y.clone()
        y2[:, pos, :] = y2[:, pos, :] + mag
        return (y2,) + tuple(out[1:]) if isinstance(out, tuple) else y2
    return hook


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
def losses(idx, tg, hooks=()):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
        return F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                               reduction='none').reshape(tg.shape).double()
    finally:
        for h in hs:
            h.remove()


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
    print(f'POKE IN PROGRAM CONTEXT | table {TABLE_RANK} corr {CORR_RANK} | poke {MAG} | '
          f'DIAGNOSTIC', flush=True)

    tbl_r, _ = compress_tables(tables, seen, TABLE_RANK)
    installed = {}
    for st in order:
        allr, _ = fit_one(fit, st, tbl_r, seen, installed, NBLK, False)
        installed[st] = allr[CORR_RANK]
    print(f'  hybrid program compiled ({time.time() - t0:.0f}s)', flush=True)
    prog_hooks = [(st, prog_hook(tbl_r[st], seen, installed[st], NBLK, False)) for st in sites]

    ev = load(EVAL)
    bb = ev[:8]
    idx = bb[:, :-1].to(DEV).contiguous()
    tg = bb[:, 1:].to(DEV)
    STATE['idx'] = idx
    cov = seen[idx]
    row = 0
    unc = [p for p in range(64, T - 8) if not bool(cov[row, p])]
    cvd = [p for p in range(64, T - 8) if bool(cov[row, p])]
    assert unc and cvd, 'need both categories in row 0'
    p_unc, p_cov = unc[0], cvd[0]

    res = {}
    for ctx_name, ctx in (('live_model', []), ('program_installed', prog_hooks)):
        base = losses(idx, tg, ctx)
        for tag, site, pos in (('attn0_uncovered', ('attn', 0), p_unc),
                               ('attn0_covered_control', ('attn', 0), p_cov)):
            # the poke is applied IN ADDITION to whatever the context installs at that site
            hooks = list(ctx) + [(site, poke_hook(pos, MAG))]
            d = (losses(idx, tg, hooks) - base).abs()
            after = torch.arange(T, device=DEV) > pos
            lc = after.unsqueeze(0) & cov
            key = f'{ctx_name}/{tag}'
            res[key] = {'position': pos, 'own_delta': float(d[row, pos]),
                        'max_delta_later_covered': float(d[lc].max()) if lc.any() else None,
                        'n_later_covered': int(lc.sum())}
            r = res[key]
            print(f'    {key:40s} own {r["own_delta"]:.3e} | max later COVERED '
                  f'{r["max_delta_later_covered"]:.3e}', flush=True)

    live_u = res['live_model/attn0_uncovered']['max_delta_later_covered']
    prog_u = res['program_installed/attn0_uncovered']['max_delta_later_covered']
    prog_c = res['program_installed/attn0_covered_control']['max_delta_later_covered']
    pa = prog_u > 1e-6
    pb = prog_c > 1e-6
    pc = live_u > 1e-6
    pd = (all(res[k]['own_delta'] > 1e-6 for k in res) and ncov == NCOV)

    print(f'\n  IN PROGRAM CONTEXT, an uncovered position reaches a later covered one '
          f'({prog_u:.3e}) -> {pa}', flush=True)
    print(f'  in program context, a COVERED position does ({prog_c:.3e}) -> {pb}', flush=True)
    print(f'  on the LIVE model it does, reproducing §1763 ({live_u:.3e}) -> {pc}', flush=True)
    print(f'  every poke landed + coverage {ncov} -> control {pd}', flush=True)

    r2 = {'config': {'table_rank': TABLE_RANK, 'corr_rank': CORR_RANK, 'magnitude': MAG,
                     'WHY': '§1763 (live model) says an uncovered position reaches later covered ones '
                            'by 0.118 nats; §1764 (program installed, two arms) says the arms differ '
                            'at every uncovered position and at NO covered one. This applies the same '
                            'poke in both contexts so the contradiction resolves to one of them.',
                     'ROLE_NOTE': 'DIAGNOSTIC. Opens no role.'},
          'pokes': res,
          'predictions': {'pred_a_propagates_in_program_context': bool(pa),
                          'pred_b_covered_poke_propagates_in_program_context': bool(pb),
                          'pred_c_live_model_reproduces_S1763': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
