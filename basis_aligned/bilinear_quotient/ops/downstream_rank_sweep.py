# DOWNSTREAM RANK SWEEP -- is the remaining composition gap a CAPACITY limit or a CLASS limit?
#
# §1750: training the 36 rank-8 maps on final CE beat the local per-site objective by 43% at identical
# cost (+0.3858 -> +0.5507 held out) and transferred between roles at ratio 1.020. It still reached
# only 32% of +1.7460, the sum of the 36 sites measured one at a time. §1749 had already proved
# ordering irrelevant -- three coordinate-descent passes changed the program by exactly zero, because
# depth causality makes one bottom-up pass a fixed point -- so the residual is not the fit order and
# is only partly the objective.
#
# Two candidates remain and they are distinguishable by one sweep:
#   CAPACITY  rank 8 is too small. Then recovery should climb steeply with rank under the downstream
#             objective, and might reach the sum of the parts.
#   CLASS     a per-token table plus a linear read of the site's own input cannot express what the
#             module does, at any rank. Then the curve flattens well below +1.7460 and more rank is
#             wasted reals.
#
# Under the LOCAL objective the rank curve was already flat (§1746: 37.94 / 38.16 / 38.14), but that
# was measured on a program whose composition was broken, so it says little about this question.
#
# Checkpoint selection is explicit here: §1750 reported its FINAL step and had peaked 180 steps
# earlier, so this run evaluates every 60 steps, selects the best by skip7000, and reports skip11000
# at that step. skip11000 is untouched by training and by selection.
#
# ROLES. Training uses the FIT rows; skip7000 selects the checkpoint; skip11000 is clean of both.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, each checked against the others' outcomes:
#   pred_a RECOVERY INCREASES WITH RANK on the held-out role across 8, 32, 128. If FALSE the class is
#          the binding constraint and additional rank is wasted reals -- which is directly useful,
#          since it fixes rank 8 as the design point and redirects effort to the program class.
#   pred_b RANK 128 REACHES THE SUM OF THE PARTS (+1.7460). Scored independently of pred_a, since the
#          curve can rise without getting there. If TRUE, the composition gap was capacity all along
#          and the whole-model program is solved at 10.6M reals.
#   pred_c DIMINISHING RETURNS: the 8->32 gain exceeds the 32->128 gain. If FALSE the curve is still
#          accelerating at rank 128 and the sweep is under-budgeted -- the defect LESSONS 31's
#          addendum records twice, so it is scored rather than assumed away.
#   pred_d CONTROLS: the rank-8 START reproduces §1748's +0.38578 within 0.002 (deterministic), the
#          rank-8 BEST reproduces §1750's peak +0.57418 within 0.03 (stochastic training, hence the
#          looser bar, stated in advance), plus table-only CE 7.35114, live CE 3.29205, coverage
#          5419 of 50257, and all 36 initialisation fits on the full 24576 positions.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANKS = (8, 32, 128)
RIDGE = 1e-2
STEPS = 360
EVERY = 60
BATCH = 4
LR = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/downstream_rank_sweep_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1748_INTERLEAVED = {'skip7000': 0.40631, 'skip11000': 0.38578}
S1750_RANK8_FINAL = {'skip7000': 0.57457, 'skip11000': 0.55074}
S1750_RANK8_PEAK = {'skip7000': 0.59388, 'skip11000': 0.57418}
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


def factor_hook(tbl, seen, A, B):
    """table[token] + x (A B). A and B are the trainable parameters; the table is frozen."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        corr = ((args[0].reshape(-1, D).float() @ A) @ B).reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub + corr, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def forward(idx, hooks):
    STATE['idx'] = idx
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def ce(rows, hooks):
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward(idx, hooks)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        cov = COV['seen'][idx[:, 64:]]
        tot += float(e[cov].sum()); cnt += int(cov.sum())
    return tot / cnt


@torch.no_grad()
def fit_tables(rows, sites):
    s = {st: torch.zeros(50257, D, device=DEV) for st in sites}
    c = torch.zeros(50257, device=DEV)
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        STATE['idx'] = idx
        cap = {}

        def mk(st):
            def hook(mod, args, out):
                cap[st] = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
                return None
            return hook
        hs = [mod_of(*st).register_forward_hook(mk(st)) for st in sites]
        try:
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
        finally:
            for h in hs:
                h.remove()
        t = idx.reshape(-1)
        for st in sites:
            s[st].index_add_(0, t, cap[st])
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
    seen = c > 0
    out = {}
    for st in sites:
        mean = s[st].sum(0) / c.sum()
        tbl = mean.unsqueeze(0).repeat(50257, 1)
        tbl[seen] = s[st][seen] / c[seen].unsqueeze(1)
        out[st] = tbl
    return out, seen


@torch.no_grad()
def fit_one(rows, st, tables, seen, installed):
    """§1748's per-site local fit, against the already-compiled prefix. Reused verbatim for init."""
    xtx = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    xtr = torch.zeros(D, D, device=DEV, dtype=torch.float64)
    n = {'k': 0}

    def cap(mod, args, out):
        y = (out[0] if isinstance(out, tuple) else out).reshape(-1, D).double()
        x = args[0].reshape(-1, D).double()
        nonlocal xtx, xtr
        xtx += x.T @ x
        xtr += x.T @ (y - tables[st][STATE['idx'].reshape(-1)].double())
        n['k'] += x.shape[0]
        return None
    hooks = [(st, cap)] + [(s2, factor_hook(tables[s2], seen, A2, B2))
                           for s2, (A2, B2) in installed.items()]
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        forward(bb[:, :-1].to(DEV).contiguous(), hooks)
    assert n['k'] > 0, f'fit at {st} never fired'
    A = xtx + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (n['k'] / D)
    U, S, Vh = torch.linalg.svd(torch.linalg.solve(A, xtr))
    return {r: ((U[:, :r] * S[:r].sqrt()).float(),
                (torch.diag(S[:r].sqrt()) @ Vh[:r]).float()) for r in RANKS}, n['k']


def main():
    t0 = time.time()
    for p in m.parameters():
        p.requires_grad_(False)
    fit = load(FIT_ROWS)
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'DOWNSTREAM RANK SWEEP | ranks {RANKS} at all 36 sites | init at the §1748 interleaved '
          f'solution, {STEPS} Adam steps on final CE, checkpoint selected on skip7000 | '
          f'DISCOVERY ONLY', flush=True)

    with torch.no_grad():
        COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
        tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    # ONE interleaved compile gives every rank: the SVD is taken once per site and truncated. The
    # context each site is fitted in uses the TOP rank, so lower ranks inherit a context built with
    # more capacity than they have -- stated because it is a real asymmetry, and it can only
    # DISADVANTAGE the low ranks, which is the safe direction for a rank sweep.
    init, fits_ok = {}, True
    for st in order:
        allr, nk = fit_one(fit, st, tables, seen,
                           {s2: v[RANKS[-1]] for s2, v in init.items()})
        fits_ok = fits_ok and (nk == 24576)
        init[st] = allr
    print(f'  interleaved init built for all ranks ({time.time() - t0:.0f}s)', flush=True)

    ev, base = {}, {}
    for ename, epath, ce_ref in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        with torch.no_grad():
            cl = ce(e, [])
            tb = ce(e, [(st, table_hook(tables[st], seen)) for st in sites])
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        base[ename] = {'live': cl, 'table_only': tb, 'stake': tb - cl}
    print(f'  table-only {base["skip7000"]["table_only"]:.5f} | stakes ' + '  '.join(
        f'{e} {base[e]["stake"]:.4f}' for e in base), flush=True)

    out = {}
    for rank in RANKS:
        cur = {}
        params = []
        for st in sites:
            A, B = init[st][rank]
            A = A.clone().requires_grad_(True); B = B.clone().requires_grad_(True)
            cur[st] = (A, B)
            params += [A, B]

        def hooks_now():
            return [(st, factor_hook(tables[st], seen, *cur[st])) for st in sites]

        with torch.no_grad():
            start = {e: base[e]['table_only'] - ce(ev[e], hooks_now()) for e in ev}
        best = {'sel': start['skip7000'], 'step': 0, 'vals': dict(start)}
        opt = torch.optim.Adam(params, lr=LR)
        g = torch.Generator().manual_seed(1750 + rank)
        nrow = fit.shape[0]
        traj = [{'step': 0, **{e: round(start[e], 5) for e in start}}]
        for step in range(STEPS):
            sel = torch.randint(0, nrow, (BATCH,), generator=g)
            bb = fit[sel]
            idx = bb[:, :-1].to(DEV).contiguous()
            lg = forward(idx, hooks_now())
            tg = bb[:, 1:].to(DEV)
            e1 = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                 reduction='none').reshape(tg.shape)[:, 64:]
            cov = COV['seen'][idx[:, 64:]]
            loss = e1[cov].mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % EVERY == 0:
                with torch.no_grad():
                    v = {e2: base[e2]['table_only'] - ce(ev[e2], hooks_now()) for e2 in ev}
                traj.append({'step': step + 1, **{e2: round(v[e2], 5) for e2 in v}})
                if v['skip7000'] > best['sel']:
                    best = {'sel': v['skip7000'], 'step': step + 1, 'vals': dict(v)}
        out[rank] = {'start': {e: round(start[e], 5) for e in start},
                     'best_step': best['step'],
                     'best': {e: round(best['vals'][e], 5) for e in best['vals']},
                     'best_frac': {e: round(best['vals'][e] / base[e]['stake'], 5)
                                   for e in best['vals']},
                     'cost_M': round(36 * 2 * rank * D / 1e6, 4),
                     'nats_per_Mreal': round(best['vals']['skip11000']
                                             / (36 * 2 * rank * D / 1e6), 4),
                     'trajectory': traj}
        r = out[rank]
        print(f'\n  rank {rank:3d}: start {start["skip11000"]:+.5f} -> best {r["best"]["skip11000"]:+.5f} '
              f'({r["best_frac"]["skip11000"]:+.2%}) at step {r["best_step"]} | '
              f'{r["cost_M"]:.3f}M reals | {r["nats_per_Mreal"]:.3f} nats/M   '
              f'[{time.time() - t0:.0f}s]', flush=True)
        print(f'    trajectory (skip11000): ' + ' '.join(
            f'{t["step"]}:{t["skip11000"]:+.4f}' for t in traj), flush=True)
        del params, cur
        torch.cuda.empty_cache()

    ho = 'skip11000'
    vals = [out[r]['best'][ho] for r in RANKS]
    pa = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    pb = vals[-1] > S1747_SUM_OF_PARTS[ho]
    gains = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    pc = all(gains[i] > gains[i + 1] for i in range(len(gains) - 1))
    pd = (abs(out[RANKS[0]]['start'][ho] - S1748_INTERLEAVED[ho]) <= 0.002
          and abs(out[RANKS[0]]['best'][ho] - S1750_RANK8_PEAK[ho]) <= 0.03
          and abs(base['skip7000']['table_only'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == 5419 and fits_ok)

    print(f'\n  recovery increases with rank under the downstream objective {vals} -> {pa}',
          flush=True)
    print(f'  rank {RANKS[-1]} exceeds the sum of the parts {S1747_SUM_OF_PARTS[ho]:+.4f} -> {pb}',
          flush=True)
    print(f'  gains per rank step {[round(x, 4) for x in gains]} diminishing -> {pc}', flush=True)
    print(f'  rank-8 start and peak reproduce §1748/§1750 + table-only + live CE + coverage '
          f'{ncov} -> control {pd}', flush=True)

    r = {'config': {'ranks': list(RANKS), 'steps': STEPS, 'batch': BATCH, 'lr': LR,
                    'objective': 'final cross-entropy on the FIT rows, covered positions from 64',
                    'checkpoint': 'best of the evaluated steps by skip7000; skip11000 reported at '
                                  'that step. §1750 reported its FINAL step and peaked earlier, so '
                                  'selection is made explicit here rather than left to the schedule.',
                    'init': 'one interleaved compile, SVD truncated per rank; the fit CONTEXT uses '
                            'the top rank, which can only disadvantage the lower ranks',
                    'ROLE_NOTE': 'DISCOVERY ONLY. skip7000 selects the checkpoint; skip11000 is '
                                 'untouched by both training and selection.'},
         'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
         'by_rank': {str(k): v for k, v in out.items()},
         'reference': {'interleaved_S1748': S1748_INTERLEAVED, 'rank8_S1750': S1750_RANK8_FINAL,
                       'sum_of_parts_S1747': S1747_SUM_OF_PARTS},
         'predictions': {'pred_a_increases_with_rank': bool(pa),
                         'pred_b_reaches_sum_of_parts': bool(pb),
                         'pred_c_diminishing': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
