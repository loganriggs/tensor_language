# FITTING THE PROGRAM TO A DOWNSTREAM OBJECTIVE
#
# The thread's arithmetic, held out: the 36 sites' `table + x W_8` corrections are worth +1.7460 nats
# measured ONE AT A TIME (§1747), -0.5462 installed together after a simultaneous fit (§1747), and
# +0.3858 after §1669's interleaved bottom-up compile (§1748). §1749 then showed three passes of
# coordinate descent change that by EXACTLY ZERO to five decimals, because a transformer is causal in
# depth: a site's fit depends only on what is compiled below it, so one bottom-up pass is already a
# fixed point. Ordering is done. It cannot be the remaining 78%.
#
# What is left is the OBJECTIVE. Every W so far minimises the error in its own module's output, while
# the program is judged by final cross-entropy. A map fitted to reproduce its own write knows nothing
# about how its residual is amplified or cancelled by the blocks above it. This run replaces the local
# objective with the global one: hold the tables fixed, initialise every W at §1748's interleaved
# solution, and train all 36 rank-8 factor pairs (663,552 parameters, 0.664M reals -- the same program
# cost as before) by gradient descent on final CE.
#
# Initialising AT the interleaved solution matters twice: the run starts from a known +0.3858 rather
# than from noise, so any movement is attributable to the objective; and step 0 is an exact
# cross-script control on §1748.
#
# ROLES. Training uses the FIT rows only. Both eval roles are untouched by training and both are
# reported. DISCOVERY ONLY -- neither role is clean for site-set questions after §1739-§1749.
#
# Registered predictions, TWO-SIDED per LESSONS 31, each checked against the others' outcomes so no
# arm is decided by another (addenda 1 and 2 to LESSON 31, both earned):
#   pred_a THE DOWNSTREAM OBJECTIVE IMPROVES ON THE LOCAL ONE, held out: the trained program recovers
#          more than +0.3858 on skip11000. If FALSE, the local per-site objective was already
#          extracting everything this program class can deliver, the composition gap is a property of
#          the CLASS rather than of the fitting, and no amount of better fitting will close it.
#   pred_b IT CLOSES THE COMPOSITION GAP: the trained program exceeds +1.7460, the sum of the 36 solo
#          recoveries. Scored independently of pred_a, since it can improve without reaching that.
#          If TRUE, whole-model compilation at 0.664M reals is a solved problem and everything after
#          this is engineering.
#   pred_c THE GAIN IS NOT A FIT-ROW ARTIFACT: the improvement over initialisation is within 25% of
#          each other on the two eval roles (ratio in [0.75, 1.33]). Both are held out from training,
#          so a large divergence would mean the program has latched onto something specific to 192
#          particular rows rather than to the model.
#   pred_d CONTROLS: step 0 reproduces §1748's +0.4063 and +0.3858 within 0.002 -- the same program
#          re-evaluated by a third script -- plus table-only CE 7.35114 within 0.005, live CE 3.29205
#          within 1e-3, and fit coverage 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANK = 8
RIDGE = 1e-2
STEPS = 300
BATCH = 4
LR = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/downstream_objective_compile_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1748_INTERLEAVED = {'skip7000': 0.40631, 'skip11000': 0.38578}
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
    return ((U[:, :RANK] * S[:RANK].sqrt()).float(),
            (torch.diag(S[:RANK].sqrt()) @ Vh[:RANK]).float()), n['k']


def main():
    t0 = time.time()
    for p in m.parameters():
        p.requires_grad_(False)
    fit = load(FIT_ROWS)
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'DOWNSTREAM OBJECTIVE COMPILE | rank {RANK} at all 36 sites | init at §1748 interleaved, '
          f'then {STEPS} Adam steps on final CE | DISCOVERY ONLY', flush=True)

    with torch.no_grad():
        COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
        tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    installed, fits_ok = {}, True
    for st in order:
        (A, B), nk = fit_one(fit, st, tables, seen, installed)
        fits_ok = fits_ok and (nk == 24576)
        installed[st] = (A, B)
    print(f'  interleaved init built ({time.time() - t0:.0f}s)', flush=True)

    params = []
    for st in sites:
        A, B = installed[st]
        A = A.clone().requires_grad_(True); B = B.clone().requires_grad_(True)
        installed[st] = (A, B)
        params += [A, B]

    def hooks_now():
        return [(st, factor_hook(tables[st], seen, *installed[st])) for st in sites]

    ev = {}
    base = {}
    for ename, epath, ce_ref in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        with torch.no_grad():
            cl = ce(e, [])
            tb = ce(e, [(st, table_hook(tables[st], seen)) for st in sites])
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        base[ename] = {'live': cl, 'table_only': tb, 'stake': tb - cl}

    with torch.no_grad():
        start = {e: base[e]['table_only'] - ce(ev[e], hooks_now()) for e in ev}
    print(f'  step 0 (= §1748): ' + '  '.join(f'{e} {start[e]:+.5f}' for e in start), flush=True)

    opt = torch.optim.Adam(params, lr=LR)
    nrow = fit.shape[0]
    g = torch.Generator().manual_seed(1749)
    for step in range(STEPS):
        sel = torch.randint(0, nrow, (BATCH,), generator=g)
        bb = fit[sel]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward(idx, hooks_now())
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = COV['seen'][idx[:, 64:]]
        loss = e[cov].mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step + 1) % 60 == 0:
            with torch.no_grad():
                cur = {e2: base[e2]['table_only'] - ce(ev[e2], hooks_now()) for e2 in ev}
            print(f'    step {step + 1:4d}  train CE {float(loss):.4f}  ' + '  '.join(
                f'{e2} {cur[e2]:+.5f}' for e2 in cur) + f'   [{time.time() - t0:.0f}s]', flush=True)

    with torch.no_grad():
        final = {e: base[e]['table_only'] - ce(ev[e], hooks_now()) for e in ev}
    gain = {e: final[e] - start[e] for e in ev}
    ho = 'skip11000'
    pa = final[ho] > S1748_INTERLEAVED[ho]
    pb = final[ho] > S1747_SUM_OF_PARTS[ho]
    ratio = gain['skip7000'] / gain[ho] if abs(gain[ho]) > 1e-9 else float('nan')
    pc = 0.75 <= ratio <= 1.33
    pd = (all(abs(start[e] - v) <= 0.002 for e, v in S1748_INTERLEAVED.items())
          and abs(base['skip7000']['table_only'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == 5419 and fits_ok)

    print(f'\n  final: ' + '  '.join(
        f'{e} {final[e]:+.5f} ({final[e] / base[e]["stake"]:+.2%})' for e in final), flush=True)
    print(f'  gain over the interleaved init: ' + '  '.join(
        f'{e} {gain[e]:+.5f}' for e in gain), flush=True)
    print(f'\n  beats the local objective held out -> {pa}', flush=True)
    print(f'  reaches the sum of the parts {S1747_SUM_OF_PARTS[ho]:+.4f} -> {pb}', flush=True)
    print(f'  gain ratio between roles {ratio:.3f} in [0.75, 1.33] -> not a fit-row artifact {pc}',
          flush=True)
    print(f'  step 0 reproduces §1748 + table-only + live CE + coverage {ncov} -> control {pd}',
          flush=True)

    r = {'config': {'rank': RANK, 'steps': STEPS, 'batch': BATCH, 'lr': LR,
                    'objective': 'final cross-entropy on the FIT rows, covered positions from 64',
                    'init': '§1748 interleaved bottom-up local fit, so step 0 is a control',
                    'trainable': f'{36 * 2 * RANK * D} reals = 0.664M, the same program cost',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Training touches the FIT rows only; both eval '
                                 'roles are untouched by training.'},
         'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
         'start': {e: round(v, 5) for e, v in start.items()},
         'final': {e: round(v, 5) for e, v in final.items()},
         'gain': {e: round(v, 5) for e, v in gain.items()},
         'reference': {'interleaved_S1748': S1748_INTERLEAVED,
                       'sum_of_parts_S1747': S1747_SUM_OF_PARTS},
         'predictions': {'pred_a_beats_local_objective': bool(pa),
                         'pred_b_closes_composition_gap': bool(pb),
                         'pred_c_transfers_between_roles': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
