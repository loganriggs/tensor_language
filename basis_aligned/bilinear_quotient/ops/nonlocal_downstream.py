# WAS THE RICH CLASS BADLY STEERED, OR DOES IT GENUINELY NOT COMPOSE?
#
# §1752, under a LOCAL per-site objective: adding a lag-1 block and a causal prefix mean lifts the
# median attention site from -23.69% to +8.18% of its own gap and the median MLP from 32.50% to
# 55.16%, while driving the JOINT program from +0.38578 to -0.80171 nats held out. Every feature
# block made each site individually better and the composed model worse.
#
# §1750, on the poorest class: replacing the local objective with final CE was worth +43% at
# identical cost, and the gain transferred between eval roles at ratio 1.020.
#
# Those two results leave one question. A local objective fits each map to reproduce its own module's
# write, which is precisely the thing §1752 shows is anti-correlated with joint fidelity -- so the
# rich class may have been STEERED into its collapse rather than incapable of composing. Under a
# global objective the extra features are capacity the optimiser can use for composition instead of
# for local mimicry.
#
# Both classes, same rank, same schedule, each initialised at its own interleaved local solution so
# that step 0 is an exact control against a published number (§1748 for A, §1752 for C).
#
# ROLES. Training uses the FIT rows; skip7000 selects the checkpoint; skip11000 is clean of both.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other so no arm is decided
# by another's outcome:
#   pred_a THE RICH CLASS WINS ONCE TRAINED GLOBALLY: C beats A on the held-out role. If FALSE, the
#          extra features do not help even when the objective is right, and the compilation program
#          should stop adding local expressiveness and look for a structurally different class.
#   pred_b TRAINING RESCUES C FROM NEGATIVE: C ends above zero. Scored independently of pred_a, since
#          C can recover to positive and still lose to A.
#   pred_c THE RICHER CLASS GAINS MORE FROM THE GLOBAL OBJECTIVE than the poorer one. This is the
#          steering hypothesis stated directly: if the local objective is what broke C, then fixing
#          the objective should move C further than it moves A. If FALSE while pred_a is TRUE, C's
#          advantage is in its initialisation rather than in what training finds.
#   pred_d CONTROLS: A's step 0 reproduces §1748's +0.40631 / +0.38578 within 0.002 and C's step 0
#          reproduces §1752's -0.78273 / -0.80171 within 0.002 -- two exact cross-script controls on
#          two different programs -- plus table-only CE 7.35114, live CE 3.29205, coverage 5419 of
#          50257, and every per-site fit on the full 24576 positions.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANK = 8
RIDGE = 1e-2
VARIANTS = {'A_current': 1, 'C_lag1_prefixmean': 3}
STEPS = 240; EVERY = 40; BATCH = 4; LR = 1e-3
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/nonlocal_downstream_results.json'
MAP = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1748_A = {'skip7000': 0.40631, 'skip11000': 0.38578}
S1752_C = {'skip7000': -0.78273, 'skip11000': -0.80171}
S1751_A_TRAINED = {'skip11000': 0.60060}
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


def prog_hook(tbl, seen, AB, nblk):
    """table[token] + features(x) (A B). A and B are the trainable factors; the table is frozen."""
    A, B = AB

    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        f = features(args[0].float(), nblk).reshape(-1, nblk * D)
        sub = sub + ((f @ A) @ B).reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def forward_grad(idx, hooks):
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
    return ((U[:, :RANK] * S[:RANK].sqrt()).float(),
            (torch.diag(S[:RANK].sqrt()) @ Vh[:RANK]).float()), n['k']


def main():
    t0 = time.time()
    assert_features_are_causal()
    print('  feature causality known-answer check PASSED (LESSONS 34)', flush=True)
    for p in m.parameters():
        p.requires_grad_(False)
    fit = load(FIT_ROWS)
    order = [(k, L) for L in range(18) for k in ('attn', 'mlp')]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'NON-LOCAL CLASS UNDER A DOWNSTREAM OBJECTIVE | rank {RANK} | {list(VARIANTS)} | '
          f'{STEPS} Adam steps on final CE, checkpoint on skip7000 | DISCOVERY ONLY', flush=True)

    with torch.no_grad():
        COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
        tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)

    ev, base = {}, {}
    for ename, epath, ce_ref in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        with torch.no_grad():
            cl = ce(e)
            tb = ce(e, [(st, table_hook(tables[st], seen)) for st in sites])
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        base[ename] = {'live': cl, 'table_only': tb, 'stake': tb - cl}

    out, fits_ok = {}, True
    for vname, nblk in VARIANTS.items():
        installed = {}
        with torch.no_grad():
            for st in order:
                AB, nk = fit_one(fit, st, tables, seen, installed, nblk)
                fits_ok = fits_ok and (nk == 24576)
                installed[st] = AB
        cur, params = {}, []
        for st in sites:
            A, B = installed[st]
            A = A.clone().requires_grad_(True); B = B.clone().requires_grad_(True)
            cur[st] = (A, B); params += [A, B]

        def hooks_now():
            return [(st, prog_hook(tables[st], seen, cur[st], nblk)) for st in sites]

        with torch.no_grad():
            start = {e: base[e]['table_only'] - ce(ev[e], hooks_now()) for e in ev}
        best = {'sel': start['skip7000'], 'step': 0, 'vals': dict(start)}
        opt = torch.optim.Adam(params, lr=LR)
        g = torch.Generator().manual_seed(1752)
        traj = [{'step': 0, **{e: round(start[e], 5) for e in start}}]
        for step in range(STEPS):
            sel = torch.randint(0, fit.shape[0], (BATCH,), generator=g)
            bb = fit[sel]
            idx = bb[:, :-1].to(DEV).contiguous()
            lg = forward_grad(idx, hooks_now())
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
        cost = round(36 * RANK * (nblk * D + D) / 1e6, 4)
        out[vname] = {'n_feature_blocks': nblk, 'cost_M': cost,
                      'start': {e: round(start[e], 5) for e in start},
                      'best_step': best['step'],
                      'best': {e: round(best['vals'][e], 5) for e in best['vals']},
                      'best_frac': {e: round(best['vals'][e] / base[e]['stake'], 5)
                                    for e in best['vals']},
                      'train_gain': {e: round(best['vals'][e] - start[e], 5) for e in start},
                      'nats_per_Mreal': round(best['vals']['skip11000'] / cost, 4),
                      'trajectory': traj}
        r = out[vname]
        print(f'\n  {vname:20s} {nblk} block(s), {cost:.3f}M: start {start["skip11000"]:+.5f} -> '
              f'best {r["best"]["skip11000"]:+.5f} ({r["best_frac"]["skip11000"]:+.2%}) at step '
              f'{r["best_step"]} | trained gain {r["train_gain"]["skip11000"]:+.5f} | '
              f'{r["nats_per_Mreal"]:.3f} nats/M   [{time.time() - t0:.0f}s]', flush=True)
        print(f'    trajectory (skip11000): ' + ' '.join(
            f'{t["step"]}:{t["skip11000"]:+.4f}' for t in traj), flush=True)
        del params, cur
        torch.cuda.empty_cache()

    ho = 'skip11000'
    A_, C_ = 'A_current', 'C_lag1_prefixmean'
    pa = out[C_]['best'][ho] > out[A_]['best'][ho]
    pb = out[C_]['best'][ho] > 0.0
    pc = out[C_]['train_gain'][ho] > out[A_]['train_gain'][ho]
    pd = (all(abs(out[A_]['start'][e] - v) <= 0.002 for e, v in S1748_A.items())
          and all(abs(out[C_]['start'][e] - v) <= 0.002 for e, v in S1752_C.items())
          and abs(base['skip7000']['table_only'] - S1738_PROGRAM_CE) <= 0.005
          and abs(base['skip7000']['live'] - 3.29205) <= 1e-3 and ncov == 5419 and fits_ok)

    print(f'\n  trained C beats trained A held out ({out[C_]["best"][ho]:+.5f} vs '
          f'{out[A_]["best"][ho]:+.5f}) -> {pa}', flush=True)
    print(f'  training rescues C from negative ({out[C_]["start"][ho]:+.5f} -> '
          f'{out[C_]["best"][ho]:+.5f}) -> {pb}', flush=True)
    print(f'  the richer class gains more from a global objective '
          f'({out[C_]["train_gain"][ho]:+.5f} vs {out[A_]["train_gain"][ho]:+.5f}) -> {pc}',
          flush=True)
    print(f'  both starts reproduce §1748 and §1752 + table-only + live CE + coverage {ncov} -> '
          f'control {pd}', flush=True)

    r = {'config': {'rank': RANK, 'steps': STEPS, 'batch': BATCH, 'lr': LR, 'variants': VARIANTS,
                    'question': '§1752 found richer local features improve every site solo and '
                                'destroy the joint program, under a LOCAL objective. §1750 found a '
                                'downstream objective worth 43% for the poorest class. This asks '
                                'whether the rich class was badly STEERED or genuinely does not '
                                'compose.',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Training uses the FIT rows; skip7000 selects the '
                                 'checkpoint; skip11000 is clean of both.'},
         'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
         'variants': out,
         'reference': {'A_local_S1748': S1748_A, 'C_local_S1752': S1752_C,
                       'A_trained_S1751': S1751_A_TRAINED},
         'predictions': {'pred_a_rich_class_wins_when_trained': bool(pa),
                         'pred_b_training_rescues_C': bool(pb),
                         'pred_c_rich_class_gains_more': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
