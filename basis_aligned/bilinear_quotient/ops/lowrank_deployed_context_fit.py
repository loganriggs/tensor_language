# DEPLOYED-CONTEXT FIT -- does the mismatch explain §1745's rank inversion, or does attention simply
# have nothing a current-position map can read?
#
# §1745 fitted `table[token] + x W_r` at the six sites §1744 says a compiler must keep, and got two
# results its own header had flagged as confounded:
#   - recovery FELL with rank: 8.29% -> 7.01% -> 5.69% of the gap, held out
#   - the five attention sites averaged -5.71% of their own gap while mlp17 closed 20.71%
# W was fitted on LIVE inputs and deployed with the other thirty sites tabled, so a higher-rank map is
# more finely tuned to inputs it never sees. §1669 -- independently fitted programs installed jointly
# gave -42.99% -- is exactly that failure, and §1745 named it as the natural reading without testing.
#
# This run removes the mismatch: the other thirty sites carry their TABLES during the fit sweep, so
# each target site sees its deployment inputs and its native output is the one the `native-6` arm
# produces. One flag; everything else identical; both roles reported.
#
# The two explanations make OPPOSITE predictions here, which is why it is worth a run:
#   MISMATCH   -> refitting in context restores monotonicity and lifts the level
#   NON-LOCAL  -> attention stays unreachable regardless, because a map reading only the current
#                 position has nothing to read (§1682: the attention output write is 83.6% non-local)
# They are not exclusive, and the arms below can distinguish both, one, or neither.
#
# ROLES. Fitting uses the fit rows; both eval roles are reported. DISCOVERY ONLY -- the SITE SET came
# from §1739-§1744 using both roles, so neither is clean for "is this the right site set"; both are
# clean for "does fitting in context change what rank r buys".
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other so no arm is decided by
# another's outcome:
#   pred_a THE RANK INVERSION IS FIXED: recovery increases from r=8 to r=32 to r=128 on both roles.
#          If FALSE the mismatch was not what inverted the curve, and the cause is elsewhere -- ridge
#          scale, or a genuine ceiling on what a linear map can add over a per-token table.
#   pred_b THE DEPLOYED FIT AT LEAST DOUBLES THE LEVEL: rank 128 closes >= 2x §1745's 5.69% held out.
#          Scored independently of pred_a, since the level can rise without the curve turning over.
#   pred_c ATTENTION IS STILL UNREACHABLE: the five attention sites' mean fraction of their own gap
#          stays below mlp17's. If FALSE -- if fitting in context rescues attention -- then §1745's
#          non-locality reading was wrong, the whole problem was the mismatch, and that is the most
#          useful outcome here because it redirects the compiler.
#   pred_d CONTROLS: table-only CE reproduces 7.35114 within 0.005, the native-6 arm reproduces
#          §1741's 1.2037 and 1.2414 within 0.001, and fit coverage is 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANKS = (8, 32, 128)
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/lowrank_deployed_context_fit_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1741_NATIVE6 = {'skip7000': 1.2037, 'skip11000': 1.2414}
# §1745, LIVE-context fit: fraction of the joint gap closed, held out
S1745_LIVE = {8: 0.0829, 32: 0.0701, 128: 0.0569}
S1745_LIVE_ATTN_MEAN = -0.0571   # mean fraction of own gap, five attention sites, rank 128
S1745_LIVE_MLP17 = 0.2071
GREEDY6 = ['mlp17', 'attn16', 'attn14', 'attn11', 'attn17', 'attn13']
PRICE_M = {'mlp': 15.926, 'attn': 7.963}
COV = {}
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def name_to_site(n):
    return ('mlp', int(n[3:])) if n.startswith('mlp') else ('attn', int(n[4:]))


def table_hook(tbl, seen):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def lowrank_hook(tbl, seen, W):
    """table[token] + x W, with the same hybrid coverage rule as the plain table hook (§1661)."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        x = args[0]
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        sub = sub + (x.reshape(-1, D).to(W.dtype) @ W).reshape(y.shape).to(y.dtype)
        sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


@torch.no_grad()
def sweep(rows, hooks=(), score=None):
    hs = list(hooks)
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
    hooks = [mod_of(*st).register_forward_hook(mk(st, j == 0)) for j, st in enumerate(sites)]
    sweep(rows, hooks=hooks)
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
def fit_lowrank(rows, target_sites, tables, seen, all_sites, deployed):
    """Ridge from each site's INPUT to (module output - table output).

    `deployed=True` installs the TABLE at every non-target site during the fit sweep, so the target
    sites see the inputs they will actually see at deployment and their native outputs are the ones
    the `native-6` arm produces. `deployed=False` reproduces §1745's live-model fit. That single flag
    is the whole experiment: §1745's recovery FELL with rank and named this mismatch (§1669) as the
    explanation without testing it."""
    xtx = {st: torch.zeros(D, D, device=DEV, dtype=torch.float64) for st in target_sites}
    xtr = {st: torch.zeros(D, D, device=DEV, dtype=torch.float64) for st in target_sites}
    n = {'k': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).reshape(-1, D).double()
            x = args[0].reshape(-1, D).double()
            r = y - tables[st][STATE['idx'].reshape(-1)].double()
            xtx[st] += x.T @ x
            xtr[st] += x.T @ r
            if first:
                n['k'] += x.shape[0]
            return None
        return hook
    hooks = [mod_of(*st).register_forward_hook(mk(st, j == 0))
             for j, st in enumerate(target_sites)]
    if deployed:
        hooks += [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                  for st in all_sites if st not in target_sites]
    sweep(rows, hooks=hooks)
    assert n['k'] > 0, 'low-rank fit never fired'
    print(f'  fitted on {n["k"]} (position, site) samples', flush=True)
    W = {}
    for st in target_sites:
        A = xtx[st] + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (n['k'] / D)
        full = torch.linalg.solve(A, xtr[st])
        U, S, Vh = torch.linalg.svd(full)
        W[st] = {r: (U[:, :r] * S[:r]) @ Vh[:r] for r in RANKS}
        W[st]['full'] = full
        print(f'    {st[0]}{st[1]:<2d} singular values: top {float(S[0]):.4f}  '
              f'r8 {float(S[7]):.4f}  r32 {float(S[31]):.4f}  r128 {float(S[127]):.4f}  '
              f'tail {float(S[-1]):.2e}', flush=True)
    return W


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    tgt = [name_to_site(n) for n in GREEDY6]
    print(f'TABLE + LOW-RANK, DEPLOYED-CONTEXT FIT | ranks {RANKS} | {GREEDY6} | DISCOVERY ONLY',
          flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)
    W = fit_lowrank(fit, tgt, tables, seen, sites, deployed=True)

    out = {}
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        cl = ce(ev)
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        tbl_only = ce(ev, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                           for st in sites])
        native6 = ce(ev, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                          for st in sites if st not in tgt])
        gap = tbl_only - native6
        arms = {}
        for r in RANKS:
            c1 = ce(ev, [mod_of(*st).register_forward_hook(
                lowrank_hook(tables[st], seen, W[st][r].float()) if st in tgt
                else table_hook(tables[st], seen)) for st in sites])
            arms[r] = {'ce': round(c1, 5), 'recovered': round(tbl_only - c1, 5),
                       'frac_of_gap': round((tbl_only - c1) / gap, 5),
                       'cost_M': round(6 * 2 * r * D / 1e6, 4)}
        # per-site, at the top rank, to score pred_c
        rtop = RANKS[-1]
        per_site = {}
        for st in tgt:
            nm = f'{st[0]}{st[1]}'
            site_native = ce(ev, [mod_of(*s).register_forward_hook(table_hook(tables[s], seen))
                                  for s in sites if s != st])
            site_lr = ce(ev, [mod_of(*s).register_forward_hook(
                lowrank_hook(tables[s], seen, W[s][rtop].float()) if s == st
                else table_hook(tables[s], seen)) for s in sites])
            g = tbl_only - site_native
            per_site[nm] = {'site_gap': round(g, 5), 'lowrank_recovered': round(tbl_only - site_lr, 5),
                            'frac_of_site_gap': round((tbl_only - site_lr) / g, 5) if abs(g) > 1e-9
                            else None}
        print(f'\n  {ename}: live {cl:.5f} | table-only {tbl_only:.5f} | native-6 {native6:.5f} '
              f'| gap {gap:.4f} nats (native-6 costs '
              f'{PRICE_M["mlp"] + 5 * PRICE_M["attn"]:.3f}M)', flush=True)
        for r in RANKS:
            a = arms[r]
            print(f'    rank {r:3d}: recovers {a["recovered"]:7.4f} nats = {a["frac_of_gap"]:6.2%} '
                  f'of the gap, for {a["cost_M"]:.4f}M reals', flush=True)
        for nm, ps in per_site.items():
            fr = ps['frac_of_site_gap']
            frs = ' n/a' if fr is None else f'{fr:.2%}'
            print(f'      {nm:7s} own gap {ps["site_gap"]:7.4f}  rank{rtop} closes {frs}',
                  flush=True)
        out[ename] = {'live_ce': round(cl, 5), 'table_only_ce': round(tbl_only, 5),
                      'native6_ce': round(native6, 5), 'native6_recovered': round(gap, 5),
                      'arms': arms, 'per_site_top_rank': per_site}
        del ev
        torch.cuda.empty_cache()

    ho = out['skip11000']
    pa = all(out[e]['arms'][RANKS[i]]['recovered'] < out[e]['arms'][RANKS[i + 1]]['recovered']
             for e in out for i in range(len(RANKS) - 1))
    pb = ho['arms'][RANKS[-1]]['frac_of_gap'] >= 2.0 * S1745_LIVE[RANKS[-1]]
    att = [v['frac_of_site_gap'] for k, v in ho['per_site_top_rank'].items()
           if k.startswith('attn') and v['frac_of_site_gap'] is not None]
    attm = sum(att) / len(att) if att else float('nan')
    mlp = ho['per_site_top_rank']['mlp17']['frac_of_site_gap']
    pc = bool(att) and attm < mlp
    pd = (abs(out['skip7000']['table_only_ce'] - S1738_PROGRAM_CE) <= 0.005 and ncov == 5419
          and all(abs(out[e]['native6_recovered'] - v) <= 0.001 for e, v in S1741_NATIVE6.items()))

    print(f'\n  recovery INCREASES with rank on both roles -> inversion fixed {pa}', flush=True)
    print(f'  rank {RANKS[-1]} held out {ho["arms"][RANKS[-1]]["frac_of_gap"]:.2%} vs §1745 live '
          f'{S1745_LIVE[RANKS[-1]]:.2%} (bar 2x) -> deployed fit helps {pb}', flush=True)
    print(f'  attention mean {attm:.2%} (§1745 live {S1745_LIVE_ATTN_MEAN:.2%}) still below mlp17 '
          f'{mlp:.2%} (§1745 live {S1745_LIVE_MLP17:.2%}) -> attention still unreachable {pc}',
          flush=True)
    print(f'  table-only CE + §1741 native-6 + coverage {ncov} -> control {pd}', flush=True)

    r = {'config': {'sites': GREEDY6, 'ranks': list(RANKS), 'ridge': RIDGE,
                    'program': 'table[token] + x W_r, hybrid coverage rule (§1661); W ridge-fitted '
                               'from the site input to (native output - table output) on the LIVE '
                               'model, then SVD-truncated',
                    'FIT_CONTEXT': 'DEPLOYED: the other thirty sites carry their tables during '
                                   'the fit sweep, so each target site sees its deployment inputs '
                                   'and its native output is the one the native-6 arm produces. '
                                   'This is the mismatch §1745 declared as its caveat, removed.',
                    'costs_M_reals': {'rank_r_per_site': '2*r*1152', 'native_attn': PRICE_M['attn'],
                                      'native_mlp': PRICE_M['mlp']},
                    'ROLE_NOTE': 'DISCOVERY ONLY. The SITE SET came from §1739-§1744 using both '
                                 'roles, so neither role is clean for "is this the right site set". '
                                 'Both are clean for "what does rank r buy at this site set".'},
         'results': out,
         'predictions': {'pred_a_rank_inversion_fixed': bool(pa),
                         'pred_b_deployed_fit_at_least_doubles': bool(pb),
                         'pred_c_attention_still_unreachable': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
