# LINEAR-CORRECTABILITY MAP -- is mlp17's 54x compression a property of that site, of late MLPs,
# or of MLPs generally?
#
# §1746: fitted in the DEPLOYED context, `table[token] + x W_128` closes 92.06% of mlp17's own gap for
# 0.295M reals against a 15.926M native module -- a 54x compression at 92% fidelity -- while the five
# attention sites in the same run averaged -5.55%. That was six sites, chosen by the allocation work.
# Nobody has asked the other thirty.
#
# This measures every site: one fit sweep in the ALL-TABLED context (each hook records the module's
# input and native output AND installs the table, so every site sees its deployment inputs), then the
# joint all-36 program at three ranks and the per-site fraction of its own gap closed.
#
# If the MLP band is broadly correctable this is most of a compiler: 36 tables plus 36 rank-8 maps
# costs 0.664M reals against 430.00M for the native modules.
#
# ROLES. Fitting uses the fit rows; both eval roles are reported. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, checked against each other:
#   pred_a THE JOINT PROGRAM IS SUBSTANTIAL: all-36 table+rank8 recovers at least 1.0 nat of the
#          4.26-nat table-program stake held out. If FALSE the corrections do not compose -- each may
#          help alone while the joint install goes off-distribution, which is §1669 and LESSONS 28
#          reappearing one level up.
#   pred_b MLPs ARE MORE CORRECTABLE THAN ATTENTION: the median fraction of own gap closed is higher
#          across the 18 MLP sites than the 18 attention sites. If FALSE, §1746's split was about
#          those six sites and not about module kind.
#   pred_c LATE MLPs LEAD: at least two of mlp15, mlp16, mlp17 are in the top five by fraction closed.
#          If FALSE, mlp17's result is not a late-band property and the correctable sites are
#          somewhere else -- which would be more informative, since the late band is where the
#          allocation work already pointed.
#   pred_d CONTROLS: the table-only CE reproduces 7.35114 within 0.005, fit coverage is 5419 of
#          50257, and the six site gaps §1746 published reproduce within 0.001 -- the same quantity
#          (table-only minus that-site-native) computed by a third script.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
RANKS = (8, 32, 128)
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/lowrank_all_sites_map_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1738_PROGRAM_CE = 7.35114
S1741_NATIVE6 = {'skip7000': 1.2037, 'skip11000': 1.2414}
# §1745, LIVE-context fit: fraction of the joint gap closed, held out
# EXACT control: each site's own gap, table-only CE minus the CE with only that site native.
# Same quantity, computed by two earlier scripts, so a mismatch means something moved.
S1746_SITE_GAPS = {'skip11000': {'mlp17': 0.5141, 'attn16': 0.3105, 'attn14': 0.2426,
                                 'attn11': 0.0575, 'attn17': 0.2056, 'attn13': 0.1828}}
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
def fit_lowrank_all(rows, all_sites, tables, seen):
    """One sweep, every site both CAPTURED and TABLED.

    A forward hook receives the module's actual output before returning a substitute, so a single
    hook can record (input, native output) AND install the table. Every site therefore sees the
    fully-tabled context it will be deployed into, and its recorded native output is what it would
    produce there -- which is exactly the `site native, everything else tabled` arm the gaps are
    measured against. §1746 showed this fit context is worth 6.7x over a live-model fit.
    """
    xtx = {st: torch.zeros(D, D, device=DEV, dtype=torch.float64) for st in all_sites}
    xtr = {st: torch.zeros(D, D, device=DEV, dtype=torch.float64) for st in all_sites}
    n = {'k': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = out[0] if isinstance(out, tuple) else out
            tok = STATE['idx'].reshape(-1)
            yf = y.reshape(-1, D).double()
            xf = args[0].reshape(-1, D).double()
            tb = tables[st][tok].double()
            xtx[st] += xf.T @ xf
            xtr[st] += xf.T @ (yf - tb)
            if first:
                n['k'] += xf.shape[0]
            sub = tables[st][tok].reshape(y.shape).to(y.dtype)
            sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
            return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
        return hook
    sweep(rows, hooks=[mod_of(*st).register_forward_hook(mk(st, j == 0))
                       for j, st in enumerate(all_sites)])
    assert n['k'] > 0, 'low-rank fit never fired'
    print(f'  fitted on {n["k"]} positions per site, all {len(all_sites)} sites in one sweep',
          flush=True)
    W = {}
    for st in all_sites:
        A = xtx[st] + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (n['k'] / D)
        full = torch.linalg.solve(A, xtr[st])
        U, S, Vh = torch.linalg.svd(full)
        W[st] = {r: ((U[:, :r] * S[:r]) @ Vh[:r]).float() for r in RANKS}
    return W


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'LINEAR-CORRECTABILITY MAP | all 36 sites | ranks {RANKS} | deployed (all-tabled) fit '
          f'context | DISCOVERY ONLY', flush=True)

    COV['seen'] = torch.zeros(50257, dtype=torch.bool, device=DEV)
    tables, seen = fit_tables(fit, sites)
    COV['seen'] = seen
    ncov = int(seen.sum())
    print(f'  fit coverage {ncov} of 50257 token ids', flush=True)
    W = fit_lowrank_all(fit, sites, tables, seen)

    out = {}
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        cl = ce(ev)
        assert abs(cl - ce_ref) <= 1e-2, f'{ename} live CE {cl:.5f} != {ce_ref}'
        tbl_only = ce(ev, [mod_of(*st).register_forward_hook(table_hook(tables[st], seen))
                           for st in sites])
        stake = tbl_only - cl
        joint = {}
        for r in RANKS:
            c1 = ce(ev, [mod_of(*st).register_forward_hook(
                lowrank_hook(tables[st], seen, W[st][r])) for st in sites])
            joint[r] = {'ce': round(c1, 5), 'recovered': round(tbl_only - c1, 5),
                        'frac_of_stake': round((tbl_only - c1) / stake, 5),
                        'cost_M': round(36 * 2 * r * D / 1e6, 4)}
        rtop = RANKS[-1]
        per_site = {}
        for st in sites:
            nm = f'{st[0]}{st[1]}'
            nat = ce(ev, [mod_of(*s).register_forward_hook(table_hook(tables[s], seen))
                          for s in sites if s != st])
            lr = ce(ev, [mod_of(*s).register_forward_hook(
                lowrank_hook(tables[s], seen, W[s][rtop]) if s == st
                else table_hook(tables[s], seen)) for s in sites])
            g = tbl_only - nat
            per_site[nm] = {'site_gap': round(g, 5), 'recovered': round(tbl_only - lr, 5),
                            'frac_of_site_gap': round((tbl_only - lr) / g, 5) if abs(g) > 1e-6
                            else None, 'kind': st[0]}
        print(f'\n  {ename}: live {cl:.5f} | table-only {tbl_only:.5f} | stake {stake:.4f} nats',
              flush=True)
        for r in RANKS:
            a = joint[r]
            print(f'    ALL-36 rank {r:3d}: recovers {a["recovered"]:7.4f} = '
                  f'{a["frac_of_stake"]:6.2%} of the stake, for {a["cost_M"]:.3f}M reals',
                  flush=True)
        ranked = sorted([n for n in per_site if per_site[n]['frac_of_site_gap'] is not None],
                        key=lambda n: -per_site[n]['frac_of_site_gap'])
        print(f'    most linearly correctable (fraction of own gap, rank {rtop}):', flush=True)
        for nm in ranked[:6]:
            p = per_site[nm]
            print(f'      {nm:7s} gap {p["site_gap"]:7.4f}  closes {p["frac_of_site_gap"]:7.2%}',
                  flush=True)
        print(f'    least:', flush=True)
        for nm in ranked[-4:]:
            p = per_site[nm]
            print(f'      {nm:7s} gap {p["site_gap"]:7.4f}  closes {p["frac_of_site_gap"]:7.2%}',
                  flush=True)
        mm = sorted(per_site[n]['frac_of_site_gap'] for n in per_site
                    if per_site[n]['kind'] == 'mlp' and per_site[n]['frac_of_site_gap'] is not None)
        aa = sorted(per_site[n]['frac_of_site_gap'] for n in per_site
                    if per_site[n]['kind'] == 'attn' and per_site[n]['frac_of_site_gap'] is not None)
        med = {'mlp': 0.5 * (mm[len(mm) // 2 - 1] + mm[len(mm) // 2]) if mm else float('nan'),
               'attn': 0.5 * (aa[len(aa) // 2 - 1] + aa[len(aa) // 2]) if aa else float('nan')}
        print(f'    median fraction closed: MLP {med["mlp"]:7.2%}   attention {med["attn"]:7.2%}',
              flush=True)
        out[ename] = {'live_ce': round(cl, 5), 'table_only_ce': round(tbl_only, 5),
                      'stake_nats': round(stake, 5), 'joint': joint, 'per_site': per_site,
                      'median_frac': {k: round(v, 5) for k, v in med.items()}, 'ranked': ranked}
        del ev
        torch.cuda.empty_cache()

    ho = out['skip11000']
    pa = ho['joint'][RANKS[0]]['recovered'] >= 1.0
    pb = ho['median_frac']['mlp'] > ho['median_frac']['attn']
    pc = len({'mlp15', 'mlp16', 'mlp17'} & set(ho['ranked'][:5])) >= 2
    gaps_ok = all(abs(ho['per_site'][n]['site_gap'] - v) <= 0.001
                  for n, v in S1746_SITE_GAPS['skip11000'].items())
    pd = (abs(out['skip7000']['table_only_ce'] - S1738_PROGRAM_CE) <= 0.005 and ncov == 5419
          and gaps_ok)

    print(f'\n  all-36 rank {RANKS[0]} recovers >=1.0 nat held out '
          f'({ho["joint"][RANKS[0]]["recovered"]:.4f}) -> {pa}', flush=True)
    print(f'  median fraction closed higher for MLPs ({ho["median_frac"]["mlp"]:.2%}) than '
          f'attention ({ho["median_frac"]["attn"]:.2%}) -> {pb}', flush=True)
    print(f'  >=2 of mlp15/16/17 in the top five -> late MLPs lead {pc}', flush=True)
    print(f'  table-only CE + six §1746 site gaps + coverage {ncov} -> control {pd}', flush=True)

    r = {'config': {'sites': 'all 36', 'ranks': list(RANKS), 'ridge': RIDGE,
                    'program': 'table[token] + x W_r at every site, hybrid coverage rule (§1661)',
                    'FIT_CONTEXT': 'ALL-TABLED. One sweep: each hook records (input, native output) '
                                   'and installs the table, so every site sees the context it is '
                                   'deployed into and its recorded target is what it would produce '
                                   'there. §1746 measured this context as worth 6.7x.',
                    'ROLE_NOTE': 'DISCOVERY ONLY. Both eval roles reported; neither is clean for '
                                 'site-set questions after §1739-§1746.'},
         'results': out,
         'predictions': {'pred_a_all36_rank8_recovers_a_nat': bool(pa),
                         'pred_b_mlps_more_correctable': bool(pb),
                         'pred_c_late_mlps_lead': bool(pc),
                         'pred_d_controls': bool(pd)},
         'runtime_s': round(time.time() - t0, 1)}
    json.dump(r, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
