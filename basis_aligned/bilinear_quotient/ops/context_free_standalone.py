# DOES THE CONTEXT-FREE TABLE SURVIVE STANDALONE SCORING?
#
# §1762, on fit-mean tables: dropping the hybrid fallback -- letting an uncovered token take the
# site's mean row instead of running the native module -- cost **178%** of the program's all-position
# recovery at the fidelity point and **988%** at the starved one. Both went NEGATIVE. The fallback was
# load-bearing, and that caveat has sat on top of every frontier figure since.
#
# The context-free tables (§1769-§1774) are a strictly better program on covered tokens. On UNCOVERED
# tokens neither construction has a row: both fall back to a mean over covered rows. So this asks
# whether a mean over CONTEXT-FREE site outputs is a better stand-in than a mean over context
# AVERAGES -- which is not obvious in either direction and is the last unmeasured term in the
# frontier's price.
#
# Codex's §1761 narrowing applies unchanged and is repeated in the hook: both arms are post-forward
# hooks, so this is zero-native-OUTPUT and not zero-native-CALL, the native compute is not removed,
# and attention `v1` is passed through in both.
#
# ROLES. skip7000 and skip11000; covered AND all-position CE in one pass, because §1761 scored only
# covered positions where the two arms are identical by construction and could not see its own
# subject. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39:
#   pred_a CONTEXT-FREE SURVIVES BETTER: at every tested rank the standalone arm loses a smaller
#          FRACTION of its hybrid all-position recovery than §1762's fit-mean 178.1%. If FALSE, the
#          better covered-token program is no better at the uncovered quarter and the fallback is
#          exactly as load-bearing as before.
#   pred_b IT STAYS POSITIVE: every standalone arm's all-position recovery is above zero, where
#          §1762's fit-mean arms went negative at both tested points. Scored independently of pred_a,
#          since a program can lose a smaller fraction of a larger number and still end below zero.
#   pred_c THE LOSS GROWS AS THE TABLE IS STARVED: the loss fraction at rank 4 exceeds that at full
#          rank, as §1762 found for fit-mean tables (988% against 178%). If FALSE the mean row's
#          quality is independent of the rank of the block it was truncated from, which would be
#          worth knowing before anyone truncates further.
#   pred_d CONTROLS: the HYBRID covered recoveries reproduce §1770/§1771's +1.37925, +1.14673,
#          +0.80353 and +0.63791 within 0.002, and coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257
TABLE_RANKS = (None, 64, 8, 4)
ARMS = ('hybrid', 'standalone')
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/context_free_standalone_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1771_SKIP11000 = {'full': 1.37925, '64': 1.14673, '8': 0.80353, '4': 0.63791}
# §1762, fit-mean tables: the standalone arm lost 178% at the fidelity point and 988% at rank 8
S1762_FITMEAN_LOSS = {'fidelity_point': 1.781, 'efficiency_point': 9.883}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl, seen, standalone=False):
    """HYBRID (§1661): table where the token was covered at fit, LIVE module elsewhere. STANDALONE:
    the table everywhere, so an uncovered token takes the site's mean row and the module's OUTPUT is
    never used. Per Codex's §1761 narrowing this is zero-native-OUTPUT, not zero-native-CALL: both
    arms are post-forward hooks, the native compute is not removed, and attention `v1` is passed
    through unchanged in both."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        if not standalone:
            sub = torch.where(seen[STATE['idx']].unsqueeze(-1), sub, y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


@torch.no_grad()
def forward_logits(idx, hooks=()):
    hs = [mod_of(*st).register_forward_hook(h) for st, h in hooks]
    STATE['idx'] = idx
    try:
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in H:
            x, v1 = blk(x, v1, x0)
        return 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def ce_both(rows, hooks=()):
    """CE on BOTH populations in one pass. §1761 scored only COVERED positions, where the two arms
    are identical by construction, and could not see the thing it was measuring."""
    acc = {'cov': [0.0, 0], 'all': [0.0, 0]}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward_logits(idx, hooks)
        tg = bb[:, 1:].to(DEV)
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:].double()
        c = COV['seen'][idx[:, 64:]]
        acc['cov'][0] += float(e[c].sum()); acc['cov'][1] += int(c.sum())
        acc['all'][0] += float(e.sum()); acc['all'][1] += int(e.numel())
    return {k: acc[k][0] / acc[k][1] for k in acc}


@torch.no_grad()
def ce(rows, hooks=()):
    return ce_both(rows, hooks)['cov']


@torch.no_grad()
def build_tables(fit, sites, seen, toks, context_free):
    """context_free=True: each site's output on a LENGTH-1 sequence per covered token (§1769).
    context_free=False: the per-token MEAN over the fit rows -- the §1747-§1758 construction, needed
    here only to produce each role's own all-tabled baseline."""
    tables = {st: torch.zeros(V, D, device=DEV) for st in sites}
    if context_free:
        cap = {}

        def mk(st):
            def hook(mod, args, out):
                cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
                return None
            return hook
        for i in range(0, toks.numel(), 256):
            t = toks[i:i + 256].to(DEV).unsqueeze(1)
            forward_logits(t, [(st, mk(st)) for st in sites])
            for st in sites:
                tables[st][t.squeeze(1)] = cap[st]
    else:
        c = torch.zeros(V, device=DEV)
        acc = {st: torch.zeros(V, D, device=DEV) for st in sites}

        def mk2(st, first):
            def hook(mod, args, out):
                y = (out[0] if isinstance(out, tuple) else out).float().reshape(-1, D)
                tk = STATE['idx'].reshape(-1)
                acc[st].index_add_(0, tk, y)
                if first:
                    c.index_add_(0, tk, torch.ones_like(tk, dtype=torch.float32))
                return None
            return hook
        for i in range(0, fit.shape[0], 8):
            forward_logits(fit[i:i + 8, :-1].to(DEV).contiguous(),
                           [(st, mk2(st, j == 0)) for j, st in enumerate(sites)])
        sn = c > 0
        for st in sites:
            tables[st][sn] = acc[st][sn] / c[sn].unsqueeze(1)
    for st in sites:
        mu = tables[st][toks.to(DEV)].mean(0)
        tables[st][~seen] = mu
    return tables


@torch.no_grad()
def truncate(tables, toks, r):
    if r is None:
        return tables, 36 * (NCOV * D + D)
    out = {}
    idx = toks.to(DEV)
    for st, tbl in tables.items():
        blk = tbl[idx].double()
        mu = blk.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(blk - mu, full_matrices=False)
        t2 = tbl.clone()
        t2[idx] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        out[st] = t2
    return out, 36 * (r * (NCOV + D) + 2 * D)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen_cpu.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    seen = seen_cpu.to(DEV)
    COV['seen'] = seen
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'CONTEXT-FREE FRONTIER, THIRD ROLE | ranks {TABLE_RANKS} | skip1200 confirms | '
          f'SECOND-CLASS CONFIRMATION', flush=True)

    fm = build_tables(fit, sites, seen, toks, context_free=False)
    fm_hooks = [(st, table_hook(fm[st], seen)) for st in sites]
    ev, base = {}, {}
    for ename, epath in EVAL_SETS:
        e = load(epath)
        ev[ename] = e
        lv = ce_both(e)
        tb = ce_both(e, fm_hooks)
        base[ename] = {'live': lv['cov'], 'live_all': lv['all'],
                       'all_tabled_fitmean': tb['cov'], 'all_all': tb['all']}
        b = base[ename]
        print(f'  {ename}: live cov {b["live"]:.5f} all {b["live_all"]:.5f} | fit-mean all-tabled '
              f'cov {b["all_tabled_fitmean"]:.5f} all {b["all_all"]:.5f}', flush=True)
    del fm, fm_hooks
    torch.cuda.empty_cache()

    cf = build_tables(fit, sites, seen, toks, context_free=True)
    print(f'  built context-free tables ({time.time() - t0:.0f}s)', flush=True)

    out = {}
    for r in TABLE_RANKS:
        tr, cost = truncate(cf, toks, r)
        key = 'full' if r is None else str(r)
        row = {'cost_M': round(cost / 1e6, 4)}
        for arm in ARMS:
            sa = (arm == 'standalone')
            hooks = [(st, table_hook(tr[st], seen, sa)) for st in sites]
            for ename in ev:
                c1 = ce_both(ev[ename], hooks)
                row[f'{arm}_{ename}'] = {
                    'cov_ce': round(c1['cov'], 5), 'all_ce': round(c1['all'], 5),
                    'cov_rec': round(base[ename]['all_tabled_fitmean'] - c1['cov'], 5),
                    'all_rec': round(base[ename]['all_all'] - c1['all'], 5)}
        out[key] = row
        h, s = row['hybrid_skip11000'], row['standalone_skip11000']
        loss = (h['all_rec'] - s['all_rec']) / h['all_rec'] if abs(h['all_rec']) > 1e-9 else None
        row['standalone_all_loss_fraction'] = None if loss is None else round(loss, 5)
        print(f'  rank {key:5s} {row["cost_M"]:8.3f}M | hybrid all {h["all_rec"]:+.5f} '
              f'standalone all {s["all_rec"]:+.5f} | loss '
              f'{"n/a" if loss is None else f"{loss:.2%}"}   [{time.time() - t0:.0f}s]', flush=True)
        if r is not None:
            del tr
            torch.cuda.empty_cache()

    keys = ['full' if r is None else str(r) for r in TABLE_RANKS]
    lf = {k: out[k]['standalone_all_loss_fraction'] for k in keys}
    pa = all(v is not None and v < S1762_FITMEAN_LOSS['fidelity_point'] for v in lf.values())
    pb = all(out[k]['standalone_skip11000']['all_rec'] > 0.0 for k in keys)
    pc = lf['4'] > lf['full']
    pd = (all(abs(out[k]['hybrid_skip11000']['cov_rec'] - v) <= 0.002
              for k, v in S1771_SKIP11000.items()) and ncov == NCOV)

    print(f'\n  standalone loss fractions {[f"{k}:{lf[k]:.1%}" for k in keys]} all below '
          f'§1762\'s fit-mean 178.1% -> context-free survives better {pa}', flush=True)
    print(f'  every standalone arm stays positive on all-position scoring -> {pb}', flush=True)
    print(f'  the loss is larger at the starved rank 4 than at full -> {pc}', flush=True)
    print(f'  hybrid covered recoveries reproduce §1770/§1771 + coverage {ncov} -> control {pd}',
          flush=True)

    r2 = {'config': {'ranks': [str(x) for x in TABLE_RANKS],
                     'third_role': 'fineweb_n96_skip1200 -- a pinned role, half the row count, never '
                                   'scored for a program-frontier quantity. It WAS the confirmation '
                                   'role for the token-class family in S1734, a different hypothesis, '
                                   'so this is a SECOND-CLASS confirmation and not a virgin role.',
                     'baseline': "each role's own fit-mean all-tabled CE, computed in-run because "
                                 'skip1200 has no published baseline',
                     'ROLE_NOTE': 'DISCOVERY plus SECOND-CLASS CONFIRMATION.'},
          'baseline': {e: {k: round(v, 5) for k, v in base[e].items()} for e in base},
          'by_table_rank': out, 'standalone_loss_fraction': lf,
          'predictions': {'pred_a_context_free_survives_standalone': bool(pa),
                          'pred_b_standalone_stays_positive': bool(pb),
                          'pred_c_loss_larger_at_starved_rank': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
