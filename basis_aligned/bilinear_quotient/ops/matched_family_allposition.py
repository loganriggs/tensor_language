# THE MATCHED ALL-POSITION COMPARISON §1775 SAID IT DID NOT CONTAIN
#
# §1775 measured the context-free program on all positions and found it 0.12 nats WORSE than the
# fit-mean all-tabled baseline -- but that baseline is HYBRID, running live modules at the uncovered
# quarter, while the context-free arm being compared was STANDALONE. Not apples to apples, and I said
# so and scoped the frontier's dominance claim to covered positions until this run exists.
#
# This is that run: both table families, both fallback rules, one all-position number each. Four
# programs per rank, every comparison in ABSOLUTE nats -- §1775's "loss fraction" had a negative
# denominator and meant nothing, which is LESSON 35 and which ops/safe_ratio.py was written for at
# §1760 and then not used.
#
# ROLES. skip11000 only, covered and all-position CE in one pass. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, read back per LESSON 39:
#   pred_a MATCHED, CONTEXT-FREE STILL WINS: at every rank the context-free STANDALONE program has
#          lower all-position CE than the fit-mean STANDALONE program. If FALSE, the context-free
#          advantage is a covered-position artifact and §1770's frontier should be restated as such
#          rather than merely scoped.
#   pred_b THE FALLBACK HURTS CONTEXT-FREE at every rank (standalone below hybrid), as §1775 found at
#          two ranks. If FALSE the reversal is rank-specific.
#   pred_c THE FALLBACK HELPS FIT-MEAN at every rank (standalone ABOVE hybrid), as §1762 found. If
#          FALSE, §1762's "load-bearing" conclusion does not generalise across ranks either and the
#          difference between the families is smaller than §1775 claims.
#   pred_d CONTROLS: the context-free arms reproduce §1775's all-position CEs -- 7.70737 / 6.46948 at
#          full rank and 8.22337 / 7.02244 at rank 4 -- within 0.002, and coverage is 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257
TABLE_RANKS = (None, 64, 8, 4)
ARMS = ('hybrid', 'standalone')
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/matched_family_allposition_results.json'
FAMILIES = ('fitmean', 'contextfree')
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1771_SKIP11000 = {'full': 1.37925, '64': 1.14673, '8': 0.80353, '4': 0.63791}
# §1762, fit-mean tables: the standalone arm lost 178% at the fidelity point and 988% at rank 8
# §1775, context-free ALL-position CE on skip11000, to be reproduced here as a control
S1775_CF = {'full': {'hybrid': 7.70737, 'standalone': 6.46948},
            '4': {'hybrid': 8.22337, 'standalone': 7.02244}}
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

    fams = {'fitmean': build_tables(fit, sites, seen, toks, context_free=False),
            'contextfree': build_tables(fit, sites, seen, toks, context_free=True)}
    print(f'  built both table families ({time.time() - t0:.0f}s)', flush=True)

    out = {}
    for r in TABLE_RANKS:
        key = 'full' if r is None else str(r)
        row = {}
        for fam in FAMILIES:
            tr, cost = truncate(fams[fam], toks, r)
            row['cost_M'] = round(cost / 1e6, 4)
            for arm in ARMS:
                sa = (arm == 'standalone')
                hooks = [(st, table_hook(tr[st], seen, sa)) for st in sites]
                c1 = ce_both(ev['skip11000'], hooks)
                row[f'{fam}_{arm}'] = {'cov_ce': round(c1['cov'], 5),
                                       'all_ce': round(c1['all'], 5)}
            if r is not None:
                del tr
                torch.cuda.empty_cache()
        out[key] = row
        print(f'  rank {key:5s} {row["cost_M"]:8.3f}M | all-position CE  ' + '  '.join(
            f'{f}/{a} {row[f"{f}_{a}"]["all_ce"]:.5f}' for f in FAMILIES for a in ARMS)
            + f'   [{time.time() - t0:.0f}s]', flush=True)

    keys = ['full' if r is None else str(r) for r in TABLE_RANKS]
    # ALL comparisons in ABSOLUTE nats. §1775's ratio had a negative denominator and meant nothing;
    # ops/safe_ratio.py exists for exactly that and a difference needs no guard at all.
    pa = all(out[k]['contextfree_standalone']['all_ce']
             < out[k]['fitmean_standalone']['all_ce'] for k in keys)
    pb = all(out[k]['contextfree_standalone']['all_ce']
             < out[k]['contextfree_hybrid']['all_ce'] for k in keys)
    pc = all(out[k]['fitmean_standalone']['all_ce']
             > out[k]['fitmean_hybrid']['all_ce'] for k in keys)
    pd = (all(abs(out[k][f'contextfree_{a}']['all_ce'] - v) <= 0.002
              for k, kv in S1775_CF.items() for a, v in kv.items()) and ncov == NCOV)

    print(f'\n  matched STANDALONE: context-free beats fit-mean on all positions at every rank '
          f'-> {pa}', flush=True)
    print(f'  within the context-free family the fallback HURTS at every rank -> {pb}', flush=True)
    print(f'  within the fit-mean family the fallback HELPS at every rank -> {pc}', flush=True)
    print(f'  §1775 context-free all-position CEs reproduce + coverage {ncov} -> control {pd}',
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
          'by_table_rank': out,
          'predictions': {'pred_a_contextfree_wins_matched_standalone': bool(pa),
                          'pred_b_fallback_hurts_contextfree': bool(pb),
                          'pred_c_fallback_helps_fitmean': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
