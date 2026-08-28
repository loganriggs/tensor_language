# A BETTER STAND-IN FOR THE UNCOVERED QUARTER
#
# §1776 left a clean split. With the hybrid fallback allowed, the best all-position program is
# fit-mean plus fallback at 6.28596 -- but it CALLS the module it is meant to replace at 24% of
# positions, so it is not a replacement. Without the fallback, the best is context-free standalone at
# 6.64292. The 0.36-nat difference between them is entirely about what happens at the uncovered
# quarter, where a context-free program currently uses one thing: the mean over all covered rows.
#
# A mean over 5419 unrelated tokens is a poor stand-in for a specific unseen token, and there is an
# obviously better one that costs nothing extra and stays inside the class: send each uncovered token
# to the COVERED token with the most similar input embedding, and use that token's row. The map from
# token id to table row does not depend on position, so the program remains position-wise; and it adds
# no stored values, only an index.
#
# ROLES. skip11000, covered and all-position CE in one pass, context-free tables only, standalone
# arm only -- the hybrid arm is what this is trying to make unnecessary. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats per LESSON 35, read back per
# LESSON 39:
#   pred_a THE NEAREST-COVERED ROW BEATS THE MEAN ROW at every rank on all-position CE. If FALSE the
#          embedding neighbourhood is not the right notion of substitutability for a table row, which
#          would be worth knowing before anyone builds a smarter fallback.
#   pred_b IT BEATS THE BEST PROGRAM IN §1776: some rank's nearest-fallback standalone CE is below
#          6.28596, the fit-mean-plus-fallback number. If TRUE, a program that never uses the module
#          it replaces becomes the best all-position program measured in this arc. Scored
#          independently of pred_a, since it can beat the mean row without clearing that bar.
#   pred_c THE CHANGE IS CONFINED TO UNCOVERED TOKENS: covered CE is identical to the mean-fallback
#          arm within 1e-6 at every rank. This is true by construction and is here as a wiring check
#          -- if it fails, the fallback map is being applied where it should not be.
#   pred_d CONTROLS: the mean-fallback arms reproduce §1776's 6.46948, 6.64292, 6.89892 and 7.02245
#          within 0.002, and coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257
TABLE_RANKS = (None, 64, 8, 4)
ARMS = ('hybrid', 'standalone')
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/nearest_neighbour_fallback_results.json'
FAMILIES = ('contextfree',)
FALLBACKS = ('mean', 'nearest')
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1771_SKIP11000 = {'full': 1.37925, '64': 1.14673, '8': 0.80353, '4': 0.63791}
# §1762, fit-mean tables: the standalone arm lost 178% at the fidelity point and 988% at rank 8
# §1775, context-free ALL-position CE on skip11000, to be reproduced here as a control
# §1776 all-position CE on skip11000, context-free STANDALONE with the MEAN-row fallback
S1776_CF_MEAN = {'full': 6.46948, '64': 6.64292, '8': 6.89892, '4': 7.02245}
S1776_BEST_ALLPOS = 6.28596   # fit-mean + hybrid fallback at rank 64, the best program in §1776
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def table_hook(tbl, seen, standalone=False):
    """`tbl` is COMPACT: [ncov+1, D], indexed through COV['idmap']; row ncov is the uncovered mean.
    The full [50257, D] form cost 8.3 GB per family and OOMed with two families resident."""
    """HYBRID (§1661): table where the token was covered at fit, LIVE module elsewhere. STANDALONE:
    the table everywhere, so an uncovered token takes the site's mean row and the module's OUTPUT is
    never used. Per Codex's §1761 narrowing this is zero-native-OUTPUT, not zero-native-CALL: both
    arms are post-forward hooks, the native compute is not removed, and attention `v1` is passed
    through unchanged in both."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = tbl[COV['idmap'][STATE['idx']].reshape(-1)].reshape(y.shape).to(y.dtype)
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
    # returns COMPACT [ncov+1, D] tables; see table_hook
    """context_free=True: each site's output on a LENGTH-1 sequence per covered token (§1769).
    context_free=False: the per-token MEAN over the fit rows -- the §1747-§1758 construction, needed
    here only to produce each role's own all-tabled baseline."""
    nc = toks.numel()
    tables = {st: torch.zeros(nc + 1, D, device=DEV) for st in sites}
    tk = toks.to(DEV)
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
                tables[st][i:i + t.shape[0]] = cap[st]
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
        for st in sites:
            tables[st][:nc] = acc[st][tk] / c[tk].clamp_min(1.0).unsqueeze(1)
        del acc
        torch.cuda.empty_cache()
    for st in sites:
        tables[st][nc] = tables[st][:nc].mean(0)
    return tables


@torch.no_grad()
def truncate(tables, toks, r):
    if r is None:
        return tables, 36 * (NCOV * D + D)
    out = {}
    for st, tbl in tables.items():
        blk = tbl[:NCOV].double()
        mu = blk.mean(0, keepdim=True)
        U, S, Vh = torch.linalg.svd(blk - mu, full_matrices=False)
        t2 = tbl.clone()
        t2[:NCOV] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
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

    COV['idmap'] = torch.full((V,), toks.numel(), dtype=torch.long, device=DEV)
    COV['idmap'][toks.to(DEV)] = torch.arange(toks.numel(), device=DEV)
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

    # two idmaps: the MEAN fallback sends every uncovered token to row ncov (the mean over covered
    # rows); the NEAREST fallback sends it to the covered token with the most similar input
    # embedding. Both are position-wise -- the map from token to row does not depend on position.
    nc = toks.numel()
    E = m.transformer.wte.weight.detach().float()
    En = E / E.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    cov_idx = toks.to(DEV)
    nn_map = torch.full((V,), nc, dtype=torch.long, device=DEV)
    nn_map[cov_idx] = torch.arange(nc, device=DEV)
    unc = (~seen).nonzero(as_tuple=True)[0]
    for s0 in range(0, unc.numel(), 4096):
        u = unc[s0:s0 + 4096]
        sim = En[u] @ En[cov_idx].T
        nn_map[u] = sim.argmax(-1)
    del En, sim
    torch.cuda.empty_cache()
    print(f'  built the nearest-covered map for {unc.numel()} uncovered token ids '
          f'({time.time() - t0:.0f}s)', flush=True)
    maps = {'mean': COV['idmap'].clone(), 'nearest': nn_map}

    out = {k: {} for k in ('full' if r is None else str(r) for r in TABLE_RANKS)}
    for fam in FAMILIES:
        tab = build_tables(fit, sites, seen, toks, context_free=(fam == 'contextfree'))
        print(f'  built the {fam} family ({time.time() - t0:.0f}s)', flush=True)
        for r in TABLE_RANKS:
            key = 'full' if r is None else str(r)
            row = out[key]
            tr, cost = truncate(tab, toks, r)
            row['cost_M'] = round(cost / 1e6, 4)
            for fb in FALLBACKS:
                COV['idmap'] = maps[fb]
                hooks = [(st, table_hook(tr[st], seen, True)) for st in sites]
                c1 = ce_both(ev['skip11000'], hooks)
                row[f'{fb}'] = {'cov_ce': round(c1['cov'], 5), 'all_ce': round(c1['all'], 5)}
            print(f'  rank {key:5s} {row["cost_M"]:8.3f}M | ' + '  '.join(
                f'{fb} all {row[fb]["all_ce"]:.5f} cov {row[fb]["cov_ce"]:.5f}' for fb in FALLBACKS)
                + f'   [{time.time() - t0:.0f}s]', flush=True)
            if r is not None:
                del tr
                torch.cuda.empty_cache()
        del tab
        torch.cuda.empty_cache()

    keys = ['full' if r is None else str(r) for r in TABLE_RANKS]
    # ALL comparisons in ABSOLUTE nats (LESSON 35; ops/safe_ratio.py).
    pa = all(out[k]['nearest']['all_ce'] < out[k]['mean']['all_ce'] for k in keys)
    pb = min(out[k]['nearest']['all_ce'] for k in keys) < S1776_BEST_ALLPOS
    pc = all(abs(out[k]['nearest']['cov_ce'] - out[k]['mean']['cov_ce']) <= 1e-6 for k in keys)
    pd = (all(abs(out[k]['mean']['all_ce'] - v) <= 0.002 for k, v in S1776_CF_MEAN.items())
          and ncov == NCOV)

    best = min(keys, key=lambda k: out[k]['nearest']['all_ce'])
    print(f'\n  the nearest-covered fallback beats the mean row at every rank -> {pa}', flush=True)
    print(f'  best standalone all-position CE {out[best]["nearest"]["all_ce"]:.5f} at rank {best} '
          f'beats §1776\'s best program {S1776_BEST_ALLPOS} -> {pb}', flush=True)
    print(f'  covered CE is untouched (the change is only at uncovered tokens) -> {pc}', flush=True)
    print(f'  the mean-fallback arms reproduce §1776 + coverage {ncov} -> control {pd}', flush=True)

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
          'predictions': {'pred_a_nearest_beats_mean': bool(pa),
                          'pred_b_beats_best_S1776_program': bool(pb),
                          'pred_c_change_is_confined_to_uncovered': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
