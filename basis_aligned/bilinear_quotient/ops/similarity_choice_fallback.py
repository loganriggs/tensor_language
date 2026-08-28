# WHICH SIMILARITY? -- input embedding, or the model's own next-token distribution.
#
# §1777/§1778: sending each uncovered token to the nearest COVERED token by input-embedding cosine
# beat a global mean by 0.28-0.43 nats and made a fully standalone position-wise program the best
# all-position program in this arc. §1779: averaging more than one neighbour strictly hurts, so the
# neighbourhood is sharp.
#
# But the input embedding is only a PROXY for what the substitution needs. A table row is a site's
# output for a token, and what makes two tokens interchangeable there is that the model predicts
# similar things after them. That is directly measurable: the length-1 output distribution, which
# §1768 already showed is the model's own per-token function. This compares the two similarities
# head to head, and reports how often they even pick the same neighbour.
#
# ROLES. skip7000, skip11000, skip1200; all-position and covered CE. skip11000's embedding arm
# reproduces §1777 as a control. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats with margins per LESSON 40, read
# back per LESSON 39:
#   pred_a OUTPUT SIMILARITY WINS at every rank on every role. If FALSE the input embedding is the
#          better notion of substitutability for a table row, which would be genuinely surprising --
#          the row is an output object and the embedding is an input one -- and would say the
#          model's geometry carries the relevant structure on the input side.
#   pred_b BY A MARGIN THAT MATTERS: at rank 64 the gain is at least 0.02 nats. Scored independently
#          of pred_a, since a uniform but tiny win would not justify computing 50,257 length-1
#          forwards instead of 5,419.
#   pred_c COVERED CE IS UNTOUCHED by either similarity, on every role, within 1e-6. A wiring check.
#   pred_d CONTROLS: the embedding arm reproduces §1777's 6.03786, 6.23480, 6.54141 and 6.69203 on
#          skip11000 within 0.002, the mean arm its 6.46948, 6.64292, 6.89892 and 7.02245, and
#          coverage is exactly 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257
TABLE_RANKS = (None, 64, 8, 4)
ARMS = ('hybrid', 'standalone')
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/similarity_choice_fallback_results.json'
SIMS = ('embedding', 'output')
FAMILIES = ('contextfree',)
FALLBACKS = SIMS + ('mean',)
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt'),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1771_SKIP11000 = {'full': 1.37925, '64': 1.14673, '8': 0.80353, '4': 0.63791}
# §1762, fit-mean tables: the standalone arm lost 178% at the fidelity point and 988% at rank 8
# §1775, context-free ALL-position CE on skip11000, to be reproduced here as a control
# §1776 all-position CE on skip11000, context-free STANDALONE with the MEAN-row fallback
S1777_NEAREST = {'full': 6.03786, '64': 6.23480, '8': 6.54141, '4': 6.69203}   # skip11000
S1777_MEAN = {'full': 6.46948, '64': 6.64292, '8': 6.89892, '4': 7.02245}
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
    unc = (~seen).nonzero(as_tuple=True)[0]
    nn = {}
    # (a) similarity in the INPUT embedding, as §1777 used
    m_emb = torch.full((V,), nc, dtype=torch.long, device=DEV)
    m_emb[cov_idx] = torch.arange(nc, device=DEV)
    for s0 in range(0, unc.numel(), 4096):
        u = unc[s0:s0 + 4096]
        m_emb[u] = (En[u] @ En[cov_idx].T).argmax(-1)
    nn['embedding'] = m_emb
    del En
    torch.cuda.empty_cache()
    # (b) similarity in the model's own LENGTH-1 OUTPUT distribution. Tokens whose next-token
    # distributions agree are the ones whose table rows should substitute; the input embedding is
    # only a proxy for that. Needs a length-1 forward for EVERY token id, not just covered ones.
    P = torch.zeros(V, 50304, dtype=torch.float16, device=DEV)
    for s0 in range(0, V, 512):
        t = torch.arange(s0, min(s0 + 512, V), device=DEV).unsqueeze(1)
        P[s0:s0 + t.shape[0]] = torch.softmax(forward_logits(t)[:, 0].float(), -1).half()
    Pn = P / P.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    m_out = torch.full((V,), nc, dtype=torch.long, device=DEV)
    m_out[cov_idx] = torch.arange(nc, device=DEV)
    Pc = Pn[cov_idx]
    for s0 in range(0, unc.numel(), 1024):
        u = unc[s0:s0 + 1024]
        m_out[u] = (Pn[u].float() @ Pc.float().T).argmax(-1)
    nn['output'] = m_out
    del P, Pn, Pc
    torch.cuda.empty_cache()
    agree = float((nn['embedding'][unc] == nn['output'][unc]).float().mean())
    print(f'  built both neighbour maps for {unc.numel()} uncovered ids; they agree on '
          f'{agree:.2%} of them ({time.time() - t0:.0f}s)', flush=True)
    COV['agree'] = agree
    mean_map = COV['idmap'].clone()

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
                COV['idmap'] = mean_map if fb == 'mean' else nn[fb]
                hooks = [(st, table_hook(tr[st], seen, True)) for st in sites]
                for en in ev:
                    c1 = ce_both(ev[en], hooks)
                    row[f'{fb}_{en}'] = {'cov_ce': round(c1['cov'], 5),
                                         'all_ce': round(c1['all'], 5),
                                         'gain_over_live': round(c1['all'] - base[en]['live_all'], 5)}
            print(f'  rank {key:5s} {row["cost_M"]:8.3f}M | skip11000 ' + ' '.join(
                f'{fb} {row[f"{fb}_skip11000"]["all_ce"]:.5f}' for fb in FALLBACKS)
                + f'   [{time.time() - t0:.0f}s]', flush=True)
            if r is not None:
                del tr
                torch.cuda.empty_cache()
        del tab
        torch.cuda.empty_cache()

    keys = ['full' if r is None else str(r) for r in TABLE_RANKS]
    # ALL comparisons in ABSOLUTE nats (LESSON 35; ops/safe_ratio.py).
    roles = [e for e, _ in EVAL_SETS]
    pa = all(out[kk][f'output_{e}']['all_ce'] < out[kk][f'embedding_{e}']['all_ce']
             for kk in keys for e in roles)
    pb = (out['64']['embedding_skip11000']['all_ce']
          - out['64']['output_skip11000']['all_ce']) >= 0.02
    pc = all(abs(out[kk][f'{s}_{e}']['cov_ce'] - out[kk][f'mean_{e}']['cov_ce']) <= 1e-6
             for kk in keys for e in roles for s in SIMS)
    pd = (all(abs(out[kk]['embedding_skip11000']['all_ce'] - v) <= 0.002
              for kk, v in S1777_NEAREST.items())
          and all(abs(out[kk]['mean_skip11000']['all_ce'] - v) <= 0.002
                  for kk, v in S1777_MEAN.items()) and ncov == NCOV)

    d64 = (out['64']['embedding_skip11000']['all_ce']
           - out['64']['output_skip11000']['all_ce'])
    print(f'\n  output-distribution similarity beats input-embedding similarity everywhere -> {pa}',
          flush=True)
    print(f'  by >=0.02 at rank 64 ({d64:+.5f}) -> {pb}', flush=True)
    print(f'  covered CE untouched by either similarity -> {pc}', flush=True)
    print(f'  the embedding arm reproduces §1777 + coverage {ncov} -> control {pd}', flush=True)
    print(f'  the two maps pick the same neighbour for {COV["agree"]:.2%} of uncovered ids',
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
          'neighbour_map_agreement': COV['agree'],
          'predictions': {'pred_a_output_similarity_wins': bool(pa),
                          'pred_b_by_at_least_0p02_at_rank64': bool(pb),
                          'pred_c_change_is_confined_to_uncovered': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
