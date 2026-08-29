# THE MATCHED-MAP FRONTIER -- test §1880's law at new ranks, and fill the 65.95M -> 230.09M gap.
#
# §1880 established that a fallback map of rank above the TABLE rank buys exactly nothing: the map is
# fitted against the truncated table rows (§1785), so at table rank r the targets span a rank-r subspace
# and any map of rank >= r is already exact inside it. Measured gain of map 512 over map 64 was +0.04465
# / +0.04832 / +0.04070 at full table rank, +0.03582 / +0.03968 / +0.03558 at 256, and EXACTLY 0.00000
# at table ranks 16, 8 and 4. The rule it implies is `map_rank = min(table_rank, 512)`.
#
# That rule was inferred from a two-column sweep at map ranks 64 and 512. It has never been tested AT a
# matched rank, and it makes a sharp, immediately cashable prediction: the frontier point m512_256
# (103.109M) should be replaceable by m256_256 (81.875M) at IDENTICAL CE -- a 21.234M saving for nothing.
# If that fails, §1880's law is an artifact of the two ranks it was measured at.
#
# The same run fills the largest hole in the frontier. §1880's table ladder jumps 256 -> full, which is
# 65.950M -> 230.087M: a 3.5x cost step for 0.045 nats, with nothing measured in between. Table ranks
# 384, 512 and 768 cost 90.920M, 121.200M and 181.758M and are all cheaper than the full table's
# 224.778M (a rank-r table beats the raw table below r = 950), so they are live candidates.
#
# ROLES. skip7000, skip11000, skip1200; all-position CE, covered as a wiring check. DISCOVERY ONLY.
# Rung 2 on the law (a matched-rank confirmation of §1880) and rung 3 on the gap.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats per LESSON 40, read back per LESSON 39.
# ANCHORS name the build AND the map rank they came from, per LESSONS 53's addendum and §1875.
#   pred_a THE LAW HOLDS AT A MATCHED RANK: at table rank 256 the map-256 build matches the map-512
#          build within 1e-5 all-position CE on all three roles, and likewise at table rank 384. This is
#          §1880's law tested where it bites rather than inferred from its endpoints. If TRUE the
#          deployable frontier point at table rank 256 costs 81.875M, not 103.109M, for the same number.
#          If FALSE the law is an artifact of the 64/512 pair and §1880 needs restating.
#   pred_b AND AN INTERMEDIATE TABLE RANK LANDS ON THE FRONTIER: at least one of table rank 384, 512 or
#          768 is Pareto-nondominated on all three roles. The 256 -> full jump is 3.5x in cost for 0.045
#          nats, so something in between should be worth standing on. If FALSE the table-rank curve is
#          convex through that whole span and the two endpoints really are the only sensible builds --
#          which would be a cleaner recommendation than the one I expect.
#   pred_c AND THE COVERED ARM IS BLIND TO ALL OF IT: covered CE identical across every build at matched
#          table rank, within 1e-6. TWELFTH KNOWN-ANSWER check; the previous eleven each returned
#          exactly 0.00e+00.
#   pred_d CONTROLS: the table-256 map-512 and full-table map-512 builds reproduce §1880's PUBLISHED
#          6.02422 / 5.99343 / 6.00680 and 5.96702 / 5.93645 / 5.96095 within 0.002, and coverage is
#          exactly 5419 of 50257. Costs are recomputed from §1754's formula, not copied.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None, 768, 512, 384, 256)
MAP_RANK = 64
# per-table-rank map ranks: the MATCHED rank (min(r,512)) and 512, which coincide above 512.
SWEEP = {'full': (512,), '768': (512,), '512': (512,), '384': (384, 512), '256': (256, 512)}
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/matched_map_frontier_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1783 = {'skip7000': {'all': 6.01897, 'cov': 6.03465},
         'skip11000': {'all': 6.00091, 'cov': 5.97900},
         'skip1200': {'all': 6.00733, 'cov': 5.96423}}
# §1784's FULL-rank learned arm was already rank-consistent, so refitting per rank must not move it
S1880 = {   # §1880 PUBLISHED, map rank 512
    'skip7000':  {'256': 6.02422, 'full': 5.96702},
    'skip11000': {'256': 5.99343, 'full': 5.93645},
    'skip1200':  {'256': 6.00680, 'full': 5.96095},
}
S1858_MAP64 = {
    'skip7000':  {'full': 6.01167, '256': 6.06004, '64': 6.17330, '16': 6.35916, '8': 6.47177, '4': 6.62422},
    'skip11000': {'full': 5.98477, '256': 6.03311, '64': 6.15261, '16': 6.35149, '8': 6.47693, '4': 6.63689},
    'skip1200':  {'full': 6.00165, '256': 6.04238, '64': 6.14463, '16': 6.32237, '8': 6.43090, '4': 6.59044},
}
S1786_R64 = {'skip7000': {'neighbour': 6.19187, 'settled': 6.17330},
             'skip11000': {'neighbour': 6.18267, 'settled': 6.15261},
             'skip1200': {'neighbour': 6.15065, 'settled': 6.14463}}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def row_hook(full_rows):
    """`full_rows` is [V, D]: every token id's site row, already resolved by whichever fallback the
    arm uses. Standalone -- no native output is ever consulted."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = full_rows[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
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
    tk = toks.to(DEV)
    unc = (~seen).nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'MATCHED-MAP FRONTIER | table ranks {RANKS} | map rank '
          f'{sorted({{r for v in SWEEP.values() for r in v}})} | DISCOVERY ONLY', flush=True)

    # the settled output-NN map (§1780/§1781), for the baseline arm
    lp = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lp[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pcn = torch.softmax(lp, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lp
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(ncov, device=DEV)
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
    del pcn
    torch.cuda.empty_cache()

    # the 36 context-free site tables on the covered tokens
    tables = {st: torch.zeros(ncov, D, device=DEV) for st in sites}
    cap = {}

    def mk(st):
        def hook(mod, args, out):
            cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
            return None
        return hook
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        forward_logits(t, [(st, mk(st)) for st in sites])
        for st in sites:
            tables[st][i:i + t.shape[0]] = cap[st]
    print(f'  built the output-NN map and 36 tables ({time.time() - t0:.0f}s)', flush=True)

    # ridge fit: embedding -> site row, on the COVERED tokens only, then rank-truncated
    Ecov = m.transformer.wte.weight.detach()[tk].float().double()
    A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (ncov / D)
    Eunc = m.transformer.wte.weight.detach()[unc].float().double()
    def fit_maps(tbl_c, mr):
        """Fit the embedding->row map against THESE rows. §1784 fitted once against the FULL-rank
        tables and reused the maps at rank 64, so its rank-64 arm had truncated covered rows and
        untruncated predicted ones -- not a coherent program. Refitting inside each basis is the
        repair."""
        mp = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tbl_c[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mp[st] = ((U[:, :mr] * S[:mr]) @ Vh[:mr])
        return mp

    def build_full(tbl_c, mode, maps):
        """[V, D] rows: covered tokens keep their exact row; uncovered get the neighbour's row or the
        learned prediction from their own embedding."""
        out = {}
        for st in sites:
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tbl_c[st]
            if mode == 'neighbour':
                fr[unc] = tbl_c[st][nnrow[unc]]
            else:
                fr[unc] = (Eunc @ maps[st]).float()
            out[st] = fr
        return out

    def truncate(r):
        if r is None:
            return tables, 36 * (NCOV * D + D)
        o = {}
        for st, tbl in tables.items():
            b = tbl.double()
            mu = b.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
            o[st] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        return o, 36 * (r * (NCOV + D) + 2 * D)

    res = {}
    for ename, epath, ref in EVAL_SETS:
        ev = load(epath)
        live = ce_both(ev)
        if ref is not None:
            assert abs(live['cov'] - ref) <= 1e-3, f'{ename} live cov {live["cov"]:.5f} != {ref}'
        row = {'live': {k: round(v, 5) for k, v in live.items()}}
        for r in RANKS:
            tc, cost = truncate(r)
            key = 'full' if r is None else str(r)
            for mr in SWEEP[key]:
                mc = 36 * mr * 2 * D
                fr = build_full(tc, 'learned', fit_maps(tc, mr))
                row[f'm{mr}_{key}'] = {**{k: round(v, 5) for k, v in ce_both(
                    ev, [(st, row_hook(fr[st])) for st in sites]).items()},
                    'cost_M': round((cost + mc) / 1e6, 4)}
                fr = None
                torch.cuda.empty_cache()
            tc = None
            torch.cuda.empty_cache()
        res[ename] = row
        print(f'\n  {ename}: live cov {live["cov"]:.5f} all {live["all"]:.5f}', flush=True)
        for r in RANKS:
            key = 'full' if r is None else str(r)
            print(f'    table {key:5s}  ' + ' | '.join(
                f'map{mr} all {row[f"m{mr}_{key}"]["all"]:.5f} '
                f'({row[f"m{mr}_{key}"]["cost_M"]:.3f}M)' for mr in SWEEP[key]), flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    keys = ['full' if r is None else str(r) for r in RANKS]

    pts = {e: {f'm{mr}_{k}': (res[e][f'm{mr}_{k}']['cost_M'], res[e][f'm{mr}_{k}']['all'])
               for k in keys for mr in SWEEP[k]} for e in roles}

    def pareto(P):
        """names not dominated by another point on BOTH cost and all-position CE."""
        return sorted([n1 for n1, (c1, v1) in P.items()
                       if not any(c2 <= c1 and v2 <= v1 and (c2 < c1 or v2 < v1)
                                  for n2, (c2, v2) in P.items() if n2 != n1)],
                      key=lambda x: P[x][0])
    front = {e: pareto(pts[e]) for e in roles}
    MID = ('384', '512', '768')
    covspread = max(abs(res[e][f'm{mr}_{k}']['cov'] - res[e][f'm{SWEEP[k][0]}_{k}']['cov'])
                    for e in roles for k in keys for mr in SWEEP[k])
    matched = max(abs(res[e][f'm{SWEEP[k][0]}_{k}']['all'] - res[e][f'm512_{k}']['all'])
                  for e in roles for k in ('256', '384'))
    pa = matched <= 1e-5
    pb = all(any(f'm512_{k}' in front[e] or f'm{SWEEP[k][0]}_{k}' in front[e] for k in MID)
             for e in roles)
    pc = covspread <= 1e-6
    pd = (all(abs(res[e][f'm512_{k}']['all'] - S1880[e][k]) <= 0.002
              for e in roles for k in ('256', 'full')) and ncov == NCOV)

    print(f'\n  MATCHED-RANK LAW (§1880): max |map_matched - map512| over table 256 and 384 = '
          f'{matched:.2e}  -> {pa}', flush=True)
    for k in ('256', '384'):
        print(f'    table {k}: ' + '  '.join(
            f'{e} map{SWEEP[k][0]} {res[e][f"m{SWEEP[k][0]}_{k}"]["all"]:.5f} '
            f'({res[e][f"m{SWEEP[k][0]}_{k}"]["cost_M"]:.3f}M) vs map512 '
            f'{res[e][f"m512_{k}"]["all"]:.5f} ({res[e][f"m512_{k}"]["cost_M"]:.3f}M)'
            for e in roles), flush=True)
    print(f'\n  PARETO FRONTIER over table rank x map rank (cost_M, all-position CE):', flush=True)
    for e in roles:
        print(f'    {e:10s} ' + '  '.join(
            f'{n}@{pts[e][n][0]:.2f}M/{pts[e][n][1]:.4f}' for n in front[e]), flush=True)
    print(f'  an INTERMEDIATE table rank (384/512/768) is on the frontier -> {pb}', flush=True)
    print(f'  and the COVERED arm is blind to the map (<1e-6) -> {pc}  max spread {covspread:.2e}',
          flush=True)
    print(f'  §1880 anchors (table 256 and full at map 512) reproduce + coverage {ncov} '
          f'-> control {pd}', flush=True)
    r2 = {'config': {'sweep': {k: list(v) for k, v in SWEEP.items()}, 'ridge': RIDGE, 'table_ranks': keys,
                     'learned': 'ridge map from the token embedding to the site row, fitted on the '
                                '5419 COVERED tokens and applied only at uncovered ones; still a '
                                'function of the current token alone',
                     'map_cost_formula': '36 * map_rank * 2 * 1152 reals',
                     'neighbour': 'the settled output-NN fallback (§1780/§1781)',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'results': res,
          'pareto_frontier': front,
          'points_cost_M_and_allpos_CE': {e: {n: list(v) for n, v in pts[e].items()} for e in roles},
          'covered_spread': covspread,
          'predictions': {'pred_a_matched_rank_law_holds': bool(pa),
                          'pred_b_intermediate_rank_on_frontier': bool(pb),
                          'pred_c_covered_blind': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
