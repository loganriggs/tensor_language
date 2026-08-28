# HOW MUCH MAP? -- sweeping the embedding->row map's rank at a fixed table rank.
#
# §1785: at table rank 64, predicting the uncovered token's site row from its embedding with a rank-64
# ridge map beats copying the output-NN neighbour's row by 0.006-0.030 nats, and is a better marginal
# buy than more table rank -- 0.00566 nats per million against 0.00367 for the rank 64 -> 256 step.
# That was one map rank. §1770/§1771 had to sweep the TABLE rank twice before the design point was
# bracketed, and the same discipline applies here.
#
# Table rank fixed at 64 -- §1771's efficiency region -- and the map rank swept over 8, 16, 64, 256.
# The map is refitted inside the truncated basis at every point (§1785's repair).
#
# ROLES. skip7000, skip11000, skip1200; all-position CE, with covered CE as a wiring check.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats with margins per LESSON 40, read
# back per LESSON 39:
#   pred_a THE MAP-RANK EFFICIENCY OPTIMUM IS INTERIOR: the best nats-per-million-of-map is at rank 16
#          or 64, not at 8 or 256, on every role. If FALSE at the top the sweep is under-budgeted --
#          the defect §1756, §1770 and §1771 each recorded and which I have now repeated twice -- and
#          I report that rather than a design point.
#   pred_b EVERY MAP RANK BEATS THE NEIGHBOUR on all-position CE, at every role. If FALSE, a small map
#          is worse than copying, and there is a floor below which predicting is the wrong move.
#   pred_c FIDELITY IS MONOTONE INCREASING IN MAP RANK at every role. If FALSE the map overfits the
#          5,419 covered tokens at high rank, which is the same failure mode §1755 found in the tables
#          and would put a ceiling on this direction.
#   pred_d CONTROLS: the neighbour arm and the map-rank-64 arm reproduce §1785's 6.19187 / 6.17330,
#          6.18267 / 6.15261 and 6.15065 / 6.14463 within 0.002, and coverage is exactly 5419.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (64,)
MAP_RANKS = (8, 16, 64, 256)
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/map_rank_sweep_results.json'
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
S1785_R64 = {'skip7000': {'neighbour': 6.19187, 'learned': 6.17330},
             'skip11000': {'neighbour': 6.18267, 'learned': 6.15261},
             'skip1200': {'neighbour': 6.15065, 'learned': 6.14463}}
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
    print(f'MAP RANK SWEEP | table rank {RANKS[0]} | map ranks {MAP_RANKS} | DISCOVERY ONLY',
          flush=True)

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
        tc, cost = truncate(RANKS[0])
        fr = build_full(tc, 'neighbour', None)
        row['neighbour'] = {**{k: round(v, 5) for k, v in
                               ce_both(ev, [(st, row_hook(fr[st])) for st in sites]).items()},
                            'cost_M': round(cost / 1e6, 4)}
        del fr
        torch.cuda.empty_cache()
        for mr in MAP_RANKS:
            mc = 36 * mr * 2 * D
            fr = build_full(tc, 'learned', fit_maps(tc, mr))
            row[f'learned_m{mr}'] = {**{k: round(v, 5) for k, v in
                                        ce_both(ev, [(st, row_hook(fr[st])) for st in sites]).items()},
                                     'cost_M': round((cost + mc) / 1e6, 4),
                                     'map_cost_M': round(mc / 1e6, 4)}
            del fr
            torch.cuda.empty_cache()
        del tc
        res[ename] = row
        print(f'\n  {ename}: live cov {live["cov"]:.5f} all {live["all"]:.5f}', flush=True)
        n = row['neighbour']
        print(f'    neighbour            all {n["all"]:.5f}  ({n["cost_M"]:.3f}M)', flush=True)
        for mr in MAP_RANKS:
            l = row[f'learned_m{mr}']
            g = n['all'] - l['all']
            print(f'    learned map rank {mr:3d}  all {l["all"]:.5f}  ({l["cost_M"]:.3f}M, map '
                  f'{l["map_cost_M"]:.3f}M)  gain {g:+.5f}  {g / l["map_cost_M"]:.5f} nats/M',
                  flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    def eff(e, mr):
        l = res[e][f'learned_m{mr}']
        return (res[e]['neighbour']['all'] - l['all']) / l['map_cost_M']
    best = {e: max(MAP_RANKS, key=lambda mr: eff(e, mr)) for e in roles}
    pa = all(MAP_RANKS[0] < best[e] < MAP_RANKS[-1] for e in roles)
    pb = all(res[e][f'learned_m{mr}']['all'] < res[e]['neighbour']['all']
             for e in roles for mr in MAP_RANKS)
    pc = all(res[e][f'learned_m{MAP_RANKS[i]}']['all']
             > res[e][f'learned_m{MAP_RANKS[i + 1]}']['all']
             for e in roles for i in range(len(MAP_RANKS) - 1))
    pd = (all(abs(res[e]['neighbour']['all'] - v['neighbour']) <= 0.002
              and abs(res[e]['learned_m64']['all'] - v['learned']) <= 0.002
              for e, v in S1785_R64.items()) and ncov == NCOV)

    print(f'\n  map-rank efficiency optimum is interior on every role {best} -> {pa}', flush=True)
    print(f'    nats per M of MAP: ' + ' | '.join(
        f'{e} ' + ' '.join(f'{mr}:{eff(e, mr):.5f}' for mr in MAP_RANKS) for e in roles),
        flush=True)
    print(f'  every map rank beats the neighbour -> {pb}', flush=True)
    print(f'  fidelity monotone increasing in map rank -> {pc}', flush=True)
    print(f'  map rank 64 and the neighbour reproduce §1785 + coverage {ncov} -> control {pd}',
          flush=True)

    r2 = {'config': {'map_ranks': list(MAP_RANKS), 'ridge': RIDGE,
                     'table_rank': RANKS[0],
                     'learned': 'ridge map from the token embedding to the site row, fitted on the '
                                '5419 COVERED tokens and applied only at uncovered ones; still a '
                                'function of the current token alone',
                     'map_cost_formula': '36 * map_rank * 2 * 1152 reals',
                     'neighbour': 'the settled output-NN fallback (§1780/§1781)',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'results': res,
          'map_rank_efficiency_optimum': {e: best[e] for e in roles},
          'predictions': {'pred_a_map_rank_optimum_is_interior': bool(pa),
                          'pred_b_every_map_rank_beats_neighbour': bool(pb),
                          'pred_c_fidelity_monotone_in_map_rank': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
