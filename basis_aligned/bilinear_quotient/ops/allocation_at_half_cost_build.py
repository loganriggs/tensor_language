# DOES THE ALLOCATION GAIN SURVIVE AT THE HALF-COST BUILD? -- the one place §1929's caveat matters.
#
# §1928 found an MLP-heavy per-site table allocation beats uniform at matched storage, and §1929 sized it:
# a clean U with a flat 64-128 optimum (out of 512) and a best gain of +0.01874 / +0.01817 / +0.01695
# all-position nats, free. Both were measured at ONE budget (uniform-256-equivalent, 103.1086M) and ONE
# coverage (5,419).
#
# §1929 flagged that neither should be assumed to transfer: §1864 found the map-rank rule budget-dependent
# and §1924 found the fallback rank lever does not transfer across coverage at all. It also said the check
# was not worth the compute "unless someone is building on it".
#
# There is exactly one place someone would. §1882's m512_512 build at 16,110 types -- table rank 512
# uniform with a rank-512 map, 360.724M -- is the half-cost build that beats §1789's deployed design by
# 46% for +0.005 nats. If the allocation gain survives there it is free on top of the largest cost result
# in the arc; if it does not, §1928/§1929 are a 5,419-only curiosity.
#
# The family is cost-flat by construction: any (a, b) with a + b = 1024 costs 318.2561M of table, and the
# rank-512 map adds 42.467M in every arm, for 360.723M total against §1882's 360.724M. §1929's optimum was
# at attn/total = 128/512 = 25%, whose analogue here is attn 256.
#
# ROLES. skip7000, skip11000, skip1200; all-position CE. DISCOVERY ONLY, 16,110 coverage. Rung 3.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats per LESSON 40, read back per LESSON 39.
#   pred_a THE GAIN SURVIVES: at least one MLP-heavy arm beats the uniform 512/512 arm on all three roles.
#          If FALSE the §1928/§1929 result is specific to the 5,419 / 103M point it was measured at, and
#          I would say so -- I have already told Codex twice that it applies to "every point on
#          §1853-§1882", which would then need correcting for a third time on this topic.
#   pred_b AND IT IS THE SAME ORDER: the best gain is between 0.006 and 0.054, i.e. within a factor of 3
#          of §1929's PUBLISHED +0.01874 / +0.01817 / +0.01695. A deliberately wide band, because the
#          budget is 3.5x larger and the coverage 3x, and the point is whether the effect is comparable
#          rather than equal. If FALSE and the gain is LARGER, the allocation matters more at frontier
#          scale than §1929 concluded and that conclusion needs revisiting.
#   pred_c AND THE OPTIMUM SCALES: the best arm is attn 256 or attn 384, bracketing §1929's 25%-of-total
#          optimum scaled to this budget (128/512 -> 256/1024). If FALSE the optimum does not scale with
#          the budget, which is the §1864 behaviour and would mean each budget needs its own sweep.
#   pred_d CONTROLS: coverage is exactly 16,110; the uniform 512/512 arm reproduces §1882's PUBLISHED
#          5.91024 / 5.85638 / 5.88223 within 0.002; and all four arms cost within 0.001M.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = ('a512', 'a384', 'a256', 'a128')   # attention rank; mlp = 1024 - it
MAP_RANK = 64
MAPRANKS = (512,)
ALLOC = {f'a{v}': {'attn': v, 'mlp': 1024 - v} for v in (512, 384, 256, 128)}
S1882_U = {'skip7000': 5.91024, 'skip11000': 5.85638, 'skip1200': 5.88223}   # §1882 m512_512 @16,110
S1929_GAIN = [0.01874, 0.01817, 0.01695]   # §1929 best gains at 5,419 / 103.1M

RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/allocation_at_half_cost_build_results.json'
# live COVERED-CE anchors are POPULATION-dependent: 3.29205 / 3.09711 were measured on the 5,419
# covered set and at 16,110 'covered' is a different, larger population (§1882, and §1905's pred_d).
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', None),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', None),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', None)]
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'
H = m.transformer.h
NCOV = 16110
S1783 = {'skip7000': {'all': 6.01897, 'cov': 6.03465},
         'skip11000': {'all': 6.00091, 'cov': 5.97900},
         'skip1200': {'all': 6.00733, 'cov': 5.96423}}
# §1784's FULL-rank learned arm was already rank-consistent, so refitting per rank must not move it
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
    print(f'ALLOCATION AT THE HALF-COST BUILD | table ranks {RANKS} | map rank '
          f'{MAP_RANK} | DISCOVERY ONLY', flush=True)

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
        """r is an ALLOC key: per-site ranks from ALLOC[r], cost summed over the actual per-site ranks."""
        a = ALLOC[r]
        o, cost = {}, 0
        for st, tbl in tables.items():
            rk = a[st[0]]
            b = tbl.double()
            mu = b.mean(0, keepdim=True)
            U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
            o[st] = (mu + (U[:, :rk] * S[:rk]) @ Vh[:rk]).float()
            cost += rk * (NCOV + D) + 2 * D
        return o, cost

    res = {}
    for ename, epath, ref in EVAL_SETS:
        ev = load(epath)
        live = ce_both(ev)
        if ref is not None:
            assert abs(live['cov'] - ref) <= 1e-3, f'{ename} live cov {live["cov"]:.5f} != {ref}'
        row = {'live': {k: round(v, 5) for k, v in live.items()}}
        for r in RANKS:
            tc, cost = truncate(r)
            key = str(r)
            for mr in MAPRANKS:
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
            key = str(r)
            b = row[f'm512_{key}']
            print(f'    {key:12s} all {b["all"]:.5f}  ({b["cost_M"]:.4f}M table)', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    keys = [str(r) for r in RANKS]

    def ce(k2, e):
        return res[e][f'm512_{k2}']['all']

    def cst(k2):
        return res[roles[0]][f'm512_{k2}']['cost_M']
    costs = [cst(k2) for k2 in keys]
    best = {e: min(keys, key=lambda k2: ce(k2, e)) for e in roles}
    gain = {e: ce('a512', e) - ce(best[e], e) for e in roles}
    pa = all(any(ce(k2, e) < ce('a512', e) for k2 in keys if k2 != 'a512') for e in roles)
    pb = all(0.006 <= gain[e] <= 0.054 for e in roles)
    pc = all(best[e] in ('a256', 'a384') for e in roles)
    pd = (ncov == NCOV
          and all(abs(ce('a512', e) - S1882_U[e]) <= 0.002 for e in roles)
          and (max(costs) - min(costs)) <= 0.001)

    print(f'\n  all-position CE, cost-flat family a+b=1024 at {costs[0]:.4f}M, 16,110 types:',
          flush=True)
    for e in roles:
        print(f'    {e:10s} ' + '  '.join(
            f'{k2[1:]}/{1024 - int(k2[1:])} {ce(k2, e):.5f}' for k2 in keys)
            + f'   best {best[e][1:]} ({gain[e]:+.5f} vs uniform)', flush=True)
    print(f'\n  the GAIN SURVIVES at the half-cost build -> {pa}  ' + '  '.join(
        f'{e} best {gain[e]:+.5f}' for e in roles), flush=True)
    print(f'  and it is the SAME ORDER as §1929 (0.006-0.054) -> {pb}  ' + '  '.join(
        f'{e} {gain[e]:+.5f} vs §1929 {S1929_GAIN[i9]:+.5f}' for i9, e in enumerate(roles)),
        flush=True)
    print(f'  and the OPTIMUM scales (best is 256 or 384) -> {pc}  ' + '  '.join(
        f'{e} {best[e][1:]}' for e in roles), flush=True)
    print(f'  coverage {ncov}, uniform reproduces §1882, costs flat -> control {pd}  ' + '  '.join(
        f'{e} {ce("a512", e):.5f} vs {S1882_U[e]:.5f}' for e in roles), flush=True)

    r2 = {'config': {'map_ranks': list(MAPRANKS), 'ridge': RIDGE, 'table_ranks': keys,
                     'learned': 'ridge map from the token embedding to the site row, fitted on the '
                                '5419 COVERED tokens and applied only at uncovered ones; still a '
                                'function of the current token alone',
                     'map_cost_formula': '36 * map_rank * 2 * 1152 reals',
                     'neighbour': 'the settled output-NN fallback (§1780/§1781)',
                     'ROLE_NOTE': 'DISCOVERY ONLY.'},
          'results': res,
          'allocation': {k2: ALLOC[k2] for k2 in ALLOC},
          'allpos_ce': {e: {k2: res[e][f'm512_{k2}']['all'] for k2 in keys} for e in roles},
          'table_cost_M': {k2: res[roles[0]][f'm512_{k2}']['cost_M'] for k2 in keys},
          'predictions': {'pred_a_gain_survives': bool(pa),
                          'pred_b_same_order': bool(pb),
                          'pred_c_optimum_scales': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
