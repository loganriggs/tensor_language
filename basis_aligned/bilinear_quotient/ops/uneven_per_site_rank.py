# UNEVEN PER-SITE TABLE RANK -- the cost question the mechanism results have been pointing at.
#
# The §1853-§1882 frontier spends table rank UNIFORMLY across all 36 sites. §1891 measured that the
# program's fidelity to the live model is carried by ATTENTION -- attention restorations change up to
# 96.56% of predictions on covered positions against the MLP maximum of 0.03% (§1904), a ratio of 3,200x
# -- and §1908 explained which attention sites and why. Nothing has asked whether the TABLE budget should
# follow that.
#
# The test is cost-matched by construction. At 5,419 types a rank-r table costs 36*(r*(NCOV+D)+2D); 18
# sites at 384 plus 18 at 128 costs 60.641M, exactly what 36 sites at 256 costs. The map is rank 512 in
# every arm (above every table rank used, per §1880/§1881, and identical so it cannot confound). So the
# three arms differ ONLY in how the same storage is distributed.
#
# Selection: this question has been on the board for Codex since 07:57Z and restated at 08:06Z, 09:27Z,
# 12:03Z and 13:31Z; they have stayed on E4 and Family-F throughout and have not claimed it. Taking it
# now rather than leave the program's most valuable open cost question idle, and saying so on the board.
#
# ROLES. skip7000, skip11000, skip1200; all-position CE. DISCOVERY ONLY, 5,419 coverage. Rung 3.
#
# Registered predictions, TWO-SIDED per LESSONS 31, absolute nats per LESSON 40, read back per LESSON 39.
#   pred_a ATTENTION-HEAVY BEATS UNIFORM AT MATCHED COST: the 18-attn-at-384 / 18-mlp-at-128 arm has
#          LOWER all-position CE than the uniform-256 arm on all three roles. If TRUE the frontier of
#          §1853-§1882 is leaving storage on the table and every point on it can be improved for free by
#          redistributing. If FALSE the uniform allocation is already right and §1891's behavioural
#          localisation does NOT imply a storage one -- which would be worth knowing, since I have been
#          telling Codex for six hours that it does.
#   pred_b AND MLP-HEAVY IS WORSE THAN UNIFORM: the reverse allocation (18 mlp at 384, 18 attn at 128)
#          has HIGHER all-position CE than uniform on all three roles. This is the two-sided control that
#          makes pred_a mean something: if BOTH uneven arms beat uniform, the gain is about unevenness
#          per se and not about attention; if both are worse, uniform is a local optimum and the axis is
#          not worth exploring.
#   pred_c AND THE THREE ARMS COST THE SAME: table storage matches within 0.001M across all three, by
#          construction. Registered because a cost-matched claim whose costs were not checked is not a
#          cost-matched claim, and §1861's iso-cost result is the standard this has to meet.
#   pred_d CONTROLS: coverage is exactly 5,419; and the UNIFORM arm reproduces §1880's PUBLISHED m512_256
#          figures 6.02422 / 5.99343 / 6.00680 within 0.002, since uniform-256-with-a-rank-512-map is
#          exactly that build.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = ('uniform', 'attn_heavy', 'mlp_heavy')
MAP_RANK = 64
MAPRANKS = (512,)
ALLOC = {'uniform': {'attn': 256, 'mlp': 256},
         'attn_heavy': {'attn': 384, 'mlp': 128},
         'mlp_heavy': {'attn': 128, 'mlp': 384}}
S1880_U256 = {'skip7000': 6.02422, 'skip11000': 5.99343, 'skip1200': 6.00680}   # §1880 m512_256
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/uneven_per_site_rank_results.json'
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
    print(f'UNEVEN PER-SITE TABLE RANK | table ranks {RANKS} | map rank '
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
            a, b = row[f'm64_{key}'], row[f'm512_{key}']
            print(f'    table {key:5s}  map64 all {a["all"]:.5f} ({a["cost_M"]:.3f}M) | '
                  f'map512 all {b["all"]:.5f} ({b["cost_M"]:.3f}M)  '
                  f'gain {a["all"] - b["all"]:+.5f} for +{b["cost_M"] - a["cost_M"]:.3f}M', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    keys = [str(r) for r in RANKS]

    def ce(k2, e):
        return res[e][f'm512_{k2}']['all']

    def cst(k2):
        return res[roles[0]][f'm512_{k2}']['cost_M']
    dA = {e: ce('uniform', e) - ce('attn_heavy', e) for e in roles}
    dM = {e: ce('mlp_heavy', e) - ce('uniform', e) for e in roles}
    costs = [cst(k2) for k2 in keys]
    pa = all(dA[e] > 0 for e in roles)
    pb = all(dM[e] > 0 for e in roles)
    pc = (max(costs) - min(costs)) <= 0.001
    pd = (ncov == NCOV
          and all(abs(ce('uniform', e) - S1880_U256[e]) <= 0.002 for e in roles))

    print(f'\n  all-position CE at MATCHED storage ({costs[0]:.3f}M table + map), 5,419 types:',
          flush=True)
    for e in roles:
        print(f'    {e:10s} uniform-256 {ce("uniform", e):.5f}   attn-heavy(384/128) '
              f'{ce("attn_heavy", e):.5f} ({-dA[e]:+.5f})   mlp-heavy(128/384) '
              f'{ce("mlp_heavy", e):.5f} ({+dM[e]:+.5f})', flush=True)
    print(f'\n  ATTENTION-HEAVY beats uniform at matched cost -> {pa}  ' + '  '.join(
        f'{e} {dA[e]:+.5f}' for e in roles), flush=True)
    print(f'  and MLP-HEAVY is worse than uniform -> {pb}  ' + '  '.join(
        f'{e} {dM[e]:+.5f}' for e in roles), flush=True)
    print(f'  and the three arms cost the same (<= 0.001M) -> {pc}  ' + '  '.join(
        f'{k2} {cst(k2):.4f}M' for k2 in keys), flush=True)
    print(f'  coverage {ncov}, uniform reproduces §1880 -> control {pd}  ' + '  '.join(
        f'{e} {ce("uniform", e):.5f} vs {S1880_U256[e]:.5f}' for e in roles), flush=True)

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
          'predictions': {'pred_a_attn_heavy_beats_uniform': bool(pa),
                          'pred_b_mlp_heavy_worse': bool(pb),
                          'pred_c_costs_matched': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
