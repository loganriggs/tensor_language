# front_table_rank_structure: HOW MANY DIMENSIONS DOES EACH FRONT MLP'S TOKEN TABLE NEED?
#
# §1662 established what fraction of each front MLP a per-token lookup reproduces:
#     mlp0 90.27%   mlp1 96.01%   mlp2 76.98%   mlp3 67.55%
# That says the table is a good program. It says nothing about how BIG the program is. A
# 50257 x 1152 table is not an explanation of anything; a rank-k table is a claim that the
# module computes k features of the token and nothing else.
#
# §780 measured mlp0's output as ~23-dimensional (effective rank 22.7) and only 44%
# linearly predictable from the embedding. If that number is right, a rank-23 table should
# recover most of mlp0's ceiling. That is a replication target, not a fresh guess.
#
# This is the measurement I attempted at §1655 and DISCARDED: that version used an
# unweighted mean over token-means as its ablation constant and gave 23.4% of eval
# positions a ZERO vector, so it disagreed with §1326 in two independent ways. Both bugs
# are gone here -- optimal constants from opt_ablation_consts_all.pt, and the §1661 hybrid
# substitution, which is the only protocol in this project whose instrument check passes.
#
# METHOD, per site L in {0,1,2,3}:
#   fit the per-token table; restrict to SEEN rows; subtract the position-weighted mean;
#   SVD the centred matrix with rows scaled by sqrt(token count) -- so the truncation
#   minimises the POSITION-weighted reconstruction error, which is the error CE actually
#   sees, rather than treating a token appearing once as equal to one appearing 10^4 times;
#   truncate to rank r, unscale, add the mean back, and measure the ceiling with the
#   hybrid hook. Recovery is reported as a FRACTION OF THE FULL TABLE'S CEILING, so a site
#   whose full table is only 67% is not penalised twice.
#
# Registered predictions:
#   pred_a THE TABLES ARE LOW RANK: at every site, rank 64 reaches >= 90% of that site's
#          full-table ceiling. 64 of 1152 dimensions.
#   pred_b §780 REPLICATES UNDER THE CORRECTED PROTOCOL: rank 23 reaches >= 80% of mlp0's
#          full-table ceiling. (§1655's discarded run got 85.9% for this with two bugs; a
#          corrected protocol should not move it much if the bugs were coverage-related.)
#   pred_c MANIPULATION CHECK -- the curve is informative rather than saturated: at every
#          site, rank 1 falls at least 20 points below the full table. If rank 1 already
#          reproduces the table, the rank axis carries no information and no statement
#          about dimensionality can be made from it.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
SITES = [0, 1, 2, 3]
RANKS = [1, 2, 4, 8, 16, 23, 32, 64, 128]
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'front_table_rank_structure_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1662_CEILINGS = {0: 0.90265, 1: 0.96010, 2: 0.76980, 3: 0.67550}
S780_EFFECTIVE_RANK = 22.7
STATE = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


@torch.no_grad()
def sweep(rows, mlp_hook=None, score=None):
    hs = [mlp_hook] if mlp_hook is not None else []
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
def fit_table(rows, L):
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)

    def collect(mod, args, out):
        t = STATE['idx'].reshape(-1)
        s.index_add_(0, t, out.float().reshape(-1, D))
        c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
        return None
    sweep(rows, mlp_hook=H[L].mlp.register_forward_hook(collect))
    seen = c > 0
    mean = s.sum(0) / c.sum()                      # position-weighted
    tbl = mean.unsqueeze(0).repeat(50257, 1)
    tbl[seen] = s[seen] / c[seen].unsqueeze(1)
    return tbl, seen, c, mean


@torch.no_grad()
def truncate(tbl, seen, cnt, mean, r):
    """Rank-r approximation of the SEEN rows, minimising POSITION-weighted error."""
    rows = tbl[seen] - mean.unsqueeze(0)
    w = cnt[seen].sqrt().unsqueeze(1)
    U, S, Vh = torch.linalg.svd(rows * w, full_matrices=False)
    k = min(r, S.numel())
    approx = (U[:, :k] * S[:k]) @ Vh[:k] / w
    out = tbl.clone()
    out[seen] = approx + mean.unsqueeze(0)
    return out


@torch.no_grad()
def ce(rows, L, mode, const_m, tbl=None, seen=None):
    mh = None
    if mode == 'const':
        mh = H[L].mlp.register_forward_hook(
            lambda mo, a, o: const_m.to(o.dtype).expand_as(o))
    elif mode == 'table':
        def th(mo, a, o):
            sub = tbl[STATE['idx'].reshape(-1)].reshape(o.shape).to(o.dtype)
            return torch.where(seen.to(DEV)[STATE['idx']].unsqueeze(-1), sub, o)
        mh = H[L].mlp.register_forward_hook(th)
    acc = {'t': 0.0, 'n': 0}

    def score(lg, tg, idx):
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').reshape(tg.shape)[:, 64:]
        cov = seen.to(DEV)[idx[:, 64:]]
        acc['t'] += float(e[cov].sum()); acc['n'] += int(cov.sum())
    sweep(rows, mlp_hook=mh, score=score)
    return acc['t'] / max(acc['n'], 1)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    print(f'FRONT TABLE RANK STRUCTURE | sites {SITES} | ranks {RANKS} | position-weighted '
          f'SVD, §1661 hybrid substitution | fit skip1200, eval skip7000', flush=True)

    out = {}
    for L in SITES:
        const_m = K[f'mlp{L}'].to(DEV).float()
        tbl, seen, cnt, mean = fit_table(fit, L)
        cl = ce(ev, L, 'live', const_m, seen=seen)
        cc = ce(ev, L, 'const', const_m, seen=seen)
        st = cc - cl
        ct = ce(ev, L, 'table', const_m, tbl, seen)
        full = (cc - ct) / st
        nseen = int(seen.sum())
        print(f'\n  mlp{L}: {nseen} tokens seen | stake {st:.4f} | FULL table ceiling '
              f'{full:.2%}  (§1662 {S1662_CEILINGS[L]:.2%})', flush=True)
        curve = {}
        for r in RANKS:
            tr = truncate(tbl, seen, cnt, mean, r)
            cr = ce(ev, L, 'table', const_m, tr, seen)
            ceil_r = (cc - cr) / st
            frac = ceil_r / full if full > 1e-9 else float('nan')
            curve[r] = {'ceiling': round(ceil_r, 5), 'frac_of_full': round(frac, 5)}
            print(f'    rank {r:4d}: ceiling {ceil_r:7.2%}  = {frac:6.2%} of full table',
                  flush=True)
        out[f'mlp{L}'] = {'tokens_seen': nseen, 'stake': round(st, 5),
                          'full_ceiling': round(full, 5),
                          's1662_ceiling': S1662_CEILINGS[L], 'curve': curve}

    r64 = [out[f'mlp{L}']['curve'][64]['frac_of_full'] for L in SITES]
    r1 = [out[f'mlp{L}']['curve'][1]['frac_of_full'] for L in SITES]
    r23_mlp0 = out['mlp0']['curve'][23]['frac_of_full']

    pa = all(v >= 0.90 for v in r64)
    pb = r23_mlp0 >= 0.80
    pc = all(v <= 0.80 for v in r1)

    print(f'\n  rank-64 fraction of full table, by site: '
          f'{[f"{v:.1%}" for v in r64]}  -> low rank {pa}', flush=True)
    print(f'  mlp0 at rank 23 (§780 effective rank {S780_EFFECTIVE_RANK}): '
          f'{r23_mlp0:.2%} of its full table', flush=True)
    print(f'  rank-1 fraction by site: {[f"{v:.1%}" for v in r1]}  -> curve informative {pc}',
          flush=True)
    print(f'  ceiling replication vs §1662: ' +
          '  '.join(f'mlp{L} {out[f"mlp{L}"]["full_ceiling"]:.2%}' for L in SITES), flush=True)

    res = {'config': {'sites': SITES, 'ranks': RANKS,
                      'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'svd': 'centred on the position-weighted mean, rows scaled by sqrt(count) '
                             'so truncation minimises POSITION-weighted error',
                      'substitution': 'HYBRID -- table at covered positions, MLP live elsewhere (§1661)',
                      'scoring': 'covered positions only',
                      's780_effective_rank': S780_EFFECTIVE_RANK,
                      'supersedes': 'S1655 (discarded: unweighted mean constant + zero fallback)'},
           'sites': out,
           'predictions': {'pred_a_rank64_ge_90pct_all_sites': bool(pa),
                           'pred_b_s780_rank23_ge_80pct_mlp0': bool(pb),
                           'pred_c_curve_informative_rank1_le_80pct': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
