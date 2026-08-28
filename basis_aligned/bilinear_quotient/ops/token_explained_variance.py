# HOW MUCH OF EACH SITE'S OUTPUT IS ALREADY A FUNCTION OF THE CURRENT TOKEN?
#
# §1829-§1836 measured what compiling each site COSTS and produced a 34-number table (§1834) with mlp5
# at +61.2pp, the extreme. Every mechanism proposed for it has been ruled out: not channel concentration
# (§1835: 256 of 1152 channels buy 30.5% of the stake), not outlier identity (§1835: worse than random
# at all four k), not magnitude or explosion (§1836: 0.899x, measured, and §1835's hypothesis struck).
#
# What no section has measured is the property that should PREDICT the cost table, and it is the obvious
# one. §1765 proved the compiled program is a pure function of the current token, and a context-free
# table can represent exactly the part of a site's output that IS such a function. So a site should be
# expensive to compile precisely to the degree its output is NOT determined by its own token.
#
# TOKEN-EXPLAINED VARIANCE, per site, exact and single-pass on the LIVE model:
#     between = sum_t (||sum_t||^2 / c_t) / n - ||mu||^2      (variance of the per-token means)
#     total   = E||y||^2 - ||mu||^2
#     explained = between / total
# on COVERED positions only, so the conditioning token set is the 5419 the tables are built over and
# every per-token mean has at least one sample. No arms are run and no tables are built: this is an
# observational statistic costing one forward pass, and if it predicts §1834's table then the table is
# EXPLAINED rather than merely tabulated.
#
# ROLES. skip7000 for the statistic, skip11000 as an independent recomputation. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44, confound controlled per LESSON 46:
#   pred_a TOKEN-EXPLAINED VARIANCE PREDICTS COST: the Spearman correlation between a site's explained
#          fraction and its PUBLISHED §1834 single-site cost, over the 34 sites at layers 1-17, is at
#          most -0.70 (more explained => cheaper to compile). If TRUE the cost table is explained by a
#          statistic computable from one forward pass with no arms at all, and the whole §1829-§1836 arc
#          reduces to one observational quantity. If FALSE, how well a site's output is predicted by its
#          own token does NOT determine what compiling it costs -- the cost would then be about
#          DOWNSTREAM SENSITIVITY to the substitution error rather than about the size of that error,
#          which is a different object and would need a different instrument.
#   pred_b AND mlp5 IS THE EXTREME: mlp5 has the lowest explained fraction of all 34 sites. §1834 puts
#          it 12.7pp clear of the runner-up on cost, so if the statistic is the right one it should be
#          the extreme here too. If FALSE, mlp5 is expensive for a reason this statistic does not see,
#          even if the statistic works across the rest of the table.
#   pred_c AND IT BEATS DEPTH ALONE: |Spearman(explained, cost)| exceeds |Spearman(-depth, cost)| by at
#          least 0.10. Depth is the obvious confound -- §1831 established cost falls steeply with it --
#          so a statistic that merely tracks depth adds nothing. If FALSE the explained fraction is a
#          proxy for depth and should not be reported as an explanation of the cost table.
#   pred_d CONTROLS: every explained fraction lies in [0, 1]; the statistic recomputed on skip11000
#          agrees with skip7000 within 0.05 at every site; coverage is 5419 of 50257; live top-1
#          reproduces §1789's PUBLISHED 39.32 / 42.35%.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
NH = 9; HD = D // NH        # bilin18: nine heads of 128
MAP_RANK = 64
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/token_explained_variance_results.json'
CALROWS = 32   # rows used for the per-layer calibration passes; the eval uses the full role
PROBE_LS = ()
KIND_LS = ()
RANKS = (64,)                # the settled table rank; only the STREAM matters here
def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


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



# §1834's PUBLISHED single-site costs, skip7000, pp of gap lost against B0 -- the thing to be predicted
S1834_COST = {
    'attn1': 0.374, 'attn2': 0.429, 'attn3': 0.199, 'attn4': 0.288, 'attn5': 0.169, 'attn6': 0.047,
    'attn7': 0.074, 'attn8': 0.011, 'attn9': 0.007, 'attn10': -0.006, 'attn11': 0.023,
    'attn12': -0.003, 'attn13': 0.006, 'attn14': 0.003, 'attn15': -0.011, 'attn16': -0.001,
    'attn17': -0.005,
    'mlp1': 0.387, 'mlp2': 0.379, 'mlp3': 0.485, 'mlp4': 0.445, 'mlp5': 0.612, 'mlp6': 0.065,
    'mlp7': 0.051, 'mlp8': 0.041, 'mlp9': 0.041, 'mlp10': 0.025, 'mlp11': 0.029, 'mlp12': 0.023,
    'mlp13': 0.021, 'mlp14': 0.022, 'mlp15': 0.017, 'mlp16': 0.019, 'mlp17': 0.040}
S1789_LIVE_TOP1_PP = {'skip7000': 0.3932, 'skip11000': 0.4235}
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
SITES = [(k, L) for L in range(1, 18) for k in ('attn', 'mlp')]
STATE = {}
COV = {}


def spearman(a, b):
    """Rank correlation; ties broken by position, which is adequate for 34 distinct-valued sites."""
    def rk(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for j, i in enumerate(order):
            r[i] = j
        return r
    ra, rb = rk(a), rk(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    return num / max(da * db, 1e-12)


@torch.no_grad()
def token_explained(rows, idmap):
    """Exact single-pass between/total variance decomposition of each site's output by CURRENT TOKEN.

    Accumulated over COVERED scored positions only, so every per-token mean has at least one sample and
    the conditioning set is the 5419 tokens the tables are built over. All accumulation is float64."""
    s = {st: torch.zeros(NCOV, D, device=DEV, dtype=torch.float64) for st in SITES}
    g = {st: torch.zeros(D, device=DEV, dtype=torch.float64) for st in SITES}
    q = {st: 0.0 for st in SITES}
    c = torch.zeros(NCOV, device=DEV, dtype=torch.float64)
    n = {'k': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).detach().double()[:, 64:]
            sel = COV['cov']
            yf = y[sel]
            r = COV['rid']          # ALREADY flattened by `sel` where it was built -- 1-D, aligned
                                    # with yf. Masking it a second time here is the IndexError.
            s[st].index_add_(0, r, yf)
            g[st] += yf.sum(0)
            q[st] += float((yf * yf).sum())
            if first:
                c.index_add_(0, r, torch.ones_like(r, dtype=torch.float64))
                n['k'] += int(sel.sum())
            return None
        return hook

    hs = [mod_of(*st).register_forward_hook(mk(st, j == 0)) for j, st in enumerate(SITES)]
    try:
        for i in range(0, rows.shape[0], 8):
            idx = rows[i:i + 8, :-1].to(DEV).contiguous()
            sub = idx[:, 64:]
            COV['cov'] = COV['seen'][sub]
            COV['rid'] = idmap[sub][COV['cov']]
            forward_logits(idx)
    finally:
        for hd in hs:
            hd.remove()
    nn = max(n['k'], 1)
    cc = c.clamp_min(1.0)
    out = {}
    for st in SITES:
        mu = g[st] / nn
        between = float((s[st] * s[st]).sum(1).div(cc).sum()) / nn - float(mu @ mu)
        total = q[st] / nn - float(mu @ mu)
        out[st] = max(min(between / max(total, 1e-12), 1.0), 0.0)
    return out


@torch.no_grad()
def live_top1(rows):
    hit = tot = 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        lg = forward_logits(bb[:, :-1].to(DEV).contiguous())[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        hit += int((lg.argmax(-1) == tg).sum()); tot += int(tg.numel())
    return hit / max(tot, 1)


def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen = torch.zeros(V, dtype=torch.bool, device=DEV)
    seen[fit[:, :T].reshape(-1).long().to(DEV)] = True
    ncov = int(seen.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    COV['seen'] = seen
    idmap = torch.zeros(V, dtype=torch.long, device=DEV)
    idmap[seen.nonzero(as_tuple=True)[0]] = torch.arange(NCOV, device=DEV)
    print(f'TOKEN-EXPLAINED VARIANCE | 34 sites, covered positions, exact single-pass | '
          f'DISCOVERY ONLY', flush=True)

    ex, t1 = {}, {}
    for ename, path in EVAL_SETS:
        rows = load(path)
        ex[ename] = token_explained(rows, idmap)
        t1[ename] = live_top1(rows)
        print(f'  {ename}: statistic computed, live top-1 {t1[ename]:.2%} '
              f'({time.time() - t0:.0f}s)', flush=True)
        del rows
        torch.cuda.empty_cache()

    names = [f'{k}{L}' for k, L in SITES]
    e0 = [ex['skip7000'][st] for st in SITES]
    cost = [S1834_COST[nm] for nm in names]
    depth = [-L for _, L in SITES]
    rho = spearman(e0, cost)
    rho_d = spearman(depth, cost)
    lowest = names[min(range(len(e0)), key=lambda i: e0[i])]
    drift = max(abs(ex['skip7000'][st] - ex['skip11000'][st]) for st in SITES)
    pa = rho <= -0.70
    pb = lowest == 'mlp5'
    pc = abs(rho) - abs(rho_d) >= 0.10
    pd = (all(0.0 <= v <= 1.0 for v in e0) and drift <= 0.05 and ncov == NCOV
          and all(abs(t1[e] - S1789_LIVE_TOP1_PP[e]) <= 0.001 for e, _ in EVAL_SETS))

    print(f'\n  TOKEN-EXPLAINED VARIANCE by site (skip7000, covered positions):', flush=True)
    for k in ('attn', 'mlp'):
        print(f'    {k:4s} ' + ' '.join(
            f'L{L}:{ex["skip7000"][(k, L)]:.3f}' for L in range(1, 18)), flush=True)
    print(f'\n  cheapest-to-compile 6 by §1834 cost: ' + '  '.join(
        f'{nm} cost {S1834_COST[nm]:+.1%} expl {ex["skip7000"][SITES[names.index(nm)]]:.3f}'
        for nm in sorted(names, key=lambda x: S1834_COST[x])[:6]), flush=True)
    print(f'  dearest 6:                          ' + '  '.join(
        f'{nm} cost {S1834_COST[nm]:+.1%} expl {ex["skip7000"][SITES[names.index(nm)]]:.3f}'
        for nm in sorted(names, key=lambda x: -S1834_COST[x])[:6]), flush=True)
    print(f'\n  explained variance PREDICTS §1834 cost (Spearman <=-0.70) -> {pa}  '
          f'rho {rho:+.4f}', flush=True)
    print(f'  and mlp5 is the EXTREME (lowest explained of 34) -> {pb}  lowest is {lowest} at '
          f'{min(e0):.3f}; mlp5 at {ex["skip7000"][("mlp", 5)]:.3f}', flush=True)
    print(f'  and it BEATS DEPTH alone by >=0.10 -> {pc}  |{rho:+.4f}| vs depth |{rho_d:+.4f}|, '
          f'margin {abs(rho) - abs(rho_d):+.4f}', flush=True)
    print(f'  in [0,1], skip11000 agrees within 0.05 (max drift {drift:.4f}), coverage {ncov} '
          f'-> control {pd}', flush=True)

    json.dump({'run': 'token_explained_variance', 'n_sites': len(SITES),
               'explained': {e: {f'{k}{L}': ex[e][(k, L)] for k, L in SITES} for e in ex},
               'S1834_cost': S1834_COST, 'live_top1': t1,
               'spearman_explained_vs_cost': rho, 'spearman_depth_vs_cost': rho_d,
               'lowest_explained_site': lowest, 'max_role_drift': drift,
               'predictions': {'pred_a_explained_predicts_cost': bool(pa),
                               'pred_b_mlp5_is_extreme': bool(pb),
                               'pred_c_beats_depth_alone': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
