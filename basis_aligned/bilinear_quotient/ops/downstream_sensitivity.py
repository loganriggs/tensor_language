# DOWNSTREAM SENSITIVITY -- the instrument the whole §1824-§1837 arc has been missing.
#
# §1837 tested whether a site's compilation cost is explained by how well its output is already a
# function of its own token -- what a context-free table can represent -- and the correlation came back
# at Spearman +0.466, the WRONG SIGN, with depth alone beating it at +0.853. The decisive pair: `mlp1`
# is the MOST token-determined site (explained 0.560, the most accurate table in the network) and costs
# +38.7pp; `attn14` is the LEAST (0.104, the worst table available) and costs +0.3pp. The site with the
# smallest substitution error does the most damage.
#
# §1837's registered failure branch is now the standing account: the cost is about DOWNSTREAM
# SENSITIVITY to the substitution error, not the size of that error. Every instrument since §1824 has
# measured the substituted stream -- its norm, mean direction, second moment, cross-position routing,
# channel structure, and the representability of the site it replaces. All of them measure the ERROR.
# None measures what the error DOES.
#
# This measures it, with the substitution taken out of the picture entirely: inject Gaussian noise of
# FIXED RELATIVE SIZE into each site's output on the LIVE model and read the loss change. Three seeds
# per site. That is the site's amplification factor -- a property of what sits ABOVE it.
#
# The first-order account then predicts damage ~ sensitivity x error, so the run also recomputes the
# best-per-token-table RMS error relative to each site's RMS output (sqrt(within-token variance /
# E||y||^2), the same decomposition as §1837) and tests the product.
#
# ROLES. skip7000 for the sweep, skip11000 for the drift control. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44, confound controlled per LESSON 46:
#   pred_a SENSITIVITY PREDICTS COST: Spearman between a site's sensitivity and its PUBLISHED §1834
#          single-site cost, over the 34 sites, is at least +0.70. If TRUE the cost table is explained
#          by what sits ABOVE each site rather than by anything about the site itself, and the standing
#          account is confirmed. If FALSE, even the downstream-sensitivity account fails, and NOTHING
#          measured about a site predicts its compilation cost except its DEPTH -- which would make the
#          §1834 table a statement about position in the stack and nothing else, and would end the
#          search for a site-level explanation rather than continue it.
#   pred_b AND IT BEATS DEPTH: that Spearman exceeds depth's PUBLISHED +0.853 by at least 0.05. Depth is
#          the standing champion and every candidate so far has been a worse proxy for it. If FALSE,
#          sensitivity is one more proxy for depth and should not be reported as an explanation, however
#          well it correlates.
#   pred_c AND THE FIRST-ORDER PRODUCT BEATS SENSITIVITY ALONE: Spearman(sensitivity x relative
#          substitution error, cost) exceeds Spearman(sensitivity, cost). This is what a first-order
#          account literally predicts -- damage is the error times the amplification. If FALSE the error
#          magnitude adds nothing even multiplicatively, and the cost is sensitivity alone, which would
#          be a stronger and stranger claim than the first-order one.
#   pred_d CONTROLS: the injected noise moves top-1 by more than 0.1pp at EVERY site, so no site is
#          scored on a perturbation that did not land (LESSON 46 -- distinguish "did not turn" from "had
#          nowhere to turn"); the explained fractions recomputed here reproduce §1837's PUBLISHED values
#          within 0.02 at every site; live top-1 reproduces §1789's PUBLISHED 39.32 / 42.35%; coverage
#          5419 of 50257.
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
OUT = PT + 'ops/downstream_sensitivity_results.json'
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
# §1837's PUBLISHED token-explained variance, skip7000 -- reproduced here as a control
S1837_EXPLAINED = {
    'attn1': 0.544, 'attn2': 0.252, 'attn3': 0.228, 'attn4': 0.190, 'attn5': 0.185, 'attn6': 0.290,
    'attn7': 0.255, 'attn8': 0.164, 'attn9': 0.230, 'attn10': 0.238, 'attn11': 0.177,
    'attn12': 0.164, 'attn13': 0.131, 'attn14': 0.104, 'attn15': 0.152, 'attn16': 0.124,
    'attn17': 0.459,
    'mlp1': 0.560, 'mlp2': 0.407, 'mlp3': 0.363, 'mlp4': 0.255, 'mlp5': 0.290, 'mlp6': 0.241,
    'mlp7': 0.224, 'mlp8': 0.233, 'mlp9': 0.221, 'mlp10': 0.218, 'mlp11': 0.213, 'mlp12': 0.224,
    'mlp13': 0.230, 'mlp14': 0.213, 'mlp15': 0.320, 'mlp16': 0.661, 'mlp17': 0.604}
S1837_DEPTH_RHO = 0.8527       # §1837's PUBLISHED Spearman of -depth against cost -- the champion
S1789_LIVE_TOP1_PP = {'skip7000': 0.3932, 'skip11000': 0.4235}
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt'),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt')]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
SITES = [(k, L) for L in range(1, 18) for k in ('attn', 'mlp')]
REL = 0.10                     # noise RMS as a fraction of the site's per-position output RMS
SEEDS = (0, 1, 2)
STATE = {}
COV = {}


def spearman(a, b):
    """Rank correlation; ties broken by position, adequate for 34 distinct-valued sites."""
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


def noise_hook(rel, seed):
    """Add Gaussian noise of RMS `rel` x the site's own per-position output RMS.

    Scaling per position rather than globally makes the perturbation the SAME RELATIVE SIZE at every
    site, which is the whole point: sensitivity must not be confounded with how large a site's writes
    happen to be."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        g = torch.Generator(device=y.device).manual_seed(seed)
        z = torch.randn(y.shape, generator=g, device=y.device, dtype=torch.float32)
        rms = y.float().pow(2).mean(-1, keepdim=True).sqrt()
        y2 = y + (z * rms * rel).to(y.dtype)
        return (y2,) + tuple(out[1:]) if isinstance(out, tuple) else y2
    return hook


@torch.no_grad()
def top1(rows, hooks=()):
    hit = tot = 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        lg = forward_logits(bb[:, :-1].to(DEV).contiguous(), hooks)[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        hit += int((lg.argmax(-1) == tg).sum()); tot += int(tg.numel())
    return hit / max(tot, 1)


@torch.no_grad()
def explained_and_relerr(rows, idmap):
    """§1837's decomposition, plus the RMS best-per-token-table error relative to the site's RMS."""
    s = {st: torch.zeros(NCOV, D, device=DEV, dtype=torch.float64) for st in SITES}
    g = {st: torch.zeros(D, device=DEV, dtype=torch.float64) for st in SITES}
    q = {st: 0.0 for st in SITES}
    c = torch.zeros(NCOV, device=DEV, dtype=torch.float64)
    n = {'k': 0}

    def mk(st, first):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).detach().double()[:, 64:]
            yf = y[COV['cov']]
            r = COV['rid']
            s[st].index_add_(0, r, yf)
            g[st] += yf.sum(0)
            q[st] += float((yf * yf).sum())
            if first:
                c.index_add_(0, r, torch.ones_like(r, dtype=torch.float64))
                n['k'] += int(COV['cov'].sum())
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
    ex, re2 = {}, {}
    for st in SITES:
        mu = g[st] / nn
        msq = q[st] / nn                                   # E||y||^2
        between = float((s[st] * s[st]).sum(1).div(cc).sum()) / nn - float(mu @ mu)
        total = msq - float(mu @ mu)
        within = max(total - between, 0.0)
        ex[st] = max(min(between / max(total, 1e-12), 1.0), 0.0)
        re2[st] = (within / max(msq, 1e-12)) ** 0.5        # RMS table error / RMS output
    return ex, re2


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
    print(f'DOWNSTREAM SENSITIVITY | {REL:.0%} relative noise, {len(SEEDS)} seeds x 34 sites | '
          f'DISCOVERY ONLY', flush=True)

    ev = load(EVAL_SETS[0][1])
    ex, relerr = explained_and_relerr(ev, idmap)
    base = top1(ev)
    base2 = top1(load(EVAL_SETS[1][1]))
    print(f'  live top-1 {base:.2%} / {base2:.2%}; decomposition done ({time.time() - t0:.0f}s)',
          flush=True)

    sens = {}
    for st in SITES:
        drops = [base - top1(ev, [(st, noise_hook(REL, sd))]) for sd in SEEDS]
        sens[st] = sum(drops) / len(drops)
    print(f'  sensitivity swept ({time.time() - t0:.0f}s)', flush=True)

    names = [f'{k}{L}' for k, L in SITES]
    cost = [S1834_COST[nm] for nm in names]
    sv = [sens[st] for st in SITES]
    pv = [sens[st] * relerr[st] for st in SITES]
    rho = spearman(sv, cost)
    rho_p = spearman(pv, cost)
    landed = min(sv)
    exdrift = max(abs(ex[st] - S1837_EXPLAINED[f'{st[0]}{st[1]}']) for st in SITES)
    pa = rho >= 0.70
    pb = rho - S1837_DEPTH_RHO >= 0.05
    pc = rho_p > rho
    pd = (landed > 0.001 and exdrift <= 0.02 and ncov == NCOV
          and abs(base - S1789_LIVE_TOP1_PP['skip7000']) <= 0.001
          and abs(base2 - S1789_LIVE_TOP1_PP['skip11000']) <= 0.001)

    print(f'\n  SENSITIVITY (top-1 pp lost to {REL:.0%} noise, mean of {len(SEEDS)} seeds):',
          flush=True)
    for k in ('attn', 'mlp'):
        print(f'    {k:4s} ' + ' '.join(
            f'L{L}:{sens[(k, L)]:.3%}' for L in range(1, 18)), flush=True)
    print(f'\n  RMS table error / RMS output:', flush=True)
    for k in ('attn', 'mlp'):
        print(f'    {k:4s} ' + ' '.join(
            f'L{L}:{relerr[(k, L)]:.3f}' for L in range(1, 18)), flush=True)
    print(f'\n  dearest 6 by §1834 cost: ' + '  '.join(
        f'{nm} cost {S1834_COST[nm]:+.1%} sens {sens[SITES[names.index(nm)]]:.2%}'
        for nm in sorted(names, key=lambda x: -S1834_COST[x])[:6]), flush=True)
    print(f'  cheapest 6:              ' + '  '.join(
        f'{nm} cost {S1834_COST[nm]:+.1%} sens {sens[SITES[names.index(nm)]]:.2%}'
        for nm in sorted(names, key=lambda x: S1834_COST[x])[:6]), flush=True)
    print(f'\n  SENSITIVITY predicts §1834 cost (Spearman >=+0.70) -> {pa}  rho {rho:+.4f}',
          flush=True)
    print(f'  and it BEATS DEPTH (§1837 published {S1837_DEPTH_RHO:+.4f}) by >=0.05 -> {pb}  '
          f'margin {rho - S1837_DEPTH_RHO:+.4f}', flush=True)
    print(f'  and the FIRST-ORDER PRODUCT beats sensitivity alone -> {pc}  '
          f'sens x err {rho_p:+.4f} vs sens {rho:+.4f}', flush=True)
    print(f'  noise landed everywhere (min {landed:.3%}), §1837 explained reproduces '
          f'(max drift {exdrift:.4f}) -> control {pd}', flush=True)

    json.dump({'run': 'downstream_sensitivity', 'rel_noise': REL, 'seeds': list(SEEDS),
               'sensitivity': {f'{k}{L}': sens[(k, L)] for k, L in SITES},
               'rel_table_error': {f'{k}{L}': relerr[(k, L)] for k, L in SITES},
               'explained': {f'{k}{L}': ex[(k, L)] for k, L in SITES},
               'S1834_cost': S1834_COST, 'live_top1': [base, base2],
               'spearman_sens_vs_cost': rho, 'spearman_product_vs_cost': rho_p,
               'spearman_depth_vs_cost_published': S1837_DEPTH_RHO,
               'min_sensitivity': landed, 'explained_drift_vs_S1837': exdrift,
               'predictions': {'pred_a_sensitivity_predicts_cost': bool(pa),
                               'pred_b_beats_depth': bool(pb),
                               'pred_c_product_beats_sensitivity': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
