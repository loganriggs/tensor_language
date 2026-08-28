# THE SUBSTITUTION-DIRECTION CURVE -- perturb along the error that ACTUALLY occurs.
#
# §1839 measured each site's sensitivity to a 50% RANDOM perturbation and it did not explain §1834's
# cost table: Spearman +0.464, against depth's +0.853. The decisive contrast: `attn5` is the most
# noise-sensitive site in the network (0.8646 nats) and costs +16.9pp to compile; `mlp5` is 53x LESS
# noise-sensitive (0.0162) and costs +61.2pp, 3.6x more.
#
# §1839 named why: a context-free table does not add noise. It replaces a site's output with its
# TOKEN-CONDITIONAL MEAN, deleting exactly the context-dependent component and leaving the rest intact.
# That error is systematic and low-dimensional, and random probing was always going to miss it.
#
# This perturbs along the real direction. For each site, replace its output with
#     mu_token + alpha * (y - mu_token)
# and sweep alpha from 1 (live, exactly) down to 0 (the site is its per-token mean). alpha = 0 is the
# BEST POSSIBLE per-token table -- the empirical conditional mean over the eval distribution -- which is
# a strictly better substitute than §1834's length-1 context-free table, so the alpha=0 column is an
# IDEALISED single-site cost and is not the same object as §1834's arm. Stated, not buried.
#
# SCOPE: the interpolation is applied at COVERED positions only, where mu_token is defined, and CE is
# scored on those same positions. Uncovered positions run live in every arm, so they contribute nothing
# to any difference.
#
# alpha = 1 is a KNOWN-ANSWER control (LESSON 34/46): the curve must pass through the live model
# EXACTLY, which is the landing check §1839's pred_d could not make.
#
# ROLES. skip7000. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44, confound controlled per LESSON 46:
#   pred_a THE COST IS FIRST-ORDER ALONG THE SUBSTITUTION DIRECTION: the Spearman between the LOCAL
#          response (CE at alpha=0.9 minus CE at alpha=1) and the FULL response (CE at alpha=0 minus CE
#          at alpha=1), over the 34 sites, is at least +0.70. If TRUE a local derivative prices a site
#          and compilation cost is a first-order effect. If FALSE the damage appears only at large
#          displacement -- the cost is NON-LINEAR in how much context-dependence is removed, no local
#          instrument can price it however well aimed, and that would explain why every instrument since
#          §1824 has failed while depth kept winning.
#   pred_b AND THE DIRECTIONAL DERIVATIVE BEATS DEPTH: Spearman(local response, §1834 PUBLISHED cost)
#          exceeds depth's PUBLISHED +0.8527 by at least 0.05. This is the sixth candidate explanation of
#          the cost table. If it fails too, depth stands alone against norm, direction, second moment,
#          cross-position structure, token-explained variance, random sensitivity and now the aimed
#          derivative -- and the right conclusion would be that the §1834 table is a statement about
#          position in the stack, full stop.
#   pred_c AND THE IDEAL TABLE STILL SINGLES OUT mlp5: mlp5's alpha=0 CE increase is the largest of all
#          34 sites. §1834 ranks it first by 12.7pp using a length-1 table scored in top-1 gap fraction;
#          this re-ranks with a strictly better table scored in CE nats. If FALSE, mlp5's primacy is an
#          artefact of the context-free table's particular weakness rather than a property of the site,
#          and §1834's headline needs qualifying.
#   pred_d CONTROLS: alpha=1 reproduces the live covered CE to within 1e-6 at EVERY site -- an exact
#          known-answer check, and the landing bar §1839 could not certify; the CE curve is monotone
#          non-decreasing as alpha falls at every site; §1837's PUBLISHED token-explained variance
#          reproduces within 0.02; live top-1 reproduces §1789's PUBLISHED 39.32%; coverage 5419.
# DOWNSTREAM SENSITIVITY, CE READOUT -- §1838's void run repeated with a probe that can land.
#
# §1838 ran this sweep with a 10% relative perturbation read out in TOP-1 and its own control
# failed: the minimum sensitivity was -0.009%, the median site moved 0.03pp (eleven tokens of
# 36,864), and the resulting Spearman of -0.234 was a correlation over noise. Two faults. The
# magnitude was 3-9x below the MEASURED real table error (15-93% of each site's output RMS,
# §1838's own table). And top-1 is a thresholded readout with a discreteness floor, so a
# perturbation must flip an argmax before it registers at all.
#
# This repeats it with CROSS-ENTROPY, which is continuous and has no floor, and with the noise at
# 50% relative -- inside the measured band rather than an order of magnitude below it. pred_d's
# bar is restated in nats so the probe can still be shown NOT to land.
#
# ORIGINAL FRAMING, unchanged, since §1838 tested none of it:
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
OUT = PT + 'ops/substitution_direction_curve_results.json'
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





S1834_COST = {
    'attn1': 0.374, 'attn2': 0.429, 'attn3': 0.199, 'attn4': 0.288, 'attn5': 0.169, 'attn6': 0.047,
    'attn7': 0.074, 'attn8': 0.011, 'attn9': 0.007, 'attn10': -0.006, 'attn11': 0.023,
    'attn12': -0.003, 'attn13': 0.006, 'attn14': 0.003, 'attn15': -0.011, 'attn16': -0.001,
    'attn17': -0.005,
    'mlp1': 0.387, 'mlp2': 0.379, 'mlp3': 0.485, 'mlp4': 0.445, 'mlp5': 0.612, 'mlp6': 0.065,
    'mlp7': 0.051, 'mlp8': 0.041, 'mlp9': 0.041, 'mlp10': 0.025, 'mlp11': 0.029, 'mlp12': 0.023,
    'mlp13': 0.021, 'mlp14': 0.022, 'mlp15': 0.017, 'mlp16': 0.019, 'mlp17': 0.040}
S1837_EXPLAINED = {
    'attn1': 0.544, 'attn2': 0.252, 'attn3': 0.228, 'attn4': 0.190, 'attn5': 0.185, 'attn6': 0.290,
    'attn7': 0.255, 'attn8': 0.164, 'attn9': 0.230, 'attn10': 0.238, 'attn11': 0.177,
    'attn12': 0.164, 'attn13': 0.131, 'attn14': 0.104, 'attn15': 0.152, 'attn16': 0.124,
    'attn17': 0.459,
    'mlp1': 0.560, 'mlp2': 0.407, 'mlp3': 0.363, 'mlp4': 0.255, 'mlp5': 0.290, 'mlp6': 0.241,
    'mlp7': 0.224, 'mlp8': 0.233, 'mlp9': 0.221, 'mlp10': 0.218, 'mlp11': 0.213, 'mlp12': 0.224,
    'mlp13': 0.230, 'mlp14': 0.213, 'mlp15': 0.320, 'mlp16': 0.661, 'mlp17': 0.604}
S1837_DEPTH_RHO = 0.8527
S1789_LIVE_TOP1_PP = {'skip7000': 0.3932}
EVAL = PT + '.rowcache/fineweb_n192_skip7000.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
SITES = [(k, L) for L in range(1, 18) for k in ('attn', 'mlp')]
ALPHAS = (1.0, 0.9, 0.75, 0.5, 0.25, 0.0)
STATE = {}
COV = {}


def spearman(a, b):
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


def alpha_hook(mu, alpha):
    """y -> mu_token + alpha * (y - mu_token), at COVERED positions only.

    alpha = 1 leaves the tensor untouched by construction, which is what makes the alpha=1 arm an exact
    known-answer control rather than an approximate one."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        if alpha == 1.0:
            return None
        idx = STATE['idx']
        cov = COV['seen'][idx]
        r = COV['idmap'][idx]
        m2 = mu[r.reshape(-1)].reshape(y.shape).to(y.dtype)
        blend = m2 + (y - m2) * alpha
        y2 = torch.where(cov.unsqueeze(-1), blend, y)
        return (y2,) + tuple(out[1:]) if isinstance(out, tuple) else y2
    return hook


@torch.no_grad()
def covered_ce(rows, hooks=()):
    """CE on COVERED scored positions -- the population the interpolation is defined on."""
    tot, n = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        lg = forward_logits(idx, hooks)[:, 64:].float()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                            reduction='none').reshape(tg.shape).double()
        c = COV['seen'][idx[:, 64:]]
        tot += float(e[c].sum()); n += int(c.sum())
    return tot / max(n, 1)


@torch.no_grad()
def live_top1(rows):
    hit = tot = 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        lg = forward_logits(bb[:, :-1].to(DEV).contiguous())[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        hit += int((lg.argmax(-1) == tg).sum()); tot += int(tg.numel())
    return hit / max(tot, 1)


@torch.no_grad()
def token_means(rows):
    """Empirical per-token mean output per site, plus §1837's explained fraction as a control."""
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
            COV['rid'] = COV['idmap'][sub][COV['cov']]
            forward_logits(idx)
    finally:
        for hd in hs:
            hd.remove()
    nn = max(n['k'], 1)
    cc = c.clamp_min(1.0)
    mu, ex = {}, {}
    for st in SITES:
        gm = g[st] / nn
        between = float((s[st] * s[st]).sum(1).div(cc).sum()) / nn - float(gm @ gm)
        total = q[st] / nn - float(gm @ gm)
        ex[st] = max(min(between / max(total, 1e-12), 1.0), 0.0)
        mu[st] = (s[st] / cc.unsqueeze(1)).float()
        # tokens never seen in THIS eval fall back to the global mean rather than to zero
        mu[st][c == 0] = gm.float()
        s[st] = None
    return mu, ex


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
    COV['idmap'] = idmap
    print(f'SUBSTITUTION-DIRECTION CURVE | alpha {ALPHAS}, 34 sites, covered positions | '
          f'DISCOVERY ONLY', flush=True)

    ev = load(EVAL)
    mu, ex = token_means(ev)
    base = covered_ce(ev)
    t1 = live_top1(ev)
    print(f'  per-token means built; live covered CE {base:.6f}, live top-1 {t1:.2%} '
          f'({time.time() - t0:.0f}s)', flush=True)

    curve = {}
    for st in SITES:
        curve[st] = {a: covered_ce(ev, [(st, alpha_hook(mu[st], a))]) for a in ALPHAS}
    print(f'  curve swept ({time.time() - t0:.0f}s)', flush=True)

    names = [f'{k}{L}' for k, L in SITES]
    local = [curve[st][0.9] - curve[st][1.0] for st in SITES]
    full = [curve[st][0.0] - curve[st][1.0] for st in SITES]
    cost = [S1834_COST[nm] for nm in names]
    rho_lf = spearman(local, full)
    rho_lc = spearman(local, cost)
    rho_fc = spearman(full, cost)
    worst = names[max(range(len(full)), key=lambda i: full[i])]
    exact = max(abs(curve[st][1.0] - base) for st in SITES)
    mono = all(curve[st][ALPHAS[j]] >= curve[st][ALPHAS[j - 1]] - 1e-9
               for st in SITES for j in range(1, len(ALPHAS)))
    exdrift = max(abs(ex[st] - S1837_EXPLAINED[f'{st[0]}{st[1]}']) for st in SITES)
    pa = rho_lf >= 0.70
    pb = rho_lc - S1837_DEPTH_RHO >= 0.05
    pc = worst == 'mlp5'
    pd = (exact <= 1e-6 and mono and exdrift <= 0.02 and ncov == NCOV
          and abs(t1 - S1789_LIVE_TOP1_PP['skip7000']) <= 0.001)

    print(f'\n  CE ABOVE LIVE along the substitution direction (covered CE {base:.5f}):', flush=True)
    for k in ('attn', 'mlp'):
        for a in (0.9, 0.5, 0.0):
            print(f'    {k:4s} a={a:<4} ' + ' '.join(
                f'L{L}:{curve[(k, L)][a] - base:+.4f}' for L in range(1, 18)), flush=True)
    print(f'\n  dearest 6 by §1834 cost: ' + '  '.join(
        f'{nm} cost {S1834_COST[nm]:+.1%} a0 {full[names.index(nm)]:+.4f} '
        f'a.9 {local[names.index(nm)]:+.4f}'
        for nm in sorted(names, key=lambda x: -S1834_COST[x])[:6]), flush=True)
    print(f'\n  the cost is FIRST-ORDER along the direction (rho local vs full >=+0.70) -> {pa}  '
          f'rho {rho_lf:+.4f}', flush=True)
    print(f'  and the DERIVATIVE beats depth (§1837 {S1837_DEPTH_RHO:+.4f}) -> {pb}  '
          f'local vs cost {rho_lc:+.4f}, margin {rho_lc - S1837_DEPTH_RHO:+.4f} '
          f'(full vs cost {rho_fc:+.4f})', flush=True)
    print(f'  and the IDEAL table still singles out mlp5 -> {pc}  largest a=0 rise is {worst} '
          f'at {max(full):+.4f}; mlp5 at {full[names.index("mlp5")]:+.4f}', flush=True)
    print(f'  alpha=1 EXACT (max dev {exact:.2e}), curve monotone {mono}, §1837 drift '
          f'{exdrift:.4f} -> control {pd}', flush=True)

    json.dump({'run': 'substitution_direction_curve', 'alphas': list(ALPHAS),
               'live_covered_ce': base, 'live_top1': t1,
               'curve': {f'{k}{L}': {str(a): curve[(k, L)][a] for a in ALPHAS} for k, L in SITES},
               'local_response': dict(zip(names, local)),
               'full_response': dict(zip(names, full)),
               'S1834_cost': S1834_COST,
               'spearman_local_vs_full': rho_lf, 'spearman_local_vs_cost': rho_lc,
               'spearman_full_vs_cost': rho_fc, 'largest_full_response_site': worst,
               'alpha1_max_deviation': exact, 'monotone': bool(mono),
               'explained_drift_vs_S1837': exdrift,
               'predictions': {'pred_a_first_order': bool(pa),
                               'pred_b_derivative_beats_depth': bool(pb),
                               'pred_c_ideal_table_singles_out_mlp5': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
