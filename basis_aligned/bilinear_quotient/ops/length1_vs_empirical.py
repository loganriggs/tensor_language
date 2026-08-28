# WHY IS mlp5 EXTREME IN §1834?  -- the LENGTH-1 table against the EMPIRICAL per-token mean.
#
# §1834 prices mlp5 first among 34 sites by 12.7pp. §1840 measured single-site damage on a live model
# with the IDEAL per-token table and found mlp5 NINTH, which forced a scoping correction. §1841 then
# tested the cause I had guessed for that -- a compiled layer 0 beneath it -- and measured the
# interaction at only 1.31x against the 2x required, so that attribution was corrected in place too.
#
# Three differences remain between the two measurements, and this run settles the one with a mechanism
# behind it. §1834 substitutes a LENGTH-1 CONTEXT-FREE table: each site's output on a one-token sequence.
# Every alpha curve in §1840/§1841 uses the EMPIRICAL PER-TOKEN MEAN over real contexts. Those are
# different objects and nothing in the record has ever compared them site by site.
#
# The statistic is the frequency-weighted RMS difference between the two tables, relative to each site's
# output RMS -- i.e. how much a single-token forward misses about what the site emits on average for that
# token in real text. §1765 proved a length-1 forward is exactly what the compiled program can see.
#
# ROLES. skip7000, covered positions, 5419 tokens. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE LENGTH-1 TABLE IS WORST AT mlp5: mlp5 has the largest relative length-1-versus-empirical
#          gap of all 34 sites. If TRUE, §1834's ranking is explained -- and it becomes a concrete
#          statement about layer 5's MLP, namely that what it emits for a token in real text is further
#          from what it emits for that token alone than anywhere else in the network. If FALSE, the
#          length-1 table is not specially bad at mlp5 either, every mechanism-bearing candidate is
#          exhausted, and mlp5's primacy in §1834 would rest on the top-1 readout or the all-positions
#          population -- i.e. it would be a scoring artefact rather than a fact about the site, and
#          §1834's headline would need more than the scoping it already carries.
#   pred_b AND THE LENGTH-1 ERROR PREDICTS COST BETTER THAN THE IDEAL ERROR: Spearman(length-1 relative
#          error, §1834 cost) exceeds Spearman(ideal relative error, cost). §1837 established the ideal
#          table's error does not predict cost (+0.466, wrong sign). If the length-1 error does better,
#          the predictor was right and the TABLE was wrong. If FALSE, neither table's error predicts
#          cost and §1837's conclusion generalises to the object §1834 actually used.
#   pred_c AND THE TWO TABLES DIFFER SUBSTANTIALLY EVERYWHERE: the median site's gap exceeds 0.50 of its
#          output RMS. If FALSE the two tables are nearly the same object, none of §1834's figures
#          depend on which is used, and §1840/§1841's ideal-table curves transfer to §1834's arms
#          directly -- which would be convenient and would make pred_a's answer moot.
#   pred_d CONTROLS: §1837's PUBLISHED token-explained variance reproduces within 0.02 at every site;
#          live top-1 reproduces §1789's PUBLISHED 39.32%; every gap is strictly positive (the two
#          tables are never identical, which they cannot be); coverage 5419 of 50257.
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
OUT = PT + 'ops/length1_vs_empirical_results.json'
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





# §1834's PUBLISHED single-site costs, skip7000, pp of gap lost against B0
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
S1789_LIVE_TOP1 = 0.3932
EVAL = PT + '.rowcache/fineweb_n192_skip7000.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
SITES = [(k, L) for L in range(1, 18) for k in ('attn', 'mlp')]
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


@torch.no_grad()
def length1_rows(toks):
    """Each site's output on a LENGTH-1 sequence -- §1834's context-free table, exactly as built there."""
    out = {st: torch.zeros(NCOV, D, device=DEV) for st in SITES}
    cap = {}

    def mk(st):
        def hook(mod, args, o):
            cap[st] = (o[0] if isinstance(o, tuple) else o)[:, 0].float()
            return None
        return hook
    for i in range(0, NCOV, 256):
        t = toks[i:i + 256].unsqueeze(1)
        forward_logits(t, [(st, mk(st)) for st in SITES])
        for st in SITES:
            out[st][i:i + t.shape[0]] = cap[st]
    return out


@torch.no_grad()
def empirical_means(rows):
    """Per-token mean over REAL contexts, plus per-token counts and each site's RMS output scale."""
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
    mu, ex, rms = {}, {}, {}
    for st in SITES:
        gm = g[st] / nn
        between = float((s[st] * s[st]).sum(1).div(cc).sum()) / nn - float(gm @ gm)
        total = q[st] / nn - float(gm @ gm)
        ex[st] = max(min(between / max(total, 1e-12), 1.0), 0.0)
        rms[st] = (q[st] / nn) ** 0.5
        mu[st] = (s[st] / cc.unsqueeze(1)).float()
        s[st] = None
    return mu, ex, rms, c


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
    toks = seen.nonzero(as_tuple=True)[0]
    idmap = torch.zeros(V, dtype=torch.long, device=DEV)
    idmap[toks] = torch.arange(NCOV, device=DEV)
    COV['idmap'] = idmap
    print(f'LENGTH-1 TABLE vs EMPIRICAL MEAN | 34 sites, {NCOV} covered tokens | DISCOVERY ONLY',
          flush=True)

    ev = load(EVAL)
    mu, ex, rms, cnt = empirical_means(ev)
    t1 = live_top1(ev)
    l1 = length1_rows(toks)
    print(f'  both tables built; live top-1 {t1:.2%} ({time.time() - t0:.0f}s)', flush=True)

    w = cnt / cnt.sum().clamp_min(1.0)         # frequency weights over the covered tokens
    gap, l1err, idealerr = {}, {}, {}
    for st in SITES:
        d2 = ((l1[st].double() - mu[st].double()) ** 2).sum(1)
        gap[st] = float(((d2 * w).sum()) ** 0.5) / max(rms[st], 1e-12)
        # RMS error of each table against the site's actual output, relative to its output RMS
        idealerr[st] = (1.0 - ex[st]) ** 0.5 * ((rms[st] ** 2 - 0.0) ** 0.5) / max(rms[st], 1e-12)
        l1err[st] = (max(idealerr[st] ** 2 + gap[st] ** 2, 0.0)) ** 0.5

    names = [f'{k}{L}' for k, L in SITES]
    cost = [S1834_COST[nm] for nm in names]
    gv = [gap[st] for st in SITES]
    iv = [idealerr[st] for st in SITES]
    lv = [l1err[st] for st in SITES]
    rho_gap = spearman(gv, cost)
    rho_ideal = spearman(iv, cost)
    rho_l1 = spearman(lv, cost)
    worst = names[max(range(len(gv)), key=lambda i: gv[i])]
    med = sorted(gv)[len(gv) // 2]
    exdrift = max(abs(ex[st] - S1837_EXPLAINED[f'{st[0]}{st[1]}']) for st in SITES)
    pa = worst == 'mlp5'
    pb = rho_l1 > rho_ideal
    pc = med > 0.50
    pd = (exdrift <= 0.02 and ncov == NCOV and abs(t1 - S1789_LIVE_TOP1) <= 0.001
          and all(v > 0.0 for v in gv))

    print(f'\n  LENGTH-1 minus EMPIRICAL MEAN, frequency-weighted RMS / site output RMS:', flush=True)
    for k in ('attn', 'mlp'):
        print(f'    {k:4s} ' + ' '.join(
            f'L{L}:{gap[(k, L)]:.3f}' for L in range(1, 18)), flush=True)
    print(f'\n  largest 6: ' + '  '.join(
        f'{nm} {gap[SITES[names.index(nm)]]:.3f} (cost {S1834_COST[nm]:+.1%})'
        for nm in sorted(names, key=lambda x: -gap[SITES[names.index(x)]])[:6]), flush=True)
    print(f'\n  the LENGTH-1 table is worst at mlp5 -> {pa}  largest is {worst} at {max(gv):.3f}; '
          f'mlp5 at {gap[("mlp", 5)]:.3f}', flush=True)
    print(f'  and the LENGTH-1 error predicts §1834 cost better than the IDEAL error -> {pb}  '
          f'length-1 {rho_l1:+.4f} vs ideal {rho_ideal:+.4f} (gap alone {rho_gap:+.4f})', flush=True)
    print(f'  and the two tables differ substantially (median gap >0.50) -> {pc}  median {med:.3f}',
          flush=True)
    print(f'  §1837 explained reproduces (drift {exdrift:.4f}), live top-1 {t1:.2%}, coverage {ncov} '
          f'-> control {pd}', flush=True)

    json.dump({'run': 'length1_vs_empirical', 'n_sites': len(SITES),
               'length1_minus_empirical_rel': {f'{k}{L}': gap[(k, L)] for k, L in SITES},
               'ideal_table_rel_error': {f'{k}{L}': idealerr[(k, L)] for k, L in SITES},
               'length1_rel_error': {f'{k}{L}': l1err[(k, L)] for k, L in SITES},
               'S1834_cost': S1834_COST, 'live_top1': t1,
               'spearman_gap_vs_cost': rho_gap, 'spearman_ideal_vs_cost': rho_ideal,
               'spearman_length1_vs_cost': rho_l1,
               'largest_gap_site': worst, 'median_gap': med,
               'explained_drift_vs_S1837': exdrift,
               'predictions': {'pred_a_length1_worst_at_mlp5': bool(pa),
                               'pred_b_length1_predicts_better': bool(pb),
                               'pred_c_tables_differ_substantially': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
