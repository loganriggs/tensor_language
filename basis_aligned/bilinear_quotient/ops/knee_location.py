# WHERE IS THE KNEE?  -- and does mlp5 become extreme only beneath a compiled layer 0?
#
# §1840 established the compilation cost is NOT first-order along the substitution direction: the local
# response at alpha = 0.9 rank-predicts the full response at alpha = 0 at only +0.298, which explains why
# eight successive instruments failed to price compilation while depth alone kept winning at +0.853.
# Every one of them was a local or first-order measurement of an effect that is neither.
#
# Two things §1840 could not answer with six alpha points, and LESSON 47 says points chosen to bracket a
# feature do not estimate a curve -- the exact mistake §1829 was built to correct.
#
#   1. WHERE the damage lives. mlp5 costs 0.0088 nats at alpha = 0.5 and 0.0512 at alpha = 0, so most of
#      it is in the last half; that is a bracket, not a knee. This sweeps NINE alphas, dense near 0.
#   2. WHETHER mlp5's primacy is an INTERACTION with a compiled layer 0. §1834 prices mlp5 first by
#      12.7pp beneath a compiled layer 0; §1840 found it NINTH on a fully live model, which forced a
#      scoping correction to §1834's headline. This runs BOTH streams so the interaction is measured
#      rather than inferred.
#
# The B0 stream is built from the SAME empirical per-token means as the sweep -- attn0 and mlp0 held at
# alpha = 0 -- so both streams use one consistent notion of "compiled" and no table is fitted anywhere.
#
# ROLES. skip7000, covered positions. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE KNEE IS LATE: for mlp5 on the B0 stream, more than half the total alpha=0 damage occurs
#          below alpha = 0.2 -- removing the first 80% of the context-dependent component costs less than
#          removing the last 20%. If TRUE compilation becomes expensive at a THRESHOLD, which a compiled
#          program could be designed against rather than merely diagnosed. If FALSE the damage accumulates
#          smoothly, "non-linear" means only "convex", and there is no threshold for a design rule.
#   pred_b AND THE INTERACTION IS REAL: mlp5's alpha=0 CE rise on the B0 stream is at least 2x its
#          PUBLISHED live-stream rise of +0.0512 nats (§1840). If TRUE, §1840's inference is confirmed --
#          mlp5 is unremarkable alone and extreme beneath a compiled layer 0, so the object is an
#          INTERACTION and not a site. If FALSE the interaction is absent and §1834's mlp5 ranking has
#          another cause -- most likely the length-1 table or the top-1 readout, which would then need
#          separating.
#   pred_c AND THE KNEE IS SHARED: the alpha at which each site reaches half its total damage varies by
#          under 0.15 across the five sites on the B0 stream. If TRUE one threshold describes the network
#          and a single design rule follows. If FALSE each site has its own knee and any rule must be
#          per-site, which would make the threshold much less useful even though it exists.
#   pred_d CONTROLS: alpha = 1 reproduces each stream's own base CE to 1e-9 at every site (the exact
#          known-answer check §1840 established); the live-stream alpha=0 values reproduce §1840's
#          PUBLISHED +0.0512 (mlp5), +0.1935 (mlp1), +0.0768 (attn5), +0.1325 (mlp17) within 0.005;
#          §1837's PUBLISHED explained variance reproduces within 0.02; coverage 5419 of 50257.
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
OUT = PT + 'ops/knee_location_results.json'
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




S1837_EXPLAINED = {'attn5': 0.185, 'attn14': 0.104, 'mlp1': 0.560, 'mlp5': 0.290, 'mlp17': 0.604}
# §1840's PUBLISHED live-stream alpha=0 CE rise, covered positions, skip7000
S1840_LIVE_A0 = {'mlp5': 0.0512, 'mlp1': 0.1935, 'attn5': 0.0768, 'mlp17': 0.1325}
EVAL = PT + '.rowcache/fineweb_n192_skip7000.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
SITES = [(k, L) for L in range(0, 18) for k in ('attn', 'mlp')]   # layer 0 included: it forms B0
TARGETS = [('mlp', 5), ('mlp', 1), ('attn', 5), ('mlp', 17), ('attn', 14)]
B0_SITES = [('attn', 0), ('mlp', 0)]
ALPHAS = (1.0, 0.5, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.0)
STATE = {}
COV = {}


def alpha_hook(mu, alpha):
    """y -> mu_token + alpha * (y - mu_token) at COVERED positions. alpha=1 is a no-op by construction."""
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
def token_means(rows):
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
        mu[st][c == 0] = gm.float()
        s[st] = None
    return mu, ex


def half_alpha(cvv, base):
    """The alpha at which half the total damage has accrued, linearly interpolated on the grid."""
    tot = cvv[0.0] - base
    if tot <= 1e-9:
        return float('nan')
    for j in range(1, len(ALPHAS)):
        hi, lo = ALPHAS[j - 1], ALPHAS[j]
        dh, dl = cvv[hi] - base, cvv[lo] - base
        if dl >= 0.5 * tot > dh:
            f = (0.5 * tot - dh) / max(dl - dh, 1e-12)
            return hi + (lo - hi) * f
    return 0.0


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
    print(f'KNEE LOCATION | alphas {ALPHAS}, {len(TARGETS)} sites x LIVE and B0 streams | '
          f'DISCOVERY ONLY', flush=True)

    ev = load(EVAL)
    mu, ex = token_means(ev)
    live_base = covered_ce(ev)
    b0hooks = [(st, alpha_hook(mu[st], 0.0)) for st in B0_SITES]
    b0_base = covered_ce(ev, b0hooks)
    print(f'  live covered CE {live_base:.6f}; B0-stream base {b0_base:.6f} '
          f'(+{b0_base - live_base:.4f}) ({time.time() - t0:.0f}s)', flush=True)

    cv = {'LIVE': {}, 'B0': {}}
    for st in TARGETS:
        cv['LIVE'][st] = {a: covered_ce(ev, [(st, alpha_hook(mu[st], a))]) for a in ALPHAS}
        cv['B0'][st] = {a: covered_ce(ev, b0hooks + [(st, alpha_hook(mu[st], a))]) for a in ALPHAS}
    print(f'  swept ({time.time() - t0:.0f}s)', flush=True)

    nm = {st: f'{st[0]}{st[1]}' for st in TARGETS}
    ha = {s2: {st: half_alpha(cv[s2][st], (live_base if s2 == 'LIVE' else b0_base))
               for st in TARGETS} for s2 in ('LIVE', 'B0')}
    m5 = cv['B0'][('mlp', 5)]
    tot5 = m5[0.0] - b0_base
    below2 = (m5[0.0] - m5[0.2]) / max(tot5, 1e-12)
    live5 = cv['LIVE'][('mlp', 5)][0.0] - live_base
    spread = max(ha['B0'].values()) - min(ha['B0'].values())
    exact = max(abs(cv['LIVE'][st][1.0] - live_base) for st in TARGETS)
    exact_b0 = max(abs(cv['B0'][st][1.0] - b0_base) for st in TARGETS)
    exdrift = max(abs(ex[st] - S1837_EXPLAINED[nm[st]]) for st in TARGETS)
    a0drift = max(abs((cv['LIVE'][st][0.0] - live_base) - S1840_LIVE_A0[nm[st]])
                  for st in TARGETS if nm[st] in S1840_LIVE_A0)
    pa = below2 > 0.50
    pb = tot5 >= 2.0 * S1840_LIVE_A0['mlp5']
    pc = spread < 0.15
    pd = (exact <= 1e-9 and exact_b0 <= 1e-9 and exdrift <= 0.02 and a0drift <= 0.005
          and ncov == NCOV)

    for s2, bs in (('LIVE', live_base), ('B0', b0_base)):
        print(f'\n  {s2} stream, CE above its own base {bs:.5f}:', flush=True)
        for st in TARGETS:
            print(f'    {nm[st]:7s} ' + ' '.join(
                f'a{a}:{cv[s2][st][a] - bs:+.4f}' for a in ALPHAS)
                + f'   half at a={ha[s2][st]:.3f}', flush=True)
    print(f'\n  the KNEE IS LATE for mlp5 on B0 (>50% of damage below a=0.2) -> {pa}  '
          f'{below2:.1%} of {tot5:+.4f} nats', flush=True)
    print(f'  and the INTERACTION is real (B0 mlp5 >= 2x live {S1840_LIVE_A0["mlp5"]:+.4f}) -> {pb}  '
          f'B0 {tot5:+.4f} vs live {live5:+.4f}, ratio {tot5 / max(live5, 1e-9):.2f}x', flush=True)
    print(f'  and the KNEE IS SHARED (half-alpha spread <0.15 on B0) -> {pc}  spread {spread:.3f}   '
          + '  '.join(f'{nm[st]} {ha["B0"][st]:.3f}' for st in TARGETS), flush=True)
    print(f'  alpha=1 exact on both streams ({exact:.1e} / {exact_b0:.1e}), §1840 a0 drift '
          f'{a0drift:.4f}, §1837 drift {exdrift:.4f} -> control {pd}', flush=True)

    json.dump({'run': 'knee_location', 'alphas': list(ALPHAS),
               'live_base': live_base, 'b0_base': b0_base,
               'curves': {s2: {nm[st]: {str(a): cv[s2][st][a] for a in ALPHAS} for st in TARGETS}
                          for s2 in ('LIVE', 'B0')},
               'half_alpha': {s2: {nm[st]: ha[s2][st] for st in TARGETS} for s2 in ('LIVE', 'B0')},
               'mlp5_b0_total': tot5, 'mlp5_live_total': live5,
               'mlp5_fraction_below_alpha_0_2': below2, 'half_alpha_spread_b0': spread,
               'alpha1_dev_live': exact, 'alpha1_dev_b0': exact_b0,
               'S1840_a0_drift': a0drift, 'S1837_explained_drift': exdrift,
               'predictions': {'pred_a_knee_is_late': bool(pa),
                               'pred_b_interaction_is_real': bool(pb),
                               'pred_c_knee_is_shared': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
