# IS THE DEEP SUFFIX FREE IN AGGREGATE?  -- the first shot in this arc at a POSITIVE result.
#
# §1831 swept a single compiled site across depth on top of a compiled layer 0 and found the cost is a
# narrow EARLY band: +attn at L1 27.4%, L2 21.9% (the worst), L3 44.9%, L5 47.9% -- then FREE from L9 up,
# at 64.1% / 64.2% / 65.3% against B0's 64.8%. Site type is interchangeable at every depth tested
# (1.3 / 3.4 / 4.5pp), so this is not about the cross-position path.
#
# Free ONE AT A TIME is not free TOGETHER. Every deep-site figure in §1831 is one perturbation that a
# fully live suffix absorbed; nine layers of them is a different object. This sweeps the boundary:
# compile layer 0 plus EVERY site from layer K to 17, for K in 13, 11, 9, 7, 5, 3.
#
# The stake is a positive claim rather than another closed account. TOP9 compiles 20 of the 36 sites --
# over half the program. If it holds most of the gap, the entire compilation cost lives in layers 1-8,
# and more than half of bilin18's sites are replaceable by per-token context-free tables for nothing.
#
# ROLES. All three. FULL RANK, settled program. §1824's sequential gain correction over each arm's
# LIVE-ATTENTION set. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44, ratio-not-share per LESSONS 57, nesting checked per
# LESSONS 58 (TOP13 c TOP11 c TOP9 c TOP7 c TOP5 c TOP3 is a genuine chain, each a superset of the last):
#   pred_a THE DEEP SUFFIX IS FREE IN AGGREGATE: TOP9 -- layer 0 plus every site from layer 9 to 17,
#          20 of 36 sites -- retains at least 50% of the gap on skip7000, against B0's 64.8%. If TRUE,
#          over half the network is compilable at little cost and the whole of the compilation penalty
#          lives in layers 1-8, which is a compression result rather than a diagnosis. If FALSE, deep
#          sites are absorbable singly but not jointly, and §1831's free deep sites were measuring the
#          live suffix's tolerance for ONE perturbation, not a redundant region.
#   pred_b AND THE FRONTIER IS SHARP: the largest drop between consecutive K in the sweep exceeds 20pp.
#          If FALSE recovery declines gradually with each layer added, there is no boundary to name, and
#          the cost is a dose-response in the NUMBER of compiled sites rather than a property of where
#          they sit.
#   pred_c AND THE AGGREGATE MATCHES THE SINGLE-SITE PROFILE: that largest drop occurs at a K in the
#          closed range 5..9, the band where §1831's single-site cost turns on (L5 47.9% -> L9 64.1%).
#          If FALSE the aggregate frontier sits somewhere the single-site sweep did not predict, which
#          means the compiled sites INTERACT and no per-site profile can price a region.
#   pred_d CONTROLS, cross-run per LESSON 42: B0 and B1 reproduce §1829's PUBLISHED sequential gap
#          fractions (skip7000 64.8% / 25.9%) within 0.03; B0+attn9 reproduces §1831's PUBLISHED 64.1%
#          within 0.03; B0 reproduces §1806's PUBLISHED raw 37.4%. Endpoints must reproduce §1789's
#          FULL-RANK top-1; the placement control must move top-1 under 0.05pp. Coverage 5419 of 50257.
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
OUT = PT + 'ops/deep_suffix_frontier_results.json'
CALROWS = 32   # rows used for the per-layer calibration passes; the eval uses the full role
PROBE_LS = ()
KIND_LS = ()
RANKS = (64,)                # the settled table rank; only the STREAM matters here
B0_SITES = [('attn', 0), ('mlp', 0)]
KS = (13, 11, 9, 7, 5, 3)             # compile layers K..17 on top of layer 0; a NESTED chain
ARMS = {'B0': list(B0_SITES), 'B1': B0_SITES + [('attn', 1), ('mlp', 1)],
        'B0_attn9': B0_SITES + [('attn', 9)]}
for _K in KS:
    ARMS[f'TOP{_K}'] = B0_SITES + [(k2, L2) for L2 in range(_K, 18) for k2 in ('attn', 'mlp')]
# §1831's PUBLISHED single deep-site sequential gap fraction, the cross-run anchor for this run
S1831_ATTN9_GAPFRAC = {'skip7000': 0.641, 'skip11000': 0.647, 'skip1200': 0.607}
# §1829's PUBLISHED sequential gap fractions at the two arms this run overlaps
S1829_SEQ_GAPFRAC = {'B0': {'skip7000': 0.648, 'skip11000': 0.653, 'skip1200': 0.613},
                     'B1': {'skip7000': 0.259, 'skip11000': 0.264, 'skip1200': 0.258}}
# §1806's PUBLISHED raw gap fraction at B0
S1806_B0_RAW = {'skip7000': 0.3742, 'skip11000': 0.3960, 'skip1200': 0.3809}
POKE_MAG = 10.0              # §1765's magnitude, kept so the live side is comparable across runs
POKE_ROWS = 8                # one batch; the matrix is a per-position mean over these rows
KSRC = (64, 88, 112, 136, 160, 184)   # source positions, all >=64 so every reader is scored
S1765_LIVE_REACH_NATS = 0.118         # §1765's PUBLISHED live-model figure, the positive control
# §1825's PUBLISHED L9 cosines in the corrected arms -- the anti-aligned layer Codex surfaced
S1825_L9_COSINE = {3: -0.134, 5: -0.628}
# §1806's PUBLISHED gap fractions, FULL-RANK build, bottom-up (0..L compiled, L+1..17 live).
S1806_FULLRANK_GAPFRAC = {0: {'skip7000': 0.3742, 'skip11000': 0.3960, 'skip1200': 0.3809},
                          3: {'skip7000': -0.4484, 'skip11000': -0.4316, 'skip1200': -0.4688},
                          5: {'skip7000': -0.4391, 'skip11000': -0.4321, 'skip1200': -0.4643}}
# FULL-RANK all-sites constants (§1789) -- the object THIS build produces. LESSON 53.
S1789_FULLRANK_TOP1_PP = {'skip7000': 0.1355, 'skip11000': 0.1425, 'skip1200': 0.1364}
S1789_LIVE_TOP1_PP = {'skip7000': 0.3932, 'skip11000': 0.4235, 'skip1200': 0.3888}
# §1822's PUBLISHED gap fractions using FULLY-COMPILED-measured gains on all live layers
S1822_GLOBALGAIN_GAPFRAC = {0: {'skip7000': -0.041, 'skip11000': -0.031, 'skip1200': -0.027},
                            3: {'skip7000': 0.103, 'skip11000': 0.100, 'skip1200': 0.120},
                            5: {'skip7000': 0.115, 'skip11000': 0.112, 'skip1200': 0.119}}
# §1823's PUBLISHED depth-matched (first-order) gap fractions
S1823_MATCHED_GAPFRAC = {0: {'skip7000': 0.557, 'skip11000': 0.567, 'skip1200': 0.531},
                         3: {'skip7000': -0.226, 'skip11000': -0.226, 'skip1200': -0.239},
                         5: {'skip7000': -0.150, 'skip11000': -0.142, 'skip1200': -0.151}}
# §1824's PUBLISHED sequentially-calibrated gap fractions
S1824_SEQ_GAPFRAC = {0: {'skip7000': 0.648, 'skip11000': 0.653, 'skip1200': 0.613},
                     3: {'skip7000': 0.119, 'skip11000': 0.116, 'skip1200': 0.120},
                     5: {'skip7000': 0.123, 'skip11000': 0.121, 'skip1200': 0.128}}
# §1819's PUBLISHED single-interface cosine for head 5.7 -- the cross-run anchor
S1819_L5H7_COSINE = 0.9990
NATIVE_PER_LAYER = 15.926e6 + 7.963e6   # §1754 accounting, one MLP + one attn
# the map is built at the CARRYABLE rank min(64, r+1) -- §1814: rank(Ws) <= r+1, so anything
# higher is the same matrix at a higher price
# UNITS IN THE NAME (LESSON 49): these are PERCENTAGE-POINT deltas over the all-compiled arm,
# not gap fractions. §1811's pred_d failed by comparing the first against the second.
# PERCENTAGE-POINT top-1 accuracies (units in the name, LESSON 49)
S1813_T1_PP = {'skip7000': 0.0990, 'skip11000': 0.1065, 'skip1200': 0.1007}
S1786_T64_PP = {'skip7000': 0.1288, 'skip11000': 0.1349, 'skip1200': 0.1289}
# ALL-POSITION CE nats for the settled rank-64 program (§1786)
S1786_T64_CE = {'skip7000': 6.17330, 'skip11000': 6.15261, 'skip1200': 6.14463}
# §1811's PERCENTAGE-POINT top-1, the arms whose dominance is being re-tested on CE
# §1788 measured these THREE INSTRUMENTS on the all-sites program only.
S1788_AGREE = {'skip7000': 0.2357, 'skip11000': 0.2271, 'skip1200': 0.2421}
S1788_KL = {'skip7000': 2.88031, 'skip11000': 3.04866, 'skip1200': 2.75451}
S1816_CE_R64 = {-1: {'skip7000': 6.1733, 'skip11000': 6.1526, 'skip1200': 6.1446},
                13: {'skip7000': 4.1815, 'skip11000': 3.9736, 'skip1200': 4.2836}}
LIVE_CE_ALL = {'skip7000': 3.13704, 'skip11000': 2.93450, 'skip1200': 3.23027}
S1807_SCALED_L5 = {'skip7000': 0.189, 'skip11000': 0.177, 'skip1200': 0.183}
S1808_SWING = {'skip7000': {3: -0.001, 5: 0.628}, 'skip11000': {3: 0.001, 5: 0.609},
               'skip1200': {3: -0.001, 5: 0.647}}
S1806_BOTUP = {'skip7000': {0: 0.374, 3: -0.448, 5: -0.439},
               'skip11000': {0: 0.396, 3: -0.432, 5: -0.423},
               'skip1200': {0: 0.381, 3: -0.469, 5: -0.455}}
S1802_ONLY = {'skip7000': {4: -0.0003, 5: -0.1254, 6: -0.1115, 13: -0.0175},
              'skip11000': {4: -0.0006, 5: -0.1311, 6: -0.1198, 13: -0.0173},
              'skip1200': {4: -0.0003, 5: -0.1267, 6: -0.1160, 13: -0.0203}}
S1789_GAP = {'skip7000': 0.3932 - 0.1355, 'skip11000': 0.4235 - 0.1425,
             'skip1200': 0.3888 - 0.1364}
S1795_BG = {'skip7000': 0.12440, 'skip11000': 0.12880, 'skip1200': 0.12250}
TAUS = (10 ** 9, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0)  # DEFER when the bigram's LOO count >= tau
S1796_UNION = {'skip7000': 0.45966, 'skip11000': 0.48224, 'skip1200': 0.47517}
PICK_ROLE = 'skip7000'
ALPHA = 0.01
S1767_FITBIGRAM_CE = {'skip7000': 7.88804, 'skip11000': 7.90729}
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1789_PROG = {'skip7000': 0.1355, 'skip11000': 0.1425, 'skip1200': 0.1364}
S1789_LIVE = {'skip7000': 0.3932, 'skip11000': 0.4235, 'skip1200': 0.3888}
S1790_LOOBG = {'skip7000': 0.1597, 'skip11000': 0.1663, 'skip1200': 0.1800}
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


def row_hook(full_rows, s=1.0):
    """Substitute the per-token row, optionally rescaled.

    The scale is applied AT HOOK TIME rather than by materialising a second [50257, D] bank per site:
    §1807 held a raw and a scaled bank and peaked at 26.4 GiB for no reason. One multiply on the
    gathered rows is identical arithmetic at a thirty-sixth of the memory."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = full_rows[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        if s != 1.0:
            sub = sub * s
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
def bigram_ce(rows, lp_table):
    """Covered-position CE of the fit-row bigram, to control that it is §1767's object."""
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV)[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        cov = COV['seen'][idx]
        r = COV['idmap'][idx].clamp(min=0)
        lp = lp_table[r].gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        tot += float((-lp.double())[cov].sum()); cnt += int(cov.sum())
    return tot / cnt


def gain_hook(g):
    """Scale a whole attention layer's write by g, via c_proj's input.

    §1821 established ONE scalar per layer suffices -- a per-head vector recovered no more than the
    live-norm-weighted layer mean -- so all nine heads scale uniformly, which is exactly scaling the
    layer's attention output."""
    def pre(mod, args):
        return (args[0] * g,) + tuple(args[1:])
    return pre


@torch.no_grad()
def layer_dirs(rows, hooks, gains=()):
    """Mean attention-write VECTOR per layer, so directions can be compared across streams.

    Hooks c_proj's OUTPUT, the live module's write, exactly as layer_norms does -- but accumulates the
    vector, not just its norm. A gain applied via a pre-hook is INSIDE this measurement, so what is
    captured is the corrected write."""
    acc = {L: torch.zeros(D, device=DEV, dtype=torch.float64) for L in range(18)}
    cnt = {L: 0 for L in range(18)}
    sq = {L: 0.0 for L in range(18)}      # SECOND moment, exact and single-pass

    def mk(L):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).detach().double()[:, 64:]
            yf = y.reshape(-1, D)
            acc[L] += yf.sum(0)
            sq[L] += float((yf * yf).sum())
            cnt[L] += y.shape[0] * y.shape[1]
            return None
        return hook

    handles = [H[L].attn.c_proj.register_forward_hook(mk(L)) for L in range(18)]
    handles += [H[L].attn.c_proj.register_forward_pre_hook(gain_hook(g)) for L, g in gains]
    try:
        for j2 in range(0, rows.shape[0], 8):
            forward_logits(rows[j2:j2 + 8, :-1].to(DEV).contiguous(), hooks)
    finally:
        for hd in handles:
            hd.remove()
    mu = {L: acc[L] / max(cnt[L], 1) for L in range(18)}
    # across-position dispersion, the §1819 statistic: sqrt(E||y||^2 - ||E y||^2) / ||E y||
    disp = {L: ((max(sq[L] / max(cnt[L], 1) - float(mu[L] @ mu[L]), 0.0)) ** 0.5)
               / max(float(mu[L].norm()), 1e-9) for L in range(18)}
    return mu, disp


@torch.no_grad()
def layer_norms(rows, hooks):
    """Mean attention-write norm per layer at scored positions, in whatever stream `hooks` gives.

    Hooks c_proj's OUTPUT -- what the LIVE module emits. An outer row_hook replaces attn's return
    value only afterwards, so this reads the same quantity in both streams, as §1804 did."""
    acc = {L: [0.0, 0] for L in range(18)}

    def mk(L):
        def hook(mod, args, out):
            y = (out[0] if isinstance(out, tuple) else out).detach().float()[:, 64:]
            acc[L][0] += float(y.norm(dim=-1).sum())
            acc[L][1] += y.shape[0] * y.shape[1]
            return None
        return hook

    handles = [H[L].attn.c_proj.register_forward_hook(mk(L)) for L in range(18)]
    try:
        for j2 in range(0, rows.shape[0], 8):
            forward_logits(rows[j2:j2 + 8, :-1].to(DEV).contiguous(), hooks)
    finally:
        for hd in handles:
            hd.remove()
    return {L: acc[L][0] / max(acc[L][1], 1) for L in range(18)}


def poke_hook(pos, mag):
    """Add a constant `mag` to every channel of this site's output at ONE position.

    Registered AFTER any substituting hook in the same list, so PyTorch hands it the substituted
    output and the poke lands on whatever the arm actually emits -- §1765's construction exactly."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        y2 = y.clone()
        y2[:, pos, :] = y2[:, pos, :] + mag
        return (y2,) + tuple(out[1:]) if isinstance(out, tuple) else y2
    return hook


@torch.no_grad()
def per_pos_losses(idx, tg, hooks=()):
    """Per-position CE, undivided -- the object a poke's effect is read off."""
    lg = forward_logits(idx, hooks).float()
    return F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                           reduction='none').reshape(tg.shape).double()


@torch.no_grad()
def influence(idx, tg, hooks, site):
    """One row of the finite-difference cross-position Jacobian per source position in KSRC.

    Returns {k: (vector of |dloss| at every q > k, scalar |dloss| at k itself)}. The own-position
    scalar is the control that the poke landed at all; the vector is the measurement."""
    b = per_pos_losses(idx, tg, hooks)
    out = {}
    for k in KSRC:
        d = (per_pos_losses(idx, tg, list(hooks) + [(site, poke_hook(k, POKE_MAG))]) - b).abs()
        out[k] = (d[:, k + 1:].mean(0), float(d[:, k].mean()))
    return out


@torch.no_grad()
def evaluate(rows, hooks, keep_mask):
    """Top-1 overall, on the head, and restricted to positions whose CURRENT token is still covered.

    That last slice is a KNOWN-ANSWER control (LESSON 34). A covered token's table is built from its
    own length-1 forward and the program is position-wise (§1765), so removing OTHER tokens from the
    covered set cannot change what happens at a position that kept its own table. Those numbers must
    be identical across every coverage fraction, not merely close."""
    a = {'n': 0, 'hit': 0, 'head_n': 0, 'head_hit': 0, 'kept_n': 0, 'kept_hit': 0, 'ce': 0.0,
         'kl': 0.0, 'agree': 0}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        lg = forward_logits(idx, hooks)[:, 64:].float()
        a['ce'] += float(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1),
                                         reduction='sum').double())
        ll = forward_logits(idx)[:, 64:].float()
        L = torch.log_softmax(ll, -1)
        P = torch.log_softmax(lg, -1)
        a['kl'] += float((L.exp() * (L - P)).sum(-1).double().sum())
        ap = lg.argmax(-1)
        a['agree'] += int((ap == ll.argmax(-1)).sum())
        h = ap == tg
        del lg, ll, L, P
        hd = COV['freq'][tg] >= 125
        kp = keep_mask[idx[:, 64:]]
        a['n'] += int(tg.numel()); a['hit'] += int(h.sum())
        a['head_n'] += int(hd.sum()); a['head_hit'] += int(h[hd].sum())
        a['kept_n'] += int(kp.sum()); a['kept_hit'] += int(h[kp].sum())
    return {'n': a['n'], 'top1': a['hit'] / max(a['n'], 1),
            'ce_all': a['ce'] / max(a['n'], 1),
            'kl': a['kl'] / max(a['n'], 1), 'agree': a['agree'] / max(a['n'], 1),
            'head_n': a['head_n'], 'top1_head': a['head_hit'] / max(a['head_n'], 1),
            'kept_n': a['kept_n'], 'top1_kept': a['kept_hit'] / max(a['kept_n'], 1)}


def probe_hook(full_rows, store, key):
    """Substitute the row as usual, but record ||live module output|| and ||row|| first.

    This runs inside the FULLY COMPILED stream, so the recorded live output is what the real module
    would emit given compiled inputs -- exactly the quantity the L5/L6 cliff is about."""
    inner = row_hook(full_rows)

    def hook(mod, args, out):
        o = (out[0] if isinstance(out, tuple) else out).detach().float()
        r = inner(mod, args, out)
        rr = (r[0] if isinstance(r, tuple) else r).detach().float()
        s = store.setdefault(key, [0.0, 0.0, 0])
        s[0] += float(o[:, 64:].norm(dim=-1).sum())
        s[1] += float(rr[:, 64:].norm(dim=-1).sum())
        s[2] += int(o[:, 64:].shape[0] * o[:, 64:].shape[1])
        return r
    return hook


def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    T = fit.shape[1] - 1
    full_seen = torch.zeros(V, dtype=torch.bool)
    full_seen[fit[:, :T].reshape(-1).long()] = True
    NFULL = int(full_seen.sum())
    assert NFULL == NCOV, f'coverage {NFULL} != {NCOV}'
    COV['freq'] = torch.bincount(fit[:, 1:T + 1].reshape(-1).long(), minlength=V).to(DEV)
    all_toks = full_seen.nonzero(as_tuple=True)[0]
    # NESTED subsets: one fixed permutation, prefixes of it. 0.125 subset of 0.25 subset of 0.5 ...
    g = torch.Generator().manual_seed(0)
    perm = all_toks[torch.randperm(NFULL, generator=g)]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'DEEP SUFFIX FRONTIER | B0 + every site from layer K to 17 | '
          f'settled program (context-free tables + output-NN fallback + rank-{MAP_RANK} map) | '
          f'DISCOVERY ONLY', flush=True)

    def build(n, rank=None, map_rank=MAP_RANK):
        tk = perm[:n].sort().values.to(DEV)
        seen = torch.zeros(V, dtype=torch.bool, device=DEV)
        seen[tk] = True
        unc = (~seen).nonzero(as_tuple=True)[0]
        # the settled fallback: output-NN neighbour (§1780/§1781), rebuilt over THIS covered set
        lpc = torch.zeros(n, W, device=DEV)
        for i in range(0, n, 256):
            t = tk[i:i + 256].unsqueeze(1)
            lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
        pcn = torch.softmax(lpc, -1)
        pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
        del lpc
        nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
        nnrow[tk] = torch.arange(n, device=DEV)
        for s0 in range(0, unc.numel(), 512):
            u = unc[s0:s0 + 512]
            p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
        del pcn
        torch.cuda.empty_cache()
        tables = {st: torch.zeros(n, D, device=DEV) for st in sites}
        cap = {}

        def mk(st):
            def hook(mod, args, out):
                cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
                return None
            return hook
        for i in range(0, n, 256):
            t = tk[i:i + 256].unsqueeze(1)
            forward_logits(t, [(st, mk(st)) for st in sites])
            for st in sites:
                tables[st][i:i + t.shape[0]] = cap[st]
        if rank is not None:
            # rank-r truncate the COVERED block, then fit the map INSIDE that basis (§1785 -- a map
            # fitted on full-rank tables and reused on truncated ones is not a coherent program)
            for st in sites:
                b = tables[st].double()
                mu = b.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
                tables[st] = (mu + (U[:, :rank] * S[:rank]) @ Vh[:rank]).float()
        # the learned embedding->row map, REFITTED inside this covered set (§1785)
        Ecov = m.transformer.wte.weight.detach()[tk].float().double()
        A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (n / D)
        Eunc = m.transformer.wte.weight.detach()[unc].float().double()
        out = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tables[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mp = (U[:, :map_rank] * S[:map_rank]) @ Vh[:map_rank]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tables[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        del tables, Ecov, Eunc, A
        torch.cuda.empty_cache()
        return out, seen, n

    # the KEPT slice is defined by the SMALLEST coverage set, so it is the same positions at every
    # fraction -- that is what makes the known-answer control meaningful.
    # coverage is fixed at full here, so the 'kept' column is simply accuracy on COVERED
    # positions -- reported for context, not used as a control (there is no coverage variation
    # for a known-answer check to bite on).
    small = perm[:NFULL].to(DEV)
    keep_mask = torch.zeros(V, dtype=torch.bool, device=DEV)
    keep_mask[small] = True

    fr, seen, _ = build(NFULL)

    # ---- per-site NORM CALIBRATION, measured in the FULLY LIVE model.
    # §1804 found the substituted rows are systematically far smaller than what the live modules emit
    # (ratio 2.71 to 152.62 across the attention layers). §1806 showed a compiled layer beneath live
    # layers is catastrophic. If that poisoning is a SCALE mismatch, matching the mean output norm
    # should repair most of it; if it is a direction/content mismatch, scaling will not help.
    live_norm, row_norm = {}, {}

    def measure(st, store):
        def hook(mod, args, out):
            o = (out[0] if isinstance(out, tuple) else out).detach().float()
            s = store.setdefault(st, [0.0, 0])
            s[0] += float(o[:, 64:].norm(dim=-1).sum())
            s[1] += int(o[:, 64:].shape[0] * o[:, 64:].shape[1])
            return None
        return hook
    evs0 = load(EVAL_SETS[0][1])
    lstore = {}
    evaluate(evs0, [(st, measure(st, lstore)) for st in sites], keep_mask)
    for st in sites:
        live_norm[st] = lstore[st][0] / max(lstore[st][1], 1)
    # The row norms must be averaged over THE SAME POSITIONS, not over the 50,257 vocabulary rows:
    # eval positions are frequency-weighted and the vocabulary is not, so a flat row mean would
    # calibrate against a distribution the model never sees. Measured with the substituting hook in
    # place, exactly as §1804's probe did.
    rstore = {}
    evaluate(evs0, [(st, probe_hook(fr[st], rstore, st)) for st in sites], keep_mask)
    del evs0
    scale = {}
    for st in sites:
        rn = rstore[st][1] / max(rstore[st][2], 1)
        row_norm[st] = rn
        scale[st] = live_norm[st] / max(rn, 1e-9)
    print('\n  per-site norm calibration (live mean / row mean), measured on the LIVE model:',
          flush=True)
    for k in ('attn', 'mlp'):
        print(f'    {k:4s} ' + ' '.join(f'L{L}:{scale[(k, L)]:6.2f}' for L in range(18)), flush=True)
    allhooks = {st: row_hook(fr[st]) for st in sites}
    scaledhooks = {st: row_hook(fr[st], scale[st]) for st in sites}
    attnhooks = {st: row_hook(fr[st], scale[st] if st[0] == 'attn' else 1.0) for st in sites}
    mlphooks = {st: row_hook(fr[st], scale[st] if st[0] == 'mlp' else 1.0) for st in sites}
    # the single-site test: mlp4 ALONE rescaled, every other row untouched
    only4 = {st: row_hook(fr[st], scale[st] if st == ('mlp', 4) else 1.0) for st in sites}
    # and its complement: every MLP EXCEPT mlp4
    not4 = {st: row_hook(fr[st], scale[st] if (st[0] == 'mlp' and st[1] != 4) else 1.0)
            for st in sites}
    evs = {e: load(p) for e, p, _ in EVAL_SETS}
    res = {}

    def run(label, hooked, hooks_src=None):
        src = hooks_src if hooks_src is not None else allhooks
        hs = [(st, src[st]) for st in hooked]
        for ename in evs:
            c = evaluate(evs[ename], hs, keep_mask)
            res.setdefault(ename, {})[label] = c
        print(f'  {label:20s} ' + '  '.join(
            f'{e} {res[e][label]["top1"]:6.2%}' for e in evs), flush=True)

    print(f'\n  arms  ({time.time() - t0:.0f}s)', flush=True)
    run('all_substituted', sites)
    run('all_sub_scaled', sites, scaledhooks)
    run('live_model', [])
    # Free the full-rank bank and every hook closure over it BEFORE building another one. The
    # first attempt held `fr` (36 x 50257 x 1152 floats = 8.3 GiB) while build() allocated a second
    # bank of the same size and OOMed at 31.25/31.36 GiB. Each rank is now built, used and released.
    del allhooks, scaledhooks, attnhooks, mlphooks, only4, not4, fr
    torch.cuda.empty_cache()
    frr = build(NFULL)[0]        # FULL RANK, matching §1806 (LESSON 53)
    allh = {st: row_hook(frr[st]) for st in sites}
    evs = {e: load(p) for e, p, _ in EVAL_SETS}
    res = {}

    def run(label, hooked, gains=None):
        hs = [(st, allh[st]) for st in hooked]
        handle = None
        if gains is not None:
            handle = H[gains[0]].attn.c_proj.register_forward_pre_hook(gain_hook(gains[1]))
        try:
            for ename in evs:
                c = evaluate(evs[ename], hs, keep_mask)
                res.setdefault(ename, {})[label] = c
        finally:
            if handle is not None:
                handle.remove()
        print(f'  {label:22s} ' + '  '.join(
            f'{e} {res[e][label]["top1"]:6.2%}' for e in evs), flush=True)

    ev0 = load(EVAL_SETS[0][1])
    ln_live = layer_norms(ev0, [])
    ln_comp = layer_norms(ev0, [(st, allh[st]) for st in sites])
    gain_global = {L: ln_live[L] / max(ln_comp[L], 1e-9) for L in range(18)}

    def run_g(label, hooked, gains):
        hs = [(st, allh[st]) for st in hooked]
        handles = [H[L].attn.c_proj.register_forward_pre_hook(gain_hook(g))
                   for L, g in gains.items()]
        try:
            for ename in evs:
                c = evaluate(evs[ename], hs, keep_mask)
                res.setdefault(ename, {})[label] = c
        finally:
            for hd in handles:
                hd.remove()
        print(f'  {label:22s} ' + '  '.join(
            f'{e} {res[e][label]["top1"]:6.2%}' for e in evs), flush=True)

    print(f'\n  arms  ({time.time() - t0:.0f}s)', flush=True)
    run_g('all_substituted', sites, {})
    run_g('live_model', [], {})
    cal = ev0[:CALROWS]
    gseq = {}

    def seq_gains(botup):
        """§1824's sequential calibration, over the layers whose ATTENTION is still LIVE in this arm.

        A gain on a compiled attention layer is a no-op: the substituting hook replaces the module's
        output, so c_proj's scaled result is discarded. The correctable set is therefore exactly the
        live-attention set, which differs between B0+attn1 (2..17) and B0+mlp1 (1..17)."""
        hs0 = [(st, allh[st]) for st in botup]
        live_attn = [j for j in range(18) if ('attn', j) not in set(botup)]
        g = {}
        for j3 in live_attn:
            handles = [H[q].attn.c_proj.register_forward_pre_hook(gain_hook(gv))
                       for q, gv in g.items()]
            try:
                ln_now = layer_norms(cal, hs0)
            finally:
                for hd in handles:
                    hd.remove()
            g[j3] = ln_live[j3] / max(ln_now[j3], 1e-9)
        return g

    for name, botup in ARMS.items():
        gseq[name] = seq_gains(botup)
        print(f'    {name:10s} live-attn gains L1-L3: ' + '  '.join(
            f'L{j3} {gseq[name][j3]:.4f}' for j3 in sorted(gseq[name])[:3]), flush=True)
        run_g(f'{name}_raw', botup, {})
        run_g(f'{name}_seq', botup, gseq[name])
    run_g('allsub_gain5', sites, {5: gain_global[5]})     # placement control

    del ev0, evs, frr, allh
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    base = {e: res[e]['all_substituted']['top1'] for e in roles}

    def d(e, k2):
        return res[e][k2]['top1'] - base[e]
    gapv = {e: res[e]['live_model']['top1'] - res[e]['all_substituted']['top1'] for e in roles}
    frac = {e: {k2: d(e, k2) / gapv[e] for k2 in res[e]} for e in roles}
    seq = {k: frac[PICK_ROLE][f'{k}_seq'] for k in ARMS}
    nsite = {k: len(v) for k, v in ARMS.items()}
    top = {K: seq[f'TOP{K}'] for K in KS}
    drops = {KS[i]: top[KS[i - 1]] - top[KS[i]] for i in range(1, len(KS))}
    frontier = max(drops, key=drops.get)
    pa = top[9] >= 0.50
    pb = drops[frontier] > 0.20
    pc = 5 <= frontier <= 9
    pd = (all(abs(frac[e]['B0_seq'] - S1829_SEQ_GAPFRAC['B0'][e]) <= 0.03 for e in roles)
          and all(abs(frac[e]['B1_seq'] - S1829_SEQ_GAPFRAC['B1'][e]) <= 0.03 for e in roles)
          and all(abs(frac[e]['B0_attn9_seq'] - S1831_ATTN9_GAPFRAC[e]) <= 0.03 for e in roles)
          and all(abs(frac[e]['B0_raw'] - S1806_B0_RAW[e]) <= 0.03 for e in roles)
          and all(abs(res[e]['all_substituted']['top1'] - S1789_FULLRANK_TOP1_PP[e]) <= 0.001
                  and abs(res[e]['live_model']['top1'] - S1789_LIVE_TOP1_PP[e]) <= 0.001
                  for e in roles)
          and all(abs(d(e, 'allsub_gain5')) < 0.0005 for e in roles)
          and NFULL == NCOV)

    print('\n  gap fraction recovered, all roles:', flush=True)
    for k in ARMS:
        print(f'    {k:10s} {nsite[k]:2d}/36 sites  ' + '  '.join(
            f'{e} raw {frac[e][f"{k}_raw"]:7.1%} seq {frac[e][f"{k}_seq"]:7.1%}'
            for e in roles), flush=True)
    print(f'\n  THE FRONTIER ({PICK_ROLE}, sequential; B0 = {seq["B0"]:.1%} at 2/36 sites):', flush=True)
    print('    ' + '  '.join(
        f'TOP{K} ({nsite[f"TOP{K}"]}/36) {top[K]:.1%}' for K in KS), flush=True)
    print('    consecutive drops: ' + '  '.join(
        f'K{KS[i-1]}->K{KS[i]} {drops[KS[i]]:+.1%}' for i in range(1, len(KS))), flush=True)
    print(f'\n  the DEEP SUFFIX is free in aggregate (TOP9 >=50%) -> {pa}  '
          f'TOP9 {top[9]:.1%} at {nsite["TOP9"]}/36 sites vs B0 {seq["B0"]:.1%} at 2/36', flush=True)
    print(f'  and the frontier is SHARP (>20pp in one step) -> {pb}  '
          f'biggest drop lands at K={frontier}, {drops[frontier]:+.1%}', flush=True)
    print(f'  and it matches §1831 single-site band (K in 5..9) -> {pc}  frontier K={frontier}',
          flush=True)
    print(f'  B0/B1 reproduce §1829, B0+attn9 reproduces §1831, endpoints §1789 -> control {pd}',
          flush=True)

    json.dump({'run': 'deep_suffix_frontier', 'role': PICK_ROLE, 'K_values': list(KS),
               'arms': {k: [list(s2) for s2 in v] for k, v in ARMS.items()},
               'sites_compiled': nsite, 'gap_fraction': frac, 'sequential': seq,
               'top_curve': {str(K): top[K] for K in KS},
               'consecutive_drops': {str(K): drops[K] for K in drops},
               'frontier_K': frontier, 'frontier_drop': drops[frontier],
               'layer_norm_live': ln_live, 'gain_sequential': gseq,
               'predictions': {'pred_a_deep_suffix_free_in_aggregate': bool(pa),
                               'pred_b_frontier_is_sharp': bool(pb),
                               'pred_c_matches_single_site_band': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
