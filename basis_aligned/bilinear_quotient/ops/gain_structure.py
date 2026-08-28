# IS THE GAIN VECTOR STRUCTURE OR CURVE-FIT?  -- following §1820.
#
# §1820 found that nine per-head scalars repair the L5 cliff essentially completely: -11.88pp -> -0.05pp,
# 99.6 / 98.8 / 99.2% recovered. It also found that head 5.7, which §1818 measured carrying 85% of the
# layer's excess NORM, accounts for only 15% of the DAMAGE -- norm share is not damage share.
#
# TRANSFER IS ALREADY ESTABLISHED AND I AM NOT RE-RUNNING IT. §1818's ratios were measured on skip7000
# ALONE, and §1820 applied them unchanged to skip11000 and skip1200, recovering 98.8% and 99.2%. So the
# gains are not fitted per role. What is NOT established is whether they are structure or coincidence:
#   - does the same construction repair a DIFFERENT layer (L6, whose top head is h1, not h7)?
#   - is the per-head detail necessary, or would ONE scalar per layer do?
#   - are the gains a property of the LAYER, or would any plausible vector work?
#
# ROLES. skip7000, skip11000, skip1200. Gains from §1818's published skip7000 measurements, applied
# unchanged everywhere. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44, constants named with their OBJECT and UNITS per LESSON 53:
#   pred_a IT IS THE CONSTRUCTION, NOT THE LAYER: nine per-head gains repair L6 to at least 90% of its
#          deficit, at every role. L6's damage is distributed differently (top head h1 at 106.8x from a
#          base of 1786.0, against L5's h7 at 158.9x from 6657.8). If FALSE the L5 repair is specific to
#          L5's particular head geometry and there is no general remedy for §1806's directional
#          poisoning -- only a special case that happened to work once.
#   pred_b THE PER-HEAD DETAIL IS NECESSARY: a SINGLE per-layer scalar -- the live-norm-weighted mean
#          ratio, which corrects the layer's total output norm exactly -- recovers under 50% at L5. If
#          FALSE, one number per layer suffices, the nine-vector is over-parameterised, and the whole
#          effect is a layer-level magnitude error after all, which would simplify the account
#          considerably and partly rehabilitate the site-level framing §1810 closed.
#   pred_c THE GAINS ARE LAYER-SPECIFIC: applying L5's gain vector at L6 recovers under 25% of L6's
#          deficit. This is the control that separates a calibration from any-vector-helps: if an
#          unrelated nine-vector repairs L6 nearly as well as its own, the repair is not carrying
#          layer information and pred_a means much less than it appears to.
#   pred_d CONTROLS, cross-run per LESSON 42 and objects named per LESSON 53: the L5 raw arm reproduces
#          §1820's PUBLISHED rank-64 deltas -0.1188 / -0.1237 / -0.1192 within 0.5pp; the baseline
#          reproduces §1786's RANK-64 top-1 0.1288 / 0.1349 / 0.1289 within 0.001 (NOT §1789's
#          full-rank figures -- quoting those at a rank-64 build is exactly what broke §1820's pred_d);
#          the live arm reproduces 0.3932 / 0.4235 / 0.3888; and the placement control -- L5's per-head
#          gain applied to the FULLY substituted program, where L5's output is discarded -- moves top-1
#          by under 0.05pp. Coverage 5419 of 50257.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
NH = 9; HD = D // NH        # bilin18: nine heads of 128
RANKS = (None,)
MAP_RANK = 64
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/gain_structure_results.json'
PROBE_LS = ()
KIND_LS = ()
RANKS = (64,)                # the settled table rank; only the STREAM matters here
# §1818's measured compiled/live per-head norm ratios, skip7000. Corrections are reciprocals.
L5_RATIO = {0: 0.47, 1: 8.68, 2: 240.76, 3: 0.97, 4: 34.02,
            5: 36.86, 6: 83.93, 7: 158.91, 8: 6.61}
L6_RATIO = {0: 2.96, 1: 106.79, 2: 2.85, 3: 9.55, 4: 23.30,
            5: 2.14, 6: 20.38, 7: 40.54, 8: 21.75}
# RANK-64 all-sites constants (§1786) -- the object THIS build produces. LESSON 53.
S1786_RANK64_TOP1_PP = {'skip7000': 0.1288, 'skip11000': 0.1349, 'skip1200': 0.1289}
S1789_LIVE_TOP1_PP = {'skip7000': 0.3932, 'skip11000': 0.4235, 'skip1200': 0.3888}
# §1820's PERCENTAGE-POINT deltas, rank-64 build, L5 held live
S1820_L5_PP = {'skip7000': -0.1188, 'skip11000': -0.1237, 'skip1200': -0.1192}
DEPTHS = (-1, 7, 10, 13)     # -1 = every site compiled
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


def gain_hook(gains):
    """Scale selected heads' contributions by rescaling c_proj's INPUT slices.

    c_proj is linear and its input is the concatenation of the nine heads, so multiplying columns
    [h*HD:(h+1)*HD] by g scales exactly head h's contribution to the residual and nothing else.
    """
    def pre(mod, args):   # noqa: the layer is bound by the caller
        x = args[0]
        if not gains:
            return None
        y = x.clone()
        for h, g in gains.items():
            y[..., h * HD:(h + 1) * HD] = y[..., h * HD:(h + 1) * HD] * g
        return (y,) + tuple(args[1:])
    return pre


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
    print(f'GAIN STRUCTURE | per-head vs one-scalar, L5 and L6, and cross-layer transfer | '
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
    frr = build(NFULL, 64, 64)[0]
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

    LIVE_OF = {L: [st for st in sites if st != ('attn', L)] for L in (5, 6)}
    inv = {L: {h: 1.0 / R[h] for h in range(NH)}
           for L, R in ((5, L5_RATIO), (6, L6_RATIO))}
    # one scalar per layer, weighted by each head's LIVE norm share (§1818), so the layer's total
    # output norm is corrected even though no individual head is
    LN5 = {0: 242.306, 1: 247.224, 2: 505.668, 3: 331.857, 4: 273.753,
           5: 247.749, 6: 518.346, 7: 6657.833, 8: 358.004}
    LN6 = {0: 113.665, 1: 1786.015, 2: 61.154, 3: 761.311, 4: 88.864,
           5: 115.322, 6: 190.754, 7: 205.051, 8: 87.538}
    lmean = {}
    for L, R, LN in ((5, L5_RATIO, LN5), (6, L6_RATIO, LN6)):
        tot = sum(LN.values())
        lmean[L] = 1.0 / (sum(LN[h] * R[h] for h in range(NH)) / tot)

    print(f'\n  arms  ({time.time() - t0:.0f}s)', flush=True)
    run('all_substituted', sites)
    run('live_model', [])
    for L in (5, 6):
        run(f'L{L}live_raw', LIVE_OF[L])
        run(f'L{L}live_perhead', LIVE_OF[L], (L, inv[L]))
        run(f'L{L}live_layerscalar', LIVE_OF[L], (L, {h: lmean[L] for h in range(NH)}))
    run('L6live_wrongvec', LIVE_OF[6], (6, inv[5]))     # L5's gains applied at L6
    run('allsub_perhead5', sites, (5, inv[5]))          # placement control
    del evs, frr, allh
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    base = {e: res[e]['all_substituted']['top1'] for e in roles}

    def d(e, k2):
        return res[e][k2]['top1'] - base[e]
    defc = {L: {e: -d(e, f'L{L}live_raw') for e in roles} for L in (5, 6)}

    def rec(L, arm):
        return {e: (d(e, arm) - d(e, f'L{L}live_raw')) / max(defc[L][e], 1e-9) for e in roles}
    r5, r6 = rec(5, 'L5live_perhead'), rec(6, 'L6live_perhead')
    s5 = rec(5, 'L5live_layerscalar')
    wrong = rec(6, 'L6live_wrongvec')
    pa = all(r6[e] >= 0.90 for e in roles)
    pb = all(s5[e] < 0.50 for e in roles)
    pc = all(wrong[e] < 0.25 for e in roles)
    pd = (all(abs(d(e, 'L5live_raw') - S1820_L5_PP[e]) <= 0.005 for e in roles)
          and all(abs(res[e]['all_substituted']['top1'] - S1786_RANK64_TOP1_PP[e]) <= 0.001
                  and abs(res[e]['live_model']['top1'] - S1789_LIVE_TOP1_PP[e]) <= 0.001
                  for e in roles)
          and all(abs(d(e, 'allsub_perhead5')) < 0.0005 for e in roles)
          and NFULL == NCOV)

    print('\n  delta vs the all-substituted baseline (pp):', flush=True)
    for e in roles:
        print(f'    {e:10s} L5 raw {100*d(e, "L5live_raw"):+7.2f} perhead '
              f'{100*d(e, "L5live_perhead"):+6.2f} layerscalar '
              f'{100*d(e, "L5live_layerscalar"):+7.2f}  |  L6 raw '
              f'{100*d(e, "L6live_raw"):+7.2f} perhead {100*d(e, "L6live_perhead"):+6.2f} '
              f'L5-vec {100*d(e, "L6live_wrongvec"):+7.2f}', flush=True)
    print(f'\n  per-head gains repair L6 too (>=90%) -> {pa}  ' + '  '.join(
        f'{e} {r6[e]:.1%} of {100*defc[6][e]:.2f}pp' for e in roles), flush=True)
    print(f'  a SINGLE layer scalar is not enough at L5 (<50%) -> {pb}  ' + '  '.join(
        f'{e} {s5[e]:.1%} (per-head {r5[e]:.1%})' for e in roles), flush=True)
    print(f'  L5\'s gain vector does NOT transfer to L6 (<25%) -> {pc}  ' + '  '.join(
        f'{e} {wrong[e]:.1%}' for e in roles), flush=True)
    print(f'  L5 raw reproduces §1820, endpoints reproduce §1786 rank-64 + live, placement '
          f'control exact -> {pd}', flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'head_gain_repair', 'NH': NH, 'HD': HD, 'gains_used': L5_RATIO,
               'delta_pp': {e: {k2: d(e, k2) for k2 in
                                ('L5live_raw', 'L5live_perhead', 'L5live_layerscalar',
                                 'L6live_raw', 'L6live_perhead', 'L6live_wrongvec',
                                 'allsub_perhead5')} for e in roles},
               'recovered_L5_perhead': r5, 'recovered_L6_perhead': r6,
               'recovered_L5_layerscalar': s5, 'recovered_L6_with_L5_vector': wrong,
               'predictions': {'pred_a_perhead_repairs_L6': bool(pa),
                               'pred_b_layer_scalar_insufficient': bool(pb),
                               'pred_c_gains_are_layer_specific': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
