# DOES THE CONSTANT ACTUALLY BREAK?  -- testing §1818's mechanism instead of fitting it.
#
# §1818 confirmed Logan's hypothesis: head 5.7 -- the §1089 CONSTANT-BIAS head, one fixed vector on the
# massive-activation/gain dims, 86% of the stack's total bias value -- carries 85.0% of layer 5's excess
# output norm when a live module is fed a compiled stream, going 6657.8 -> 1057986.8.
#
# §1818 also said plainly what it did NOT establish. The story that fits is that the head's CONSTANCY is
# a property of its attention PATTERN (mass parked on the sink) rather than of its output map, so a
# context-free stream disperses the softmax and one fixed write becomes an average over everything.
# §1818 measured NORMS, not patterns, so that story is fitted to observations rather than tested --
# exactly what §1706 and LESSON 37 warn about.
#
# This tests the property that actually matters, without reconstructing attention patterns (which would
# need rope and the q/k projections rebuilt by hand, with every chance of a silent error). §1089's
# certification is that the head's FUNCTION IS ONE FIXED VECTOR, and that is directly measurable: a
# constant head's output must barely vary ACROSS POSITIONS. Exact single-pass statistics -- sum(c),
# sum(||c||^2), count -- give dispersion = sqrt(E||c||^2 - ||E c||^2) / ||E c||, ~0 for a constant and
# O(1) for anything position-dependent. Direction is tracked separately by the cosine between the live
# and compiled mean output vectors.
#
# ROLES. skip7000; layers 4, 5, 6, all nine heads each. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a §1089 REPRODUCES ON A NEW INSTRUMENT: in the LIVE stream head 5.7's across-position
#          dispersion is below 0.15. §1089 certified constancy by a fidelity argument (one fixed vector
#          recovers ~0.985 local fidelity; §437's "constant explains 101%"); this is an independent
#          GEOMETRIC measurement of the same claim on these rows. If FALSE, either §1089 does not hold
#          on this data or my decomposition does not mean what I think it does -- and nothing else in
#          the run is readable.
#   pred_b AND IT BREAKS: in the COMPILED stream that dispersion exceeds 1.0 -- the head stops being
#          constant. If FALSE the head is still writing a near-fixed vector and has merely been
#          AMPLIFIED, which refutes the softmax-dispersal story and points at input magnitude instead.
#   pred_c AND IT WRITES SOMEWHERE ELSE: the cosine between its live and compiled mean output vectors is
#          below 0.50. Scored separately from pred_b because dispersion and direction are independent --
#          a head can scatter around the SAME mean (dispersion up, cosine high) or shift to a new mean
#          while staying tight. Those license different accounts and one predicate would blur them
#          (LESSON 44).
#   pred_d CONTROLS: the per-head decomposition stays exact (heads plus bias reconstruct each layer's
#          write below 1e-3 relative in both streams, LESSON 34); and the per-head mean norms reproduce
#          §1818's PUBLISHED L5h7 6657.833 -> 1057986.774, L5h2 505.668 -> 121745.939 and L4h7
#          3247.251 -> 3711.856 within 1% -- the same object measured again before it is explained.
#          Coverage 5419 of 50257.
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
OUT = PT + 'ops/does_the_constant_break_results.json'
PROBE_LS = ()
KIND_LS = ()
RANKS = (64,)                # the settled table rank; only the STREAM matters here
PROBE_LS = (4, 5, 6)         # the cliff layers and the adjacent control
S1818_NORM = {'L5h7': (6657.833, 1057986.774), 'L5h2': (505.668, 121745.939),
              'L4h7': (3247.251, 3711.856)}   # (live, compiled) mean norms, §1818
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


@torch.no_grad()
def head_norms(rows, hooks, tag):
    """Mean per-head output norm at scored positions, in whatever stream `hooks` produces.

    Attention writes `c_proj(einsum(...).reshape(B, T, D))`, so hooking c_proj's INPUT gives the
    concatenated pre-projection. Head h's contribution to the residual is its own HD-slice pushed
    through the matching COLUMNS of c_proj.weight ([out, in] for nn.Linear). Heads plus bias sum
    EXACTLY to the layer's write, which pred_d CHECKS rather than assumes (LESSON 34)."""
    acc = {(L, h): [0.0, 0] for L in PROBE_LS for h in range(NH)}
    recon = {L: [0.0, 0.0] for L in PROBE_LS}
    # single-pass EXACT statistics for across-position dispersion:
    #   Var = E||c||^2 - ||E c||^2, from sum(c), sum(||c||^2), count.
    ssum = {(L, h): torch.zeros(D, device=DEV, dtype=torch.float64) for L in PROBE_LS
            for h in range(NH)}
    ssq = {(L, h): 0.0 for L in PROBE_LS for h in range(NH)}

    def mk(L):
        def hook(mod, args, out):
            o = (args[0] if isinstance(args, tuple) else args).detach().float()[:, 64:]
            B, P, _ = o.shape
            oh = o.reshape(B, P, NH, HD)
            Wp = mod.weight.detach().float()
            tot = torch.zeros(B, P, D, device=o.device)
            for h in range(NH):
                c = oh[:, :, h, :] @ Wp[:, h * HD:(h + 1) * HD].T
                acc[(L, h)][0] += float(c.norm(dim=-1).sum())
                acc[(L, h)][1] += B * P
                cf = c.double().reshape(-1, D)
                ssum[(L, h)] += cf.sum(0)
                ssq[(L, h)] += float((cf * cf).sum())
                tot = tot + c
            y = (out[0] if isinstance(out, tuple) else out).detach().float()[:, 64:]
            if getattr(mod, 'bias', None) is not None:
                tot = tot + mod.bias.detach().float()
            recon[L][0] += float((tot - y).norm())
            recon[L][1] += float(y.norm())
            return None
        return hook

    handles = [H[L].attn.c_proj.register_forward_hook(mk(L)) for L in PROBE_LS]
    try:
        for j in range(0, rows.shape[0], 8):
            forward_logits(rows[j:j + 8, :-1].to(DEV).contiguous(), hooks)
    finally:
        for hd in handles:
            hd.remove()
    o = {f'L{L}h{h}': acc[(L, h)][0] / max(acc[(L, h)][1], 1)
         for L in PROBE_LS for h in range(NH)}
    disp, meanvec = {}, {}
    for L in PROBE_LS:
        for h in range(NH):
            n = max(acc[(L, h)][1], 1)
            mu = ssum[(L, h)] / n
            var = max(ssq[(L, h)] / n - float(mu @ mu), 0.0)
            disp[f'L{L}h{h}'] = (var ** 0.5) / max(float(mu.norm()), 1e-9)
            meanvec[f'L{L}h{h}'] = mu
    rec = {f'L{L}': recon[L][0] / max(recon[L][1], 1e-9) for L in PROBE_LS}
    print(f'  [{tag}] reconstruction residual (must be ~0): ' + '  '.join(
        f'{k2} {v:.2e}' for k2, v in rec.items()), flush=True)
    return o, rec, disp, meanvec


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
    print(f'DOES THE CONSTANT BREAK | dispersion + direction per head at L{PROBE_LS} | '
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
    hks = [(st, row_hook(frr[st])) for st in sites]
    ev = load(EVAL_SETS[0][1])
    live_n, rec_l, live_d, live_mu = head_norms(ev, [], 'live')
    comp_n, rec_c, comp_d, comp_mu = head_norms(ev, hks, 'compiled')
    del ev, frr, hks
    torch.cuda.empty_cache()

    ratio = {k2: comp_n[k2] / max(live_n[k2], 1e-9) for k2 in live_n}
    cos = {k2: float(torch.nn.functional.cosine_similarity(
        live_mu[k2].unsqueeze(0), comp_mu[k2].unsqueeze(0)).item()) for k2 in live_mu}

    print('\n  across-position dispersion  std/||mean||  (a CONSTANT head is ~0):', flush=True)
    for L in PROBE_LS:
        print(f'    L{L}:  ' + '  '.join(
            f'h{h} {live_d[f"L{L}h{h}"]:6.3f}->{comp_d[f"L{L}h{h}"]:8.3f}' for h in range(NH)),
            flush=True)
    print('\n  cosine(live mean vector, compiled mean vector):', flush=True)
    for L in PROBE_LS:
        print(f'    L{L}:  ' + '  '.join(f'h{h} {cos[f"L{L}h{h}"]:+.3f}' for h in range(NH)),
              flush=True)
    print('\n  per-head mean output norm, live -> compiled:', flush=True)
    for L in PROBE_LS:
        print(f'    L{L}:  ' + '  '.join(
            f'h{h} {live_n[f"L{L}h{h}"]:8.1f}->{comp_n[f"L{L}h{h}"]:11.1f}' for h in range(NH)),
            flush=True)

    med_live = sorted(live_d[f'L5h{h}'] for h in range(NH))[NH // 2]
    pa = live_d['L5h7'] < 0.15
    pb = comp_d['L5h7'] > 1.0
    pc = cos['L5h7'] < 0.50
    pd = (all(rec_l[f'L{L}'] < 1e-3 and rec_c[f'L{L}'] < 1e-3 for L in PROBE_LS)
          and all(abs(live_n[k2] / v[0] - 1.0) <= 0.01 and abs(comp_n[k2] / v[1] - 1.0) <= 0.01
                  for k2, v in S1818_NORM.items())
          and NFULL == NCOV)

    print(f'\n  §1089 reproduces: 5.7 is CONSTANT live (disp <0.15) -> {pa}  '
          f'{live_d["L5h7"]:.4f}  (median other L5 head {med_live:.3f})', flush=True)
    print(f'  and it STOPS being constant when compiled (disp >1.0) -> {pb}  '
          f'{comp_d["L5h7"]:.4f}', flush=True)
    print(f'  and it writes a DIFFERENT direction, not a scaled one (cos <0.50) -> {pc}  '
          f'{cos["L5h7"]:+.4f}', flush=True)
    print(f'  decomposition exact + norms reproduce §1818 -> control {pd}', flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'does_the_constant_break', 'probe_layers': list(PROBE_LS),
               'NH': NH, 'HD': HD, 'live_norm': live_n, 'compiled_norm': comp_n, 'ratio': ratio,
               'dispersion_live': live_d, 'dispersion_compiled': comp_d, 'cosine': cos,
               'reconstruction_live': rec_l, 'reconstruction_compiled': rec_c,
               'predictions': {'pred_a_57_is_constant_live': bool(pa),
                               'pred_b_57_stops_being_constant': bool(pb),
                               'pred_c_direction_changes': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
