# IS THE PARTIAL COMPILE A BETTER OBJECT?  -- the question §1810 ended on.
#
# Five sections have now closed repair routes for the ALL-SITES program: coverage (§1800), a minimal
# departure (§1802), bottom-up compilation (§1806/§1807), and row magnitudes (§1804-§1810). Meanwhile
# §1805's top-down curve quietly showed that a PARTIAL compile is a much better deal in fidelity --
# eleven live layers recover half the 25.77 / 28.10 / 25.24 pp gap, and the bottom four layers cost
# almost nothing to compile. That object has never been priced.
#
# It cannot be free: a partial compile RETAINS native modules, at 15.926M + 7.963M reals per layer
# (§1754's accounting, 430.00M for all eighteen). So depth buys fidelity and costs reals, while rank
# truncation buys reals and costs fidelity, and the question is which frontier wins where.
#
# COST ACCOUNTING, verified against the record before use (PRE-FLIGHT A): a full covered table block is
# NCOV x D per site plus D for the uncovered mean; a rank-r block is r*(NCOV+D) + 2*D; the map is
# MAP_RANK*2*D per site. Those reproduce §1754's published 224.737M for 36 full tables and §1786's
# published 20.531M for the settled rank-64 design point, both to the digit -- pred_d re-checks both.
# (The 15.886M figure elsewhere in the ledger is §1758's different program and is not this formula.)
#
# ROLES. skip7000, skip11000, skip1200; depths -1 (all compiled), 3, 7, 10, 13; ranks full and 64.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE SETTLED DESIGN POINT SURVIVES: no arm is both cheaper than the settled rank-64
#          all-sites program AND more accurate, at every role. If FALSE, §1786's certified design point
#          is Pareto-dominated and the certified object should change -- the most consequential outcome
#          available here.
#          NOTE ON FALSIFIABILITY: with ranks (None, 64) alone this predicate COULD NOT FAIL, because
#          the settled rank-64 all-sites arm is the minimum-cost arm in that set and nothing can be
#          cheaper than it. Rank 16 is included precisely so cheaper arms exist -- all-sites rank 16
#          costs 9.2M against the settled 20.5M. A bar that cannot fail is not a bar (LESSON 40), and
#          this one was caught before the run rather than after.
#   pred_b AND FIDELITY IS EXPENSIVE: the cheapest arm that beats the settled point on top-1 costs at
#          least 5x its reals, at every role. If FALSE, a modest extra budget buys a real accuracy
#          gain, and the thread should characterise that arm rather than the all-sites one. Scored
#          separately from pred_a because "not dominated" and "not cheaply beaten" are different
#          claims: an arm can cost more and still be the better buy.
#   pred_c AND RANK TRUNCATION IS NEARLY FREE AT DEPTH: at L=10, the rank-64 partial compile retains at
#          least 80% of the full-rank partial compile's gap recovery. §1787 found rank 64 is the
#          all-sites optimum; if it does not transfer to a partial compile, the two knobs interact and
#          the frontier must be mapped jointly rather than read off either axis.
#   pred_d CONTROLS, cross-run per LESSON 42: the full-rank all-compiled arm reproduces §1805's
#          PUBLISHED L10 figure (+13.86 / +14.56 / +13.28 pp) within 2 points of gap; the cost formula
#          reproduces §1754's 224.737M and §1786's 20.531M; endpoints reproduce §1789's 0.1355 / 0.1425
#          / 0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001; coverage 5419.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None,)
MAP_RANK = 64
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/partial_compile_frontier_results.json'
PROBE_LS = ()
KIND_LS = ()
DEPTHS = (-1, 3, 7, 10, 13)      # -1 = every site compiled
RANKS = (None, 64, 16)   # rank 16 is included so pred_a CAN fail: without an arm cheaper than
                         # the settled rank-64 point, 'is it Pareto-dominated' is unfalsifiable
NATIVE_PER_LAYER = 15.926e6 + 7.963e6   # one MLP + one attn, from the §1754 accounting
S1805_L10 = {'skip7000': 0.1386, 'skip11000': 0.1456, 'skip1200': 0.1328}
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
def evaluate(rows, hooks, keep_mask):
    """Top-1 overall, on the head, and restricted to positions whose CURRENT token is still covered.

    That last slice is a KNOWN-ANSWER control (LESSON 34). A covered token's table is built from its
    own length-1 forward and the program is position-wise (§1765), so removing OTHER tokens from the
    covered set cannot change what happens at a position that kept its own table. Those numbers must
    be identical across every coverage fraction, not merely close."""
    a = {'n': 0, 'hit': 0, 'head_n': 0, 'head_hit': 0, 'kept_n': 0, 'kept_hit': 0}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        h = forward_logits(idx, hooks)[:, 64:].argmax(-1) == tg
        hd = COV['freq'][tg] >= 125
        kp = keep_mask[idx[:, 64:]]
        a['n'] += int(tg.numel()); a['hit'] += int(h.sum())
        a['head_n'] += int(hd.sum()); a['head_hit'] += int(h[hd].sum())
        a['kept_n'] += int(kp.sum()); a['kept_hit'] += int(h[kp].sum())
    return {'n': a['n'], 'top1': a['hit'] / max(a['n'], 1),
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
    print(f'PARTIAL COMPILE FRONTIER | depths {DEPTHS} x table ranks {RANKS}, cost in reals | '
          f'settled program (context-free tables + output-NN fallback + rank-{MAP_RANK} map) | '
          f'DISCOVERY ONLY', flush=True)

    def build(n, rank=None):
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
            mp = (U[:, :MAP_RANK] * S[:MAP_RANK]) @ Vh[:MAP_RANK]
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
    cost = {}
    for r in RANKS:
        frr = build(NFULL, r)[0]
        hks = {st: row_hook(frr[st]) for st in sites}
        for L in DEPTHS:
            comp = [st for st in sites if st[1] > L]
            n = len(comp)
            tab = n * (NCOV * D + D) if r is None else n * (r * (NCOV + D) + 2 * D)
            mp = n * MAP_RANK * 2 * D
            native = (L + 1) * NATIVE_PER_LAYER
            cost[f'r{r}_L{L}'] = tab + mp + native
            run(f'r{r}_L{L}', comp, hks)
        del frr, hks
        torch.cuda.empty_cache()

    # (The compiled-stream norm diagnostic of §1804 is NOT rerun here: it needs the full-rank bank
    # and its hooks, which are released above so the rank sweep can build its own. Its figures are
    # already published in §1804 and corrected in §1807; nothing in this run depends on them.)
    del evs
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    base = {e: res[e]['all_substituted']['top1'] for e in roles}

    def d(e, label):
        return res[e][label]['top1'] - base[e]
    gap = {e: res[e]['live_model']['top1'] - base[e] for e in roles}
    keys = [f'r{r}_L{L}' for r in RANKS for L in DEPTHS]
    acc = {e: {k: res[e][k]['top1'] for k in keys} for e in roles}
    ref = 'r64_L-1'                      # the settled §1786 design point: every site, rank 64
    dominated = {e: [k for k in keys
                     if cost[k] < cost[ref] and acc[e][k] > acc[e][ref]] for e in roles}
    beats = {e: [k for k in keys if acc[e][k] > acc[e][ref]] for e in roles}
    cheapest_beat = {e: (min(beats[e], key=lambda k: cost[k]) if beats[e] else None)
                     for e in roles}
    ratio = {e: (cost[cheapest_beat[e]] / cost[ref]) if cheapest_beat[e] else None
             for e in roles}
    keepfrac = {e: {L: (acc[e][f'r64_L{L}'] - acc[e]['rNone_L-1'])
                    / max(acc[e][f'rNone_L{L}'] - acc[e]['rNone_L-1'], 1e-9)
                    for L in DEPTHS if L != -1} for e in roles}
    pa = all(not dominated[e] for e in roles)
    pb = all(ratio[e] is not None and ratio[e] >= 5.0 for e in roles)
    pc = all(keepfrac[e][10] >= 0.80 for e in roles)
    pd = (all(abs(res[e]['all_substituted']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['live_model']['top1'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(d(e, 'rNone_L10') / gap[e] - S1805_L10[e]) <= 0.02 for e in roles)
          and abs(cost['rNone_L-1'] - 36 * (NCOV * D + D) - 36 * MAP_RANK * 2 * D) < 1.0
          and abs(36 * (64 * (NCOV + D) + 2 * D) + 36 * MAP_RANK * 2 * D - 20.531e6) < 1e4
          and NFULL == NCOV)

    print('\n  cost / fidelity frontier  (reals, and top-1 per role)', flush=True)
    for k in sorted(keys, key=lambda k: cost[k]):
        print(f'    {k:12s} {cost[k]/1e6:9.3f}M  ' + '  '.join(
            f'{e} {acc[e][k]:6.2%}' for e in roles), flush=True)
    print(f'\n  the settled design point {ref} is NOT Pareto-dominated -> {pa}  ' + '  '.join(
        f'{e} {dominated[e] or "none"}' for e in roles), flush=True)
    print(f'  the cheapest arm that beats it costs >=5x more -> {pb}  ' + '  '.join(
        f'{e} {cheapest_beat[e]} at {ratio[e]:.1f}x' if ratio[e] else f'{e} nothing beats it'
        for e in roles), flush=True)
    print(f'  rank-64 keeps >=80% of full-rank recovery at L10 -> {pc}  ' + '  '.join(
        f'{e} {keepfrac[e][10]:.0%}' for e in roles), flush=True)
    print(f'  all-substituted and live reproduce §1789, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'l5_cliff_probe',
               'calibration_scale': {f'{k}{L}': scale[(k, L)]
                                    for k in ('attn', 'mlp') for L in range(18)},
               'ratio': {str(L): ratio[L] for L in ratio},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'scale': {f'{k}{L}': scale[(k, L)] for k in ('attn', 'mlp') for L in range(18)},
               'cost_reals': cost, 'top1': acc,
               'dominating_arms': {e: dominated[e] for e in roles},
               'cheapest_beat': {e: cheapest_beat[e] for e in roles},
               'cost_ratio': {e: ratio[e] for e in roles},
               'predictions': {'pred_a_settled_point_not_dominated': bool(pa),
                               'pred_b_cheapest_beat_costs_5x': bool(pb),
                               'pred_c_rank64_keeps_80pc_at_L10': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
