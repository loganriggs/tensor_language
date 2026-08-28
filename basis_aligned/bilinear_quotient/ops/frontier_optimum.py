# WHERE IS THE ACTUAL DESIGN POINT?  -- §1811 showed the current one is not it.
#
# §1811 swept depth x rank and found that FOUR OF FIVE full-rank arms are Pareto-dominated: the
# all-sites full-rank program (230.087M reals, 13.55%) is beaten by a rank-16 compile of the top eleven
# layers (196.210M, 18.71%) -- cheaper AND five points better. Full rank is dropped here for that
# reason; it costs a rebuild and buys nothing.
#
# What §1811 did NOT do is locate an optimum. It sampled ranks 16 and 64 only, so "rank 16 dominates
# full rank" leaves open whether 4, 8 or 32 is better still, and whether the best accuracy-per-real is
# bought by compiling everything cheaply or by compiling less of the stack at higher fidelity.
#
# ROLES. skip7000, skip11000, skip1200; depths -1 (all compiled), 7, 10, 13 x ranks 4, 8, 16, 32, 64.
# DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44, and constants named with their UNITS per LESSON 49:
#   pred_a THE EFFICIENCY OPTIMUM IS A PARTIAL COMPILE: the arm buying the most top-1 per million reals
#          above the cheapest arm (r4_L-1) is NOT an all-sites arm, at every role. If FALSE the best
#          accuracy-per-real is still bought by compiling the whole stack as cheaply as possible, and
#          §1811's dominance result -- while true -- does not move the design point, only the pricing
#          of the full-rank arms.
#   pred_b FIDELITY MATTERS AT DEPTH: at L10 the best rank by accuracy is 32 or 64, at every role. If
#          FALSE a rank of 4, 8 or 16 is best even where only seven layers are compiled, which would
#          say the tables carry so little at depth that their rank is irrelevant -- and would make the
#          cheapest rank the right choice everywhere.
#   pred_c AND THERE IS A FLOOR: at L10 rank 8 loses at least 1 percentage point against rank 64, at
#          every role. §1786 found a floor in the map rank (ranks 8 and 16 were worse than copying);
#          this asks whether the TABLE rank has one too. Scored separately from pred_b because "32 or
#          64 is best" and "8 is materially worse" are different claims -- the best rank can win by a
#          margin too small to matter.
#   pred_d CONTROLS, cross-run per LESSON 42: r16_L7 and r64_L13 reproduce §1811's PUBLISHED top-1 of
#          0.1871 / 0.1942 / 0.1852 and 0.3280 / 0.3474 / 0.3226 within 0.01 -- quoted as
#          PERCENTAGE-POINT accuracies, with the units in the constant's name, because §1811's pred_d
#          failed by comparing a pp delta against a gap fraction; the endpoints reproduce §1789's
#          0.1355 / 0.1425 / 0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001; the rank-64 all-sites
#          cost reproduces §1786's 20.531M; coverage 5419 of 50257.
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
OUT = PT + 'ops/frontier_optimum_results.json'
PROBE_LS = ()
KIND_LS = ()
DEPTHS = (-1, 7, 10, 13)         # -1 = every site compiled
RANKS = (4, 8, 16, 32, 64)   # full rank is dropped: §1811 showed it is dominated at every
                             # budget tested, so it buys nothing here and costs a rebuild
NATIVE_PER_LAYER = 15.926e6 + 7.963e6   # one MLP + one attn, from the §1754 accounting
# UNITS IN THE NAME (LESSON 49): these are PERCENTAGE-POINT deltas over the all-compiled arm,
# not gap fractions. §1811's pred_d failed by comparing the first against the second.
S1811_PP = {'r16_L7': {'skip7000': 0.1871, 'skip11000': 0.1942, 'skip1200': 0.1852},
            'r64_L13': {'skip7000': 0.3280, 'skip11000': 0.3474, 'skip1200': 0.3226}}
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
    print(f'FRONTIER OPTIMUM | depths {DEPTHS} x ranks {RANKS}, locating the design point | '
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
    base = {e: acc[e]['r4_L-1'] for e in roles}          # cheapest arm, the efficiency origin
    # efficiency = accuracy bought per million reals above the cheapest arm
    eff = {e: {k: ((acc[e][k] - base[e]) / ((cost[k] - cost['r4_L-1']) / 1e6))
               for k in keys if cost[k] > cost['r4_L-1']} for e in roles}
    best_eff = {e: max(eff[e], key=lambda k: eff[e][k]) for e in roles}
    pareto = {e: [k for k in keys
                  if not any(cost[j] <= cost[k] and acc[e][j] > acc[e][k] for j in keys)]
              for e in roles}
    # best rank at each depth, by accuracy
    best_rank = {e: {L: max(RANKS, key=lambda r: acc[e][f'r{r}_L{L}']) for L in DEPTHS}
                 for e in roles}
    floor = {e: {L: acc[e][f'r8_L{L}'] - acc[e][f'r64_L{L}'] for L in DEPTHS} for e in roles}
    pa = all(best_eff[e].split('_')[1] != 'L-1' for e in roles)
    pb = all(best_rank[e][10] in (32, 64) for e in roles)
    pc = all(floor[e][10] <= -0.01 for e in roles)
    pd = (all(abs(res[e]['all_substituted']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['live_model']['top1'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(acc[e][k] - v[e]) <= 0.01 for k, v in S1811_PP.items() for e in roles)
          and abs(36 * (64 * (NCOV + D) + 2 * D) + 36 * MAP_RANK * 2 * D - 20.531e6) < 1e4
          and NFULL == NCOV)

    print('\n  cost / fidelity  (reals, top-1 per role)', flush=True)
    for k in sorted(keys, key=lambda k: cost[k]):
        mark = ' *' if all(k in pareto[e] for e in roles) else '  '
        print(f'   {mark}{k:10s} {cost[k]/1e6:9.3f}M  ' + '  '.join(
            f'{e} {acc[e][k]:6.2%}' for e in roles), flush=True)
    print('   (* = on the Pareto frontier at every role)', flush=True)
    print('\n  best rank at each depth, by accuracy:', flush=True)
    for e in roles:
        print(f'    {e:10s} ' + '  '.join(f'L{L}: rank {best_rank[e][L]}' for L in DEPTHS),
              flush=True)
    print(f'\n  the efficiency optimum is a PARTIAL compile -> {pa}  ' + '  '.join(
        f'{e} {best_eff[e]} ({eff[e][best_eff[e]]:.4f} pp per M)' for e in roles), flush=True)
    print(f'  rank 32 or 64 is best at L10 -> {pb}  ' + '  '.join(
        f'{e} rank {best_rank[e][10]}' for e in roles), flush=True)
    print(f'  and rank 8 loses >=1pp against rank 64 at L10 -> {pc}  ' + '  '.join(
        f'{e} {100*floor[e][10]:+.2f}pp' for e in roles), flush=True)
    print(f'  all-substituted and live reproduce §1789, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'l5_cliff_probe',
               'calibration_scale': {f'{k}{L}': scale[(k, L)]
                                    for k in ('attn', 'mlp') for L in range(18)},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'scale': {f'{k}{L}': scale[(k, L)] for k in ('attn', 'mlp') for L in range(18)},
               'cost_reals': cost, 'top1': acc,
               'pareto': {e: pareto[e] for e in roles},
               'efficiency_pp_per_M': {e: eff[e] for e in roles},
               'best_efficiency': {e: best_eff[e] for e in roles},
               'best_rank_by_depth': {e: {str(L): best_rank[e][L] for L in DEPTHS}
                                      for e in roles},
               'predictions': {'pred_a_efficiency_optimum_is_partial': bool(pa),
                               'pred_b_rank32_or_64_best_at_L10': bool(pb),
                               'pred_c_rank8_loses_1pp_at_L10': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
