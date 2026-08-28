# THE SAME CLAIM FROM THE OTHER DIRECTION, AND THE ACTIONABLE ARM.
#
# §1805 measured a top-down compile curve -- depths 0..L live, the rest compiled -- and found the bottom
# four layers nearly free (2.2 / 2.2 / 1.7% of the gap), half the gap reachable with eleven live layers,
# and the per-layer marginals peaking in the MIDDLE at L7 (+10.5) and L8 (+12.0) rather than late. That
# was one curve read one way; §1805's own LESSON 47 is that a shape needs sampling designed for it.
#
# This runs the MIRROR -- depths 0..L compiled, L+1..17 live -- from the SAME build, so the two curves
# are directly comparable and the per-layer marginal cost of compiling a layer is read twice, from
# independent directions. It also runs the arm §1805 pointed at: compile everything EXCEPT the
# expensive middle band L6-L9, and its complement.
#
# ROLES. skip7000, skip11000, skip1200; settled program of §1786 at full coverage. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE BOTTOM IS FREE, MEASURED THE OTHER WAY: compiling layers 0-3 and leaving 4-17 live costs
#          at most 5% of the gap, at every role. §1805 says keeping 0-3 live BUYS 2.2%; this says
#          compiling them COSTS little. They are different measurements of the same layers -- the first
#          against a fully compiled program, the second against a fully live one -- and a large
#          disagreement would mean layer contributions are strongly context-dependent on what else is
#          compiled, which would undermine reading either curve layer-by-layer at all.
#   pred_b THE TWO DIRECTIONS AGREE ON WHERE THE COST IS: the bottom-up curve's most expensive single
#          layer falls in L6-L9, at every role. If FALSE the two curves disagree about which layers
#          matter, and neither supports a per-layer reading -- the honest conclusion would then be that
#          only whole-prefix and whole-suffix figures are meaningful.
#   pred_c THE TARGETED COMPILE IS WORTH SOMETHING: compiling all fourteen layers OUTSIDE L6-L9, leaving
#          only those four live, recovers at least 40% of the gap. §1805's marginals say L6-L9 carry
#          ~37 points of gap-fraction between them; if a program keeping only those four live gets 40%,
#          the marginals compose and a targeted partial compile is a real object. If FALSE they do not
#          compose, and the curve is telling us about orderings rather than about layers.
#   pred_d CONTROLS, cross-run per LESSON 42: the all-substituted and live arms reproduce §1789's
#          PUBLISHED 0.1355 / 0.1425 / 0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001; the L3, L10 and
#          L13 top-down arms reproduce §1805's PUBLISHED +0.56 / +13.86 / +19.29 pp (skip7000 and
#          matching figures for the other roles) within 0.5pp; and the bottom-up arm at L17 -- every
#          layer compiled -- must land within 2 points of gap-fraction of zero, which is the same object
#          as 'all_substituted' reached by a different code path.
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
OUT = PT + 'ops/compile_bottom_up_results.json'
PROBE_LS = tuple(range(18))
BAND = (6, 7, 8, 9)   # the four most expensive layers by §1805's marginals
S1805_TOPDOWN = {'skip7000': {3: 0.0056, 10: 0.1386, 13: 0.1929},
                 'skip11000': {3: 0.0060, 10: 0.1456, 13: 0.2073},
                 'skip1200': {3: 0.0043, 10: 0.1328, 13: 0.1872}}
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


def row_hook(full_rows):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = full_rows[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
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
    print(f'COMPILE BOTTOM-UP | both curves from one build, plus the all-but-{BAND} arm | '
          f'settled program (context-free tables + output-NN fallback + rank-{MAP_RANK} map) | '
          f'DISCOVERY ONLY', flush=True)

    def build(n):
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
    allhooks = {st: row_hook(fr[st]) for st in sites}
    evs = {e: load(p) for e, p, _ in EVAL_SETS}
    res = {}

    def run(label, hooked):
        hs = [(st, allhooks[st]) for st in hooked]
        for ename in evs:
            c = evaluate(evs[ename], hs, keep_mask)
            res.setdefault(ename, {})[label] = c
        print(f'  {label:16s} ' + '  '.join(
            f'{e} {res[e][label]["top1"]:6.2%}/{res[e][label]["top1_head"]:6.2%}'
            for e in evs), flush=True)

    print(f'\n  arms [overall/head]  ({time.time() - t0:.0f}s)', flush=True)
    run('all_substituted', sites)
    run('live_model', [])
    for L in PROBE_LS:
        # ONLY: this attention layer live, everything else substituted (reproduces §1802)
        # SUFFIX-LIVE: depths 0..L COMPILED, depths L+1..17 live -- compile from the BOTTOM UP
        run(f'L{L}_botup', [st for st in sites if st[1] <= L])
        # and the top-down arm again, so both curves come from ONE build and are directly comparable
        run(f'L{L}_topdown', [st for st in sites if st[1] > L])

    # the actionable arm §1805 pointed at: compile EVERYTHING except the expensive middle band
    run('band_live', [st for st in sites if st[1] not in BAND])
    run('band_compiled', [st for st in sites if st[1] in BAND])

    # ---- mechanism diagnostic: in the FULLY COMPILED stream, how big is what each live attention
    # module would have emitted, against the row that replaces it?  (LESSON 44: emit the quantity.)
    store = {}
    phooks = [(st, (probe_hook(fr[st], store, st[1]) if st[0] == 'attn' else allhooks[st]))
              for st in sites]
    evaluate(evs['skip7000'], phooks, keep_mask)
    norm = {L: (store[L][0] / max(store[L][2], 1), store[L][1] / max(store[L][2], 1))
            for L in sorted(store)}
    print('\n  mean ||live attn output|| vs ||substituted row||, in the compiled stream:', flush=True)
    for L in sorted(norm):
        print(f'    L{L:<2d} live {norm[L][0]:12.3f}  row {norm[L][1]:9.3f}  '
              f'ratio {norm[L][0] / max(norm[L][1], 1e-9):8.3f}', flush=True)
    del evs
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    base = {e: res[e]['all_substituted']['top1'] for e in roles}

    def d(e, label):
        return res[e][label]['top1'] - base[e]
    ratio = {L: norm[L][0] / max(norm[L][1], 1e-9) for L in norm}
    gap = {e: res[e]['live_model']['top1'] - base[e] for e in roles}
    top = {e: {L: d(e, f'L{L}_topdown') / gap[e] for L in PROBE_LS} for e in roles}
    bot = {e: {L: d(e, f'L{L}_botup') / gap[e] for L in PROBE_LS} for e in roles}
    # marginal cost of COMPILING layer L, read off each curve independently
    mtop = {e: {L: top[e][L] - (top[e][L - 1] if L else 0.0) for L in PROBE_LS} for e in roles}
    mbot = {e: {L: (bot[e][L - 1] if L else 1.0) - bot[e][L] for L in PROBE_LS} for e in roles}
    argmax_bot = {e: max(range(18), key=lambda L: mbot[e][L]) for e in roles}
    pa = all(1.0 - bot[e][3] <= 0.05 for e in roles)
    pb = all(argmax_bot[e] in BAND for e in roles)
    pc = all(d(e, 'band_live') / gap[e] >= 0.40 for e in roles)
    pd = (all(abs(res[e]['all_substituted']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['live_model']['top1'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(d(e, f'L{L}_topdown') - S1805_TOPDOWN[e][L]) <= 0.005
                  for e in roles for L in (3, 10, 13))
          and all(abs(bot[e][17] - 0.0) <= 0.02 for e in roles)
          and NFULL == NCOV)

    print('\n  gap recovered: TOP-DOWN (0..L live) vs BOTTOM-UP (L+1..17 live), skip7000',
          flush=True)
    for L in PROBE_LS:
        e = roles[0]
        print(f'    L{L:<2d}  topdown {top[e][L]:6.1%}  bottomup {bot[e][L]:6.1%}   '
              f'marginal cost of compiling L: topdown {100*mtop[e][L]:+5.1f}  '
              f'bottomup {100*mbot[e][L]:+5.1f}', flush=True)
    print(f'\n  compiling the BOTTOM FOUR layers costs <=5% of the gap -> {pa}  ' + '  '.join(
        f'{e} {100*(1.0 - bot[e][3]):.1f}%' for e in roles), flush=True)
    print(f'  the bottom-up curve\'s most expensive layer is in {BAND} -> {pb}  ' + '  '.join(
        f'{e} L{argmax_bot[e]} ({100*mbot[e][argmax_bot[e]]:+.1f})' for e in roles), flush=True)
    print(f'  compiling ALL BUT {BAND} still recovers >=40% of the gap -> {pc}  ' + '  '.join(
        f'{e} {d(e, "band_live") / gap[e]:.1%} (band compiled alone: '
        f'{d(e, "band_compiled") / gap[e]:.1%})' for e in roles), flush=True)
    print(f'  all-substituted and live reproduce §1789, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'l5_cliff_probe',
               'norm_live_vs_row': {str(L): list(norm[L]) for L in norm},
               'ratio': {str(L): ratio[L] for L in ratio},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'topdown': {e: {str(L): top[e][L] for L in PROBE_LS} for e in roles},
               'bottomup': {e: {str(L): bot[e][L] for L in PROBE_LS} for e in roles},
               'marginal_topdown': {e: {str(L): mtop[e][L] for L in PROBE_LS} for e in roles},
               'marginal_bottomup': {e: {str(L): mbot[e][L] for L in PROBE_LS} for e in roles},
               'band': list(BAND),
               'predictions': {'pred_a_bottom_four_are_free': bool(pa),
                               'pred_b_bottomup_peak_in_band': bool(pb),
                               'pred_c_band_live_recovers_40pc': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
