# WHY L5-L6?  -- the 12-point cliff §1802 found and refused to theorise about.
#
# §1802 swept all 18 attention layers, holding one live in an otherwise fully compiled program. The best
# was worth +0.38 / +0.47 / +0.35 pp. But L5 cost -12.54 / -13.11 / -12.67 pp and L6 cost -11.15 /
# -11.98 / -11.60 -- nearly the program's entire accuracy -- while the adjacent L4 cost 0.03. A
# 12-point cliff between adjacent layers, reproducing on all three roles.
#
# Two mechanisms are consistent with that and this run separates them.
#   INPUT MISMATCH: the live module receives a residual stream built from context-free rows below and
#     is out of distribution on its own input.
#   OUTPUT MISMATCH: the live module emits something the SUBSTITUTED layers above were never fitted
#     against, and the damage is done downstream of it.
# The separating arms are a SUFFIX (this layer and everything above it live) and a PREFIX (this layer
# and everything below it live). If the suffix rescues the arm, the damage was downstream; if the
# prefix rescues it, the damage was the input.
#
# It also emits the mechanism quantity directly (LESSON 44): inside the fully compiled stream, the mean
# norm of what each live attention module WOULD have emitted, against the norm of the row that replaces
# it. All 18 layers, one pass, non-invasive -- the probe substitutes exactly as before and only records.
#
# L = 4, 5, 6, 13 are probed: the two cliff layers, the adjacent layer that costs nothing, and the third
# -2pp layer as a weaker instance of whatever this is.
#
# ROLES. skip7000, skip11000, skip1200; settled program of §1786 at full coverage. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE DAMAGE IS DOWNSTREAM: the L5 SUFFIX arm lands within 2pp of the all-substituted baseline
#          or better, at every role. If FALSE the harm is not repaired by giving the layers above a real
#          module to consume, and an output-mismatch story is wrong.
#   pred_b AND IT IS NOT THE INPUT: the L5 PREFIX arm stays at least 5pp BELOW the baseline, at every
#          role. Scored separately because both could rescue -- if they do, the cliff needs both
#          conditions and neither mechanism alone explains it, which is a third outcome my binary
#          predicates could otherwise blur (LESSON 44).
#   pred_c AND L5/L6 ARE LOUD: the mean ratio ||live output|| / ||substituted row|| at L5 and L6 is at
#          least 2x its value at L4. If FALSE the cliff is not a magnitude effect and whatever pred_a
#          and pred_b establish carries the whole explanation. Reported for all 18 layers regardless.
#   pred_d CONTROLS, cross-run per LESSON 42: the all-substituted and live arms reproduce §1789's
#          PUBLISHED 0.1355 / 0.1425 / 0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001, and the four
#          'only' arms reproduce §1802's PUBLISHED deltas for L4, L5, L6 and L13 within 0.5pp --
#          confirming this script rebuilds the same cliff before it is allowed to explain it; coverage
#          is 5419 of 50257.
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
OUT = PT + 'ops/l5_cliff_probe_results.json'
PROBE_LS = (4, 5, 6, 13)
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
    print(f'L5/L6 CLIFF PROBE | only / suffix / prefix arms at L in {PROBE_LS}, plus the '
          f'live-vs-row norm diagnostic | '
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
        run(f'L{L}_only', [st for st in sites if st != ('attn', L)])
        # SUFFIX: this layer and EVERYTHING ABOVE it live -- tests downstream mismatch
        run(f'L{L}_suffix', [st for st in sites if st[1] < L])
        # PREFIX: this layer and EVERYTHING BELOW it live -- tests the input it receives
        run(f'L{L}_prefix', [st for st in sites if st[1] > L])

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
        print(f'    L{L:<2d} live {norm[L][0]:9.3f}  row {norm[L][1]:9.3f}  '
              f'ratio {norm[L][0] / max(norm[L][1], 1e-9):7.3f}', flush=True)
    del evs
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    base = {e: res[e]['all_substituted']['top1'] for e in roles}

    def d(e, label):
        return res[e][label]['top1'] - base[e]
    ratio = {L: norm[L][0] / max(norm[L][1], 1e-9) for L in norm}
    pa = all(d(e, 'L5_suffix') >= -0.02 for e in roles)
    pb = all(d(e, 'L5_prefix') <= -0.05 for e in roles)
    pc = all(ratio[L] >= 2.0 * ratio[4] for L in (5, 6))
    pd = (all(abs(res[e]['all_substituted']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['live_model']['top1'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(d(e, f'L{L}_only') - S1802_ONLY[e][L]) <= 0.005
                  for e in roles for L in PROBE_LS)
          and NFULL == NCOV)

    print('\n  delta vs the all-substituted baseline (pp):', flush=True)
    for e in roles:
        print(f'    {e:10s} ' + '  '.join(
            f'L{L}: only {100*d(e, f"L{L}_only"):+7.2f} suffix {100*d(e, f"L{L}_suffix"):+7.2f} '
            f'prefix {100*d(e, f"L{L}_prefix"):+7.2f}' for L in PROBE_LS), flush=True)
    print(f'\n  restoring EVERYTHING ABOVE L5 rescues it (within 2pp of baseline) -> {pa}  '
          + '  '.join(f'{e} {100*d(e, "L5_suffix"):+.2f}pp' for e in roles), flush=True)
    print(f'  restoring everything BELOW L5 does NOT (still <=-5pp) -> {pb}  ' + '  '.join(
        f'{e} {100*d(e, "L5_prefix"):+.2f}pp' for e in roles), flush=True)
    print(f'  L5/L6 emit >=2x the norm ratio of L4 -> {pc}  '
          f'L4 {ratio[4]:.3f}  L5 {ratio[5]:.3f}  L6 {ratio[6]:.3f}', flush=True)
    print(f'  all-substituted and live reproduce §1789, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'l5_cliff_probe',
               'norm_live_vs_row': {str(L): list(norm[L]) for L in norm},
               'ratio': {str(L): ratio[L] for L in ratio},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'predictions': {'pred_a_suffix_rescues_L5': bool(pa),
                               'pred_b_prefix_does_not_rescue_L5': bool(pb),
                               'pred_c_L5L6_norm_ratio_is_large': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
