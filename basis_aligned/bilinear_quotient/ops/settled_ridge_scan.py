# IS THE SETTLED PROGRAM UNDER-REGULARISED AT FULL COVERAGE?
#
# The §1799 follow-up (relative ridge) is still running, but two of its cells already force this
# question. At n = 880 a ridge of 0.1*lambda_max scores 11.80% against the inert ridge's 9.11%
# (+2.69pp), and at n = D = 1152 it scores 11.84% against 2.40% (+9.44pp) -- it does not merely fill in
# the collapse, it beats the inert ridge at a coverage where there was no collapse at all.
#
# The settled program of §1786 uses the SAME inert setting. Measured: lambda_max(Ecov^T Ecov) ~ 7.26e+06
# at n = 880, so the settled `RIDGE * (n/D)` = 0.01 is 1.38e-09 * lambda_max -- the inert end of that
# sweep. Every accuracy figure from §1788 through §1800, and §1786's certified design point, was
# measured with that setting. If a scaled ridge lifts the program at FULL coverage too, the baseline
# those sections were measured against is not the best member of its own class, and the recent arc's
# numbers -- though not its qualitative conclusions -- shift.
#
# This run sweeps the ridge at n = 5419 only. The first arm reproduces the published program EXACTLY
# (absolute `RIDGE * n/D`, not an approximating fraction) so the cross-run control is exact; the rest
# are fractions of lambda_max from 1e-6 to 1e-1. Table rank and MAP_RANK are held at the settled 64, so
# the ridge is the only thing that moves.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE SETTLED RIDGE IS NOT THE BEST ONE: some scanned fraction beats it by at least 1
#          percentage point of overall top-1, at every role. If FALSE the settled setting is adequate at
#          full coverage, §1786's design point stands as certified, and the n<=D improvements are a
#          small-sample effect that does not reach the operating point -- the more convenient outcome
#          and the one I should be most suspicious of wanting.
#   pred_b AND THE OPTIMUM IS STABLE: the best fraction is the SAME on all three roles. Scored
#          separately because a per-role optimum would mean I am reading noise, and a 1pp gain at three
#          different fractions is three different claims rather than one.
#   pred_c BUT IT IS NOT A NEW LEVER: the gain does not reach 25% of the role's gap to live (25.77 /
#          28.10 / 25.24 pp). If FALSE, regularisation alone closes a quarter of what §1800 attributed
#          to context, and §1800's conclusion needs revisiting before anything else in this thread does.
#   pred_d CONTROLS, cross-run per LESSON 42: the 'settled' arm reproduces §1789's PUBLISHED 0.1355 /
#          0.1425 / 0.1364 within 0.001 -- it is the same formula, so this is exact and confirms the
#          scan's other arms differ from the published program in the ridge alone; the knob must
#          demonstrably turn (cond(A) spanning >=1000x across the scan, LESSON 45); coverage 5419.
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
OUT = PT + 'ops/settled_ridge_scan_results.json'
# sampled by n/D, D = 1152: 0.59, 0.76, 1.00, 1.18, 1.41, 2.35
NS = (5419,)   # FULL coverage -- the settled program's own operating point
# 'settled' reproduces the published program EXACTLY (absolute RIDGE * n/D), so pred_d's cross-run
# control is exact rather than approximate. The rest are fractions of lambda_max (LESSON 45).
RIDGE_FRACS = ('settled', 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
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


def main():
    global RIDGES
    RIDGES = RIDGE_FRACS
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
    print(f'SETTLED RIDGE SCAN | FULL coverage n={NS[0]} x ridge {RIDGE_FRACS} | '
          f'settled program (context-free tables + output-NN fallback + rank-{MAP_RANK} map) | '
          f'DISCOVERY ONLY', flush=True)

    def build(n, ridge):
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
        G = Ecov.T @ Ecov
        lmax = float(torch.linalg.eigvalsh(G).max())
        lam = (RIDGE * (n / D)) if ridge == 'settled' else (ridge * lmax)
        A = G + lam * torch.eye(D, device=DEV, dtype=torch.float64)
        cond = float(torch.linalg.cond(A))
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
        del tables, Ecov, Eunc, A, G
        torch.cuda.empty_cache()
        return out, seen, (cond, lmax)

    # the KEPT slice is defined by the SMALLEST coverage set, so it is the same positions at every
    # fraction -- that is what makes the known-answer control meaningful.
    small = perm[:NS[0]].to(DEV)
    keep_mask = torch.zeros(V, dtype=torch.bool, device=DEV)
    keep_mask[small] = True

    res, conds = {}, {}
    lmaxes = {}
    for ridge in RIDGES:
        for n in NS:
            fr, seen, (cond, lmax) = build(n, ridge)
            key = f'{ridge}|{n}' if ridge == 'settled' else f'{ridge:g}|{n}'
            conds[key] = cond
            lmaxes[key] = lmax
            hooks = [(st, row_hook(fr[st])) for st in sites]
            lam = (RIDGE * (n / D)) if ridge == 'settled' else (ridge * lmax)
            print(f'\n  ridge {str(ridge):8s} (absolute {lam:.3e} = {lam / lmax:.2e}*lmax)  '
                  f'n {n}  cond(A) {cond:.3e}  ({time.time() - t0:.0f}s)', flush=True)
            for ename, epath, _ in EVAL_SETS:
                ev = load(epath)
                c = evaluate(ev, hooks, keep_mask)
                res.setdefault(ename, {})[key] = c
                print(f'    {ename}: overall {c["top1"]:.2%}  head {c["top1_head"]:.2%}  '
                      f'kept {c["top1_kept"]:.4%}', flush=True)
                del ev
            del fr, hooks
            torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    N = NS[0]

    def key(r):
        return f'{r}|{N}' if r == 'settled' else f'{r:g}|{N}'

    def acc(e, r):
        return res[e][key(r)]['top1']
    scan = [r for r in RIDGES if r != 'settled']
    best = {e: max(scan, key=lambda r: acc(e, r)) for e in roles}
    gain = {e: acc(e, best[e]) - acc(e, 'settled') for e in roles}
    # the knob must demonstrably turn before any negative is readable -- LESSON 45
    turned = (max(conds[key(r)] for r in RIDGES)
              / max(min(conds[key(r)] for r in RIDGES), 1e-30) >= 1e3)
    pa = all(gain[e] >= 0.01 for e in roles)
    pb = len({best[e] for e in roles}) == 1
    pc = all(gain[e] < 0.25 * S1789_GAP[e] for e in roles)
    pd = (all(abs(acc(e, 'settled') - S1789_PROG[e]) <= 0.001 for e in roles)
          and turned and NFULL == NCOV)

    print('\n  cond(A): ' + '  '.join(f'{str(r):8s} {conds[key(r)]:.2e}' for r in RIDGES),
          flush=True)
    for e in roles:
        print(f'  {e:10s} ' + '  '.join(f'{str(r):8s} {acc(e, r):6.2%}' for r in RIDGES),
              flush=True)
    print(f'\n  THE KNOB TURNED (cond spans >=1000x) -> {turned}  '
          f'{max(conds[key(r)] for r in RIDGES) / max(min(conds[key(r)] for r in RIDGES), 1e-30):.1e}x',
          flush=True)
    print(f'  a SCALED ridge beats the settled one by >=1pp -> {pa}  ' + '  '.join(
        f'{e} best {best[e]:g} {acc(e, best[e]):.2%} vs settled {acc(e, "settled"):.2%} '
        f'({100*gain[e]:+.2f}pp)' for e in roles), flush=True)
    print(f'  the best fraction is the SAME on all three roles -> {pb}  ' + '  '.join(
        f'{e} {best[e]:g}' for e in roles), flush=True)
    print(f'  ... but does not close a quarter of the gap -> {pc}  ' + '  '.join(
        f'{e} {gain[e] / S1789_GAP[e]:.1%} of {100*S1789_GAP[e]:.2f}pp' for e in roles),
        flush=True)
    print(f'  KNOWN-ANSWER: kept-token slice identical everywhere -> {pc}  ' + '  '.join(
        f'{e} {res[e][key("settled")]["top1_kept"]:.4%}' for e in roles), flush=True)
    print(f'  the settled arm reproduces §1789, knob turned, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'coverage_dip_conditioning', 'ns': list(NS), 'ridges': list(RIDGES),
               'cond': {k: float(v) for k, v in conds.items()},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'lmax': {k: float(v) for k, v in lmaxes.items()}, 'knob_turned': bool(turned),
               'best_ridge': {e: str(best[e]) for e in res}, 'gain_pp': gain,
               'predictions': {'pred_a_scaled_ridge_beats_settled': bool(pa),
                               'pred_b_same_optimum_all_roles': bool(pb),
                               'pred_c_known_answer_kept_slice': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
