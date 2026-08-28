# IS THE n=D COLLAPSE CONDITIONING?  -- asked again, with a ridge that actually moves.
#
# §1799 established WHERE: at n = D = 1152 exactly, the program collapses to 2.35 / 2.44 / 2.22% top-1,
# below every other coverage including 677 types, while its known-answer control stayed identical to
# four decimals in all twelve cells -- so the tables are untouched and the fault is entirely in the
# uncovered-token path.
#
# It established NOTHING about why. Its ridge sweep raised an ABSOLUTE ridge by 100x and produced
# byte-identical condition numbers at n >= 1355 and identical accuracy at five of six points: the ridge
# enters as `ridge * I * (n/D)` against a data term whose eigenvalues are orders of magnitude larger, so
# "100x the settled value" was still numerically zero. LESSON 45.
#
# HERE THE RIDGE IS A FRACTION OF lambda_max(Ecov^T Ecov), computed per cell. 1e-8 is inert by
# construction and 1e-1 dominates the data term by construction, so the sweep is known to bracket the
# behaviour rather than hoped to. cond(A) and lambda_max are recorded in every cell, and pred_d refuses
# to pass unless the condition number actually fell by >=1000x across the sweep -- the negative from
# §1799 is not repeatable here, because a sweep that does not move its own knob cannot be read.
#
# n is narrowed to 880 / 1152 / 1355 / 1620 (n/D = 0.76, 1.00, 1.18, 1.41), which brackets the collapse;
# the far points are already settled by §1798/§1799 and cost 5 minutes each to re-derive.
#
# ROLES. skip7000, skip11000, skip1200; settled program of §1786. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a A RELATIVE RIDGE FILLS IT IN: at ridge = 1e-1 * lambda_max, top-1 at n = D exceeds its value
#          at the inert ridge by at least 3 percentage points, at every role. If FALSE -- with the knob
#          demonstrably turned, which pred_d now enforces -- the collapse is NOT the conditioning of
#          this solve, and the output-NN fallback becomes the sole remaining suspect: its neighbour set
#          is also rebuilt per coverage, and the next run must vary the fallback while holding the map
#          fixed.
#   pred_b AND THE CURVE BECOMES MONOTONE: at that ridge, top-1 does not decrease across 880 -> 1152 ->
#          1355 -> 1620, at every role. Scored separately from pred_a because filling the hole partially
#          would satisfy a 3pp bar while leaving a curve that still turns over -- which would say the
#          ridge treats a symptom rather than the cause.
#   pred_c KNOWN-ANSWER, per LESSON 34: top-1 restricted to positions whose current token is in the
#          smallest covered set (n = 880) is identical to within 0.05pp across ALL sixteen cells. A
#          covered token's table is its own length-1 forward and the program is position-wise (§1765);
#          no ridge can reach such a position. If this moves, nothing else here is readable.
#   pred_d CONTROLS: at the inert ridge, n = 1152 and n = 1355 reproduce §1799's PUBLISHED 0.0235 /
#          0.0244 / 0.0222 and 0.0694 / 0.0710 / 0.0679 within 0.005 (cross-run, LESSON 42; the wider
#          tolerance is because the inert ridge is not bit-identical to the settled one); AND the knob
#          must have turned -- cond(A) at the inert ridge is at least 1000x its value at the largest,
#          at every n. Without that this run repeats §1799's mistake and no prediction is readable.
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
OUT = PT + 'ops/coverage_dip_relative_ridge_results.json'
# sampled by n/D, D = 1152: 0.59, 0.76, 1.00, 1.18, 1.41, 2.35
NS = (880, 1152, 1355, 1620)
# ridge as a FRACTION of lambda_max(Ecov^T Ecov) -- LESSON 45. 1e-8 is inert by construction,
# 1e-1 dominates the data term by construction, so the sweep is known to bracket the behaviour.
RIDGE_FRACS = (1e-8, 1e-5, 1e-3, 1e-1)
S1799 = {'skip7000': {1152: 0.02350, 1355: 0.06940},
         'skip11000': {1152: 0.02440, 1355: 0.07100},
         'skip1200': {1152: 0.02220, 1355: 0.06790}}
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
    print(f'COVERAGE DIP, RELATIVE RIDGE | n in {NS} (D={D}) x ridge {RIDGE_FRACS} * lmax | '
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
        A = G + (ridge * lmax) * torch.eye(D, device=DEV, dtype=torch.float64)
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
            key = f'{ridge:g}|{n}'
            conds[key] = cond
            lmaxes[key] = lmax
            hooks = [(st, row_hook(fr[st])) for st in sites]
            print(f'\n  ridge {ridge:g}*lmax ({ridge * lmax:.3e})  n {n} (n/D {n / D:.3f})  '
                  f'cond(A) {cond:.3e}  ({time.time() - t0:.0f}s)', flush=True)
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
    lo, hi = RIDGES[0], RIDGES[-1]

    def acc(e, r, n):
        return res[e][f'{r:g}|{n}']['top1']
    # the knob must demonstrably turn before any negative is readable -- LESSON 45
    turned = all(conds[f'{lo:g}|{n}'] / max(conds[f'{hi:g}|{n}'], 1e-30) >= 1e3 for n in NS)
    pa = all(acc(e, hi, 1152) - acc(e, lo, 1152) >= 0.03 for e in roles)
    pb = all(acc(e, hi, n2) >= acc(e, hi, n1) - 1e-12
             for e in roles for n1, n2 in zip(NS, NS[1:]))
    pc = all(abs(res[e][f'{r:g}|{n}']['top1_kept']
                 - res[e][f'{lo:g}|{NS[0]}']['top1_kept']) <= 0.0005
             for e in roles for r in RIDGES for n in NS)
    pd = (all(abs(acc(e, lo, 1152) - S1799[e][1152]) <= 0.005
              and abs(acc(e, lo, 1355) - S1799[e][1355]) <= 0.005 for e in roles)
          and turned and NFULL == NCOV)

    for r in RIDGES:
        print(f'\n  cond(A) at ridge {r:g}*lmax: ' + '  '.join(
            f'n {n} {conds[f"{r:g}|{n}"]:.2e}' for n in NS), flush=True)
    print(f'\n  THE KNOB TURNED (cond drops >=1000x from {lo:g} to {hi:g}) -> {turned}  '
          + '  '.join(f'n {n} {conds[f"{lo:g}|{n}"] / max(conds[f"{hi:g}|{n}"], 1e-30):.1e}x'
                      for n in NS), flush=True)
    print(f'  a RELATIVE ridge fills in the n=D collapse by >=3pp -> {pa}  ' + '  '.join(
        f'{e} {acc(e, hi, 1152):.2%} vs {acc(e, lo, 1152):.2%} '
        f'({100*(acc(e, hi, 1152) - acc(e, lo, 1152)):+.2f}pp)' for e in roles), flush=True)
    print(f'  ... and the whole curve becomes monotone -> {pb}  ' + '  '.join(
        f'{e} ' + '/'.join(f'{acc(e, hi, n):.2%}' for n in NS) for e in roles), flush=True)
    print(f'  KNOWN-ANSWER: kept-token slice identical everywhere -> {pc}  ' + '  '.join(
        f'{e} {res[e][f"{lo:g}|{NS[0]}"]["top1_kept"]:.4%}' for e in roles), flush=True)
    print(f'  inert-ridge n=1152 and n=1355 reproduce §1799, knob turned, coverage {NFULL} '
          f'-> control {pd}', flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'coverage_dip_conditioning', 'ns': list(NS), 'ridges': list(RIDGES),
               'cond': {k: float(v) for k, v in conds.items()},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'lmax': {k: float(v) for k, v in lmaxes.items()}, 'knob_turned': bool(turned),
               'predictions': {'pred_a_relative_ridge_fills_the_collapse': bool(pa),
                               'pred_b_curve_becomes_monotone': bool(pb),
                               'pred_c_known_answer_kept_slice': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
