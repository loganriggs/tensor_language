# IS THE §1798 DIP THE RIDGE SOLVE'S CONDITIONING?  -- the question §1798 ended on.
#
# §1798 found the coverage curve is NOT monotone: 677 covered types beat 1355 by 2.98 / 3.49 / 3.14 pp
# on all three roles. Its known-answer control was identical to four decimals at every fraction
# (15.4576 / 17.6908 / 14.1635%), so the tables and §1765's position-wise property are untouched and
# the entire dip lives in the uncovered-token path -- the output-NN fallback and the rank-64
# embedding->row map, both refitted inside each covered set.
#
# The map solves a D x D normal system from n covered rows with D = 1152. The four sampled fractions
# gave n/D = 4.704, 2.352, 1.176, 0.588 and the minimum fell at 1.176 -- where the fit is most nearly
# square. That is the shape of an ill-conditioned interpolation threshold rather than of a coverage
# effect, and it is directly testable.
#
# DESIGN. Sample n around D: 677, 880, 1152, 1355, 1620, 2710 (n/D = 0.59, 0.76, 1.00, 1.18, 1.41,
# 2.35), each at the settled ridge and at 100x that ridge. cond(A) is recorded for every cell, so the
# mechanism is measured and not inferred from the accuracy shape alone (LESSON 44: emit the
# discriminating quantity, not only the bar). Subsets stay NESTED from one fixed permutation.
#
# ROLES. skip7000, skip11000, skip1200; settled program of §1786. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE DIP IS AT n ~ D: at the settled ridge, every n with 0.9 <= n/D <= 1.25 scores BELOW both
#          n = 677 and n = 2710, at every role. If FALSE the minimum is somewhere else and the n/D
#          alignment in §1798 was a coincidence of where I happened to sample -- in which case the
#          fallback has a different defect and §1798's diagnosis, not just its curve, is wrong.
#   pred_b RAISING THE RIDGE REMOVES IT: at 100x the settled ridge, overall top-1 is monotone
#          non-decreasing in n across all six values, at every role. If FALSE the pathology is not the
#          conditioning of this solve -- it would then most likely be the output-NN fallback, whose
#          neighbour set also shrinks with coverage, and the next run must separate those two rather
#          than tune the ridge.
#   pred_c KNOWN-ANSWER, per LESSON 34: top-1 restricted to positions whose current token is in the
#          SMALLEST covered set (n = 677) must be identical to within 0.05pp across ALL twelve cells,
#          both ridges included. A covered token's table is its own length-1 forward and the program is
#          position-wise (§1765); neither the ridge nor the neighbour set can reach such a position. If
#          this moves, the build is wrong and no other number here is readable.
#   pred_d CONTROLS, cross-run per LESSON 42: at the settled ridge, n = 677 and n = 1355 reproduce
#          §1798's PUBLISHED 0.0992 / 0.1059 / 0.0993 and 0.0694 / 0.0710 / 0.0679 within 0.001 --
#          confirming this script rebuilds the same objects before it is allowed to explain them;
#          coverage is exactly 5419 of 50257.
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
OUT = PT + 'ops/coverage_dip_conditioning_results.json'
# sampled by n/D, D = 1152: 0.59, 0.76, 1.00, 1.18, 1.41, 2.35
NS = (677, 880, 1152, 1355, 1620, 2710)
RIDGES = None  # set in main() from the module RIDGE
S1798 = {'skip7000': {677: 0.09920, 1355: 0.06940}, 'skip11000': {677: 0.10590, 1355: 0.07100},
         'skip1200': {677: 0.09930, 1355: 0.06790}}
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
    RIDGES = (RIDGE, 100.0 * RIDGE)
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
    print(f'COVERAGE DIP CONDITIONING | n in {NS} (D={D}) x ridge {RIDGES} | '
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
        A = Ecov.T @ Ecov + ridge * torch.eye(D, device=DEV, dtype=torch.float64) * (n / D)
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
        del tables, Ecov, Eunc, A
        torch.cuda.empty_cache()
        return out, seen, cond

    # the KEPT slice is defined by the SMALLEST coverage set, so it is the same positions at every
    # fraction -- that is what makes the known-answer control meaningful.
    small = perm[:NS[0]].to(DEV)
    keep_mask = torch.zeros(V, dtype=torch.bool, device=DEV)
    keep_mask[small] = True

    res, conds = {}, {}
    for ridge in RIDGES:
        for n in NS:
            fr, seen, cond = build(n, ridge)
            key = f'{ridge:g}|{n}'
            conds[key] = cond
            hooks = [(st, row_hook(fr[st])) for st in sites]
            print(f'\n  ridge {ridge:g}  n {n} (n/D {n / D:.3f})  cond(A) {cond:.3e}  '
                  f'({time.time() - t0:.0f}s)', flush=True)
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
    NEAR = [n for n in NS if 0.9 <= n / D <= 1.25]
    FAR = [NS[0], NS[-1]]

    def acc(e, r, n):
        return res[e][f'{r:g}|{n}']['top1']
    pa = all(max(acc(e, lo, n) for n in NEAR) < min(acc(e, lo, n) for n in FAR) for e in roles)
    pb = all(acc(e, hi, n2) >= acc(e, hi, n1) - 1e-12
             for e in roles for n1, n2 in zip(NS, NS[1:]))
    pc = all(abs(res[e][f'{r:g}|{n}']['top1_kept']
                 - res[e][f'{lo:g}|{NS[0]}']['top1_kept']) <= 0.0005
             for e in roles for r in RIDGES for n in NS)
    pd = (all(abs(acc(e, lo, 677) - S1798[e][677]) <= 0.001
              and abs(acc(e, lo, 1355) - S1798[e][1355]) <= 0.001 for e in roles)
          and NFULL == NCOV)

    print(f'\n  cond(A) by n at ridge {lo:g}: ' + '  '.join(
        f'n {n} {conds[f"{lo:g}|{n}"]:.2e}' for n in NS), flush=True)
    print(f'  cond(A) by n at ridge {hi:g}: ' + '  '.join(
        f'n {n} {conds[f"{hi:g}|{n}"]:.2e}' for n in NS), flush=True)
    print(f'\n  at the LOW ridge the dip sits at n~D -> {pa}  ' + '  '.join(
        f'{e} near ' + '/'.join(f'{acc(e, lo, n):.2%}' for n in NEAR)
        + ' vs far ' + '/'.join(f'{acc(e, lo, n):.2%}' for n in FAR) for e in roles), flush=True)
    print(f'  RAISING THE RIDGE removes the non-monotonicity -> {pb}  ' + '  '.join(
        f'{e} ' + '/'.join(f'{acc(e, hi, n):.2%}' for n in NS) for e in roles), flush=True)
    print(f'  KNOWN-ANSWER: kept-token slice identical everywhere -> {pc}  ' + '  '.join(
        f'{e} {res[e][f"{lo:g}|{NS[0]}"]["top1_kept"]:.4%}' for e in roles), flush=True)
    print(f'  low-ridge n=677 and n=1355 reproduce §1798, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'coverage_dip_conditioning', 'ns': list(NS), 'ridges': list(RIDGES),
               'cond': {k: float(v) for k, v in conds.items()},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'predictions': {'pred_a_dip_sits_at_n_equals_D': bool(pa),
                               'pred_b_ridge_removes_nonmonotonicity': bool(pb),
                               'pred_c_known_answer_kept_slice': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
