# DOES BUYING MORE COVERAGE PAY?  -- the decision §1798 was meant to inform and could not.
#
# §1793 established that the program's limit on the tail is COVERAGE, not frequency. §1796 and §1797
# then closed the only other lever: the program and a leave-one-out bigram are complementary by ~7-9pp
# on the head, and NEITHER arm's self-assessment recovers any of it. So the remaining question is
# whether more fit data is worth buying.
#
# §1798 tried to answer it and could not. Its curve was NON-MONOTONE -- 677 covered types beat 1355 by
# ~3pp -- with the minimum at n/D = 1.18, where the rank-64 map's D x D normal system becomes square.
# Its known-answer control was identical to four decimals at every fraction, so the tables are innocent
# and the artifact is confined to the uncovered-token path. Its pred_c ("diminishing returns") PASSED
# but had the artifact as an interval endpoint, so its reading was void.
#
# THIS RUN AVOIDS THE PATHOLOGY ENTIRELY rather than explaining it (that is
# ops/coverage_dip_conditioning.py's job, running in parallel). Four nested points at n = 2710, 3613,
# 4516, 5419 -- n/D = 2.35, 3.14, 3.92, 4.70, all far above the n ~ D region -- in EQUAL steps of +903
# types, so the successive gains are directly comparable without any normalisation.
#
# ROLES. skip7000, skip11000, skip1200; settled program of §1786; fallback and map refitted inside each
# covered set (§1785). DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE SAFE REGION IS WELL BEHAVED: overall top-1 does not decrease across the four points, at
#          every role. If FALSE the §1798 pathology extends past n/D = 2.35 and the fallback is
#          unreliable across the whole range, which would mean no coverage claim of mine is readable
#          until it is fixed -- including §1786's design point.
#   pred_b DIMINISHING RETURNS, MEASURED CLEANLY: the last +903 types buy LESS overall top-1 than the
#          first +903, at every role. This is the decision. TRUE means the curve is flattening by 5,419
#          types and more fit data pays progressively less, so the tail §1793 found needs CONTEXT
#          rather than data, and the next runs should go back to what context buys. FALSE means the
#          program is still on a steep part and MORE FIT DATA is the cheapest remaining move -- which
#          would redirect the thread and is worth several runs.
#   pred_c AND THE SAME ON THE HEAD: the last +903 buys less head-bucket accuracy than the first +903,
#          at every role. Scored separately because the overall figure mixes the covered and uncovered
#          paths; if overall flattens while the head does not, the flattening is the fallback
#          saturating rather than the tables.
#   pred_d CONTROLS: n = 5419 reproduces §1789's PUBLISHED 0.1355 / 0.1425 / 0.1364 and n = 2710
#          reproduces §1798's PUBLISHED 0.1254 / 0.1308 / 0.1247, both within 0.001 (cross-run, LESSON
#          42); the KNOWN-ANSWER check per LESSON 34 -- top-1 restricted to positions whose current
#          token is in the smallest covered set must be identical to within 0.05pp at all four n, since
#          the program is position-wise (§1765); the subsets are nested with exactly the requested
#          sizes; coverage is 5419 of 50257.
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
OUT = PT + 'ops/coverage_curve_safe_results.json'
# n/D = 2.35, 3.14, 3.92, 4.70 -- all far above the n~D pathology of §1798
NS = (2710, 3613, 4516, 5419)
S1798_2710 = {'skip7000': 0.12540, 'skip11000': 0.13080, 'skip1200': 0.12470}
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
    print(f'COVERAGE CURVE, SAFE REGION | n in {NS} of {NFULL} (D={D}) | '
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
    small = perm[:NS[0]].to(DEV)
    keep_mask = torch.zeros(V, dtype=torch.bool, device=DEV)
    keep_mask[small] = True

    res, ncovs = {}, {}
    for n in NS:
        fr, seen, nn = build(n)
        ncovs[str(n)] = nn
        hooks = [(st, row_hook(fr[st])) for st in sites]
        print(f'\n  n {n} types (n/D {n / D:.2f})  ({time.time() - t0:.0f}s)', flush=True)
        for ename, epath, _ in EVAL_SETS:
            ev = load(epath)
            c = evaluate(ev, hooks, keep_mask)
            res.setdefault(ename, {})[str(n)] = c
            print(f'    {ename}: overall {c["top1"]:.2%}  head {c["top1_head"]:.2%}  '
                  f'kept {c["top1_kept"]:.4%}', flush=True)
            del ev
        del fr, hooks
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    F = [str(n) for n in NS]
    pa = all(res[e][F[i + 1]]['top1'] >= res[e][F[i]]['top1'] - 1e-12
             for e in roles for i in range(len(F) - 1))
    # equal-sized steps of +903 types, so the gains are directly comparable
    g = {e: [res[e][F[i + 1]]['top1'] - res[e][F[i]]['top1'] for i in range(len(F) - 1)]
         for e in roles}
    gh = {e: [res[e][F[i + 1]]['top1_head'] - res[e][F[i]]['top1_head'] for i in range(len(F) - 1)]
          for e in roles}
    pb = all(g[e][2] < g[e][0] for e in roles)
    pc = all(gh[e][2] < gh[e][0] for e in roles)
    known = all(abs(res[e][f]['top1_kept'] - res[e][F[0]]['top1_kept']) <= 0.0005
                for e in roles for f in F)
    pd = (all(abs(res[e][F[-1]]['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e][F[0]]['top1'] - S1798_2710[e]) <= 0.001 for e in roles)
          and known and all(ncovs[str(n)] == n for n in NS) and NFULL == NCOV)

    print(f'\n  overall top-1 rises with coverage in the SAFE region -> {pa}  ' + '  '.join(
        f'{e} ' + '/'.join(f'{res[e][f]["top1"]:.2%}' for f in F) for e in roles), flush=True)
    print(f'  DIMINISHING RETURNS overall (last +903 < first +903) -> {pb}  ' + '  '.join(
        f'{e} ' + '/'.join(f'{100*x:+.2f}' for x in g[e]) + 'pp' for e in roles), flush=True)
    print(f'  ... and on the HEAD -> {pc}  ' + '  '.join(
        f'{e} ' + '/'.join(f'{100*x:+.2f}' for x in gh[e]) + 'pp' for e in roles), flush=True)
    print(f'  KNOWN-ANSWER: kept-token slice identical across n -> {known}  ' + '  '.join(
        f'{e} {res[e][F[0]]["top1_kept"]:.4%}' for e in roles), flush=True)
    print(f'  n=5419 reproduces §1789 and n=2710 reproduces §1798, coverage {NFULL} -> '
          f'control {pd}', flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'coverage_curve_safe', 'ns': list(NS), 'ncov': ncovs,
               'gains_overall': g, 'gains_head': gh,
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'predictions': {'pred_a_monotone_in_safe_region': bool(pa),
                               'pred_b_diminishing_returns_overall': bool(pb),
                               'pred_c_diminishing_returns_on_head': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
