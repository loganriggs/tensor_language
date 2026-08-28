# DID §1790's BIGRAM SEE THE ANSWER?  -- auditing a published claim of my own before retracting it.
#
# The v2 rank run failed its cross-run control a SECOND time, and this time not on the new code.
# Reproducing §1790's leave-one-out bigram by two independent leak-free routes gives top-1
# 12.23 / 12.73 / 11.88% (raw-count argmax) and 12.48 / 12.93 / 12.32% (unigram tie-break), against
# the 15.97 / 16.63 / 18.00% §1790 PUBLISHED. Under either leak-free policy the PROGRAM (13.55 /
# 14.25 / 13.64%) BEATS the bigram at every role, reversing §1790's headline.
#
# THE SUSPECTED MECHANISM, stated before the run so the run can refute it. §1790 wrote
#     c1 = v[..., 0] - (k[..., 0] == tg).float()
#     return torch.where(c1 >= v[..., 1], k[..., 0], k[..., 1])
# When the target is the top-1 and its leave-one-out count merely TIES the runner-up, `>=` keeps the
# target. The observation the leave-one-out was supposed to remove still decides the prediction. With
# ~6.8 eval observations per covered current type almost every count is 1-3, so ties are pervasive and
# the arm is held on the answer exactly where the evidence has been withdrawn.
#
# This run puts §1790's verbatim code path beside a leak-free one on IDENTICAL positions and counts
# how many of its hits are held by such a tie. DISCOVERY ONLY -- and its conclusion, if pred_d holds,
# requires RETRACTING a published claim, which is Logan's call and not mine.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39:
#   pred_a THE ARM IS INFLATED: §1790's loo_argmax beats a leak-free argmax on the same positions by
#          at least 2 percentage points, at every role. If FALSE the discrepancy lies somewhere other
#          than the tie-break and §1790 may still stand -- I would have accused it wrongly.
#   pred_b AND TIES ARE THE MECHANISM: at least 20% of §1790's correct predictions are positions where
#          the target was its top-1 and, after the decrement, only TIED the runner-up. Scored
#          separately because pred_a establishes a gap and this establishes its cause; a large gap
#          with few tie-held hits would mean the right symptom and the wrong diagnosis.
#   pred_c AND THE HEADLINE REVERSES: leak-free, the program's top-1 exceeds the bigram's at every
#          role. If FALSE the bigram still wins and only the margin was wrong, which needs a
#          correction but not a retraction.
#   pred_d CONTROLS, and this one is the whole run per LESSON 42: the §1790 arm must reproduce
#          §1790's PUBLISHED 0.1597 / 0.1663 / 0.1800 within 0.001 -- otherwise I have not re-created
#          the code path I am accusing and NO conclusion may be drawn in either direction; program and
#          live must reproduce §1789's 0.1355 / 0.1425 / 0.1364 and 0.3932 / 0.4235 / 0.3888 within
#          0.001; the fit-row bigram's covered CE must reproduce §1767's 7.88804 / 7.90729 within
#          0.01; coverage exactly 5419 of 50257.
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
OUT = PT + 'ops/loo_tiebreak_audit_results.json'
KS = (1, 5, 10, 50, 100)
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
RB = 4
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
def audit(rows, hooks, cnt_bank, ncov):
    """Re-run §1790's EXACT loo_argmax beside a leak-free one, on identical positions.

    §1790 chose `torch.where(c1 >= v1, k0, k1)` where `c1 = v0 - (k0 == tg)`. When the target is the
    top-1 and its leave-one-out count TIES the runner-up, `>=` keeps the target -- so the prediction
    is held on the answer by a tie the removed observation was supposed to break. The leak-free arm
    decrements the target's own cell and then takes the argmax over the whole row."""
    top2 = cnt_bank.topk(2, dim=-1)
    a = {'n': 0, 's1790': 0, 'clean': 0, 's1790_via_tie': 0, 'prog': 0, 'live': 0}
    for i in range(0, rows.shape[0], 4):
        bb = rows[i:i + 4]
        idx = bb[:, :-1].to(DEV).contiguous()
        cur = idx[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        r = COV['idmap'][cur]
        r = torch.where(r >= 0, r, torch.full_like(r, ncov))

        # --- §1790's exact code path, copied verbatim from ops/bigram_reachable_accuracy.py
        v, k = top2.values[r], top2.indices[r]
        c1 = v[..., 0] - (k[..., 0] == tg).float()
        s1790 = torch.where(c1 >= v[..., 1], k[..., 0], k[..., 1])

        # --- leak-free: remove the observation, then argmax the row
        c = cnt_bank[r]
        own = c.gather(-1, tg.unsqueeze(-1))
        c.scatter_(-1, tg.unsqueeze(-1), own - 1.0)
        clean = (c + 0.5 * COV['back']).argmax(-1)
        del c

        hit90 = (s1790 == tg)
        # the mechanism: §1790 was right, the target was its top-1, and after removing the
        # observation the target only TIED the runner-up rather than beating it
        tie = hit90 & (k[..., 0] == tg) & (c1 == v[..., 1])
        a['n'] += int(tg.numel())
        a['s1790'] += int(hit90.sum())
        a['clean'] += int((clean == tg).sum())
        a['s1790_via_tie'] += int(tie.sum())
        a['prog'] += int((forward_logits(idx, hooks)[:, 64:].argmax(-1) == tg).sum())
        a['live'] += int((forward_logits(idx)[:, 64:].argmax(-1) == tg).sum())
        torch.cuda.empty_cache()
    n = a['n']
    return {'acc_s1790': a['s1790'] / n, 'acc_clean': a['clean'] / n,
            'acc_prog': a['prog'] / n, 'acc_live': a['live'] / n,
            'tie_share_of_s1790_hits': a['s1790_via_tie'] / max(a['s1790'], 1),
            's1790_hits': a['s1790'], 'tie_hits': a['s1790_via_tie'], 'n': n}


def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    seen_cpu = torch.zeros(V, dtype=torch.bool)
    seen_cpu[fit[:, :T].reshape(-1).long()] = True
    ncov = int(seen_cpu.sum())
    assert ncov == NCOV, f'coverage {ncov} != {NCOV}'
    seen = seen_cpu.to(DEV)
    COV['seen'] = seen
    toks = seen_cpu.nonzero(as_tuple=True)[0]
    tk = toks.to(DEV)
    unc = (~seen).nonzero(as_tuple=True)[0]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    COV['freq'] = torch.bincount(fit[:, 1:T + 1].reshape(-1).long(),
                                 minlength=V).to(DEV)

    # ---- arm 3: the FIT-ROW bigram, the fair floor (same rows the program was fitted on).
    # add-alpha with unigram backoff, alpha 0.01 as selected in §1767; uncovered current tokens
    # fall back to the unigram argmax, which is what a bigram-with-backoff actually predicts there.
    cur = fit[:, :T].reshape(-1).long()
    nxt = fit[:, 1:T + 1].reshape(-1).long()
    idmap = torch.full((V,), -1, dtype=torch.long)
    idmap[seen_cpu.nonzero(as_tuple=True)[0]] = torch.arange(ncov)
    ri = idmap[cur]
    keep = ri >= 0
    cnts = torch.zeros(ncov, V, dtype=torch.float32, device=DEV)
    cnts.index_put_((ri[keep].to(DEV), nxt[keep].to(DEV)),
                    torch.ones(int(keep.sum()), device=DEV), accumulate=True)
    uni = cnts.sum(0)
    back = torch.softmax(torch.log((uni + 1.0) / (uni.sum() + V)), 0).unsqueeze(0)
    pfit = (cnts + ALPHA * V * back) / (cnts.sum(1, keepdim=True) + ALPHA * V)
    COV['fitbg_ce_table'] = torch.log(pfit.clamp_min(1e-30))
    COV['back'] = back
    bg_arg = pfit.argmax(-1)
    fitbg = torch.full((V,), int(uni.argmax()), dtype=torch.long, device=DEV)
    fitbg[seen_cpu.nonzero(as_tuple=True)[0].to(DEV)] = bg_arg
    COV['fitbg'] = fitbg
    COV['idmap'] = idmap.to(DEV)
    print(f'  fit-row bigram: {int(keep.sum())} observations over {ncov} covered current types; '
          f'{int((cnts.sum(1) > 0).sum())} types observed', flush=True)
    print(f'ACCURACY BY TARGET FREQUENCY | buckets {BUCKETS} on the fit-row count of the TRUE '
          f'target | full-rank settled program | DISCOVERY ONLY', flush=True)

    # the settled fallback: output-NN neighbour (§1780/§1781)
    lpc = torch.zeros(ncov, W, device=DEV)
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
    pcn = torch.softmax(lpc, -1)
    pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
    del lpc
    nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
    nnrow[tk] = torch.arange(ncov, device=DEV)
    for s0 in range(0, unc.numel(), 512):
        u = unc[s0:s0 + 512]
        p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
        p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
    del pcn
    torch.cuda.empty_cache()

    tables = {st: torch.zeros(ncov, D, device=DEV) for st in sites}
    cap = {}

    def mk(st):
        def hook(mod, args, out):
            cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
            return None
        return hook
    for i in range(0, ncov, 256):
        t = tk[i:i + 256].unsqueeze(1)
        forward_logits(t, [(st, mk(st)) for st in sites])
        for st in sites:
            tables[st][i:i + t.shape[0]] = cap[st]
    Ecov = m.transformer.wte.weight.detach()[tk].float().double()
    A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (ncov / D)
    Eunc = m.transformer.wte.weight.detach()[unc].float().double()
    print(f'  built the settled fallback and 36 tables ({time.time() - t0:.0f}s)', flush=True)

    def program_rows(r):
        if r is None:
            tc = tables
        else:
            tc = {}
            for st, tbl in tables.items():
                b = tbl.double()
                mu = b.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
                tc[st] = (mu + (U[:, :r] * S[:r]) @ Vh[:r]).float()
        out = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tc[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mp = (U[:, :MAP_RANK] * S[:MAP_RANK]) @ Vh[:MAP_RANK]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tc[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        return out

    # ---- arm 4: the EVAL-ROW LEAVE-ONE-OUT bigram -- NOT a fair floor but an upper bound on
    # what any bigram could do here, since it is fitted on the very rows it is scored on. Leave-one-out
    # is exact: take the top-2 counts and, when the top is the target itself, decrement and re-compare.
    loo_state = {}

    def build_loo(rows, ename):
        """LOO bigram SCORES on this role's own rows only.  Ranking by counts + alpha*V*backoff is
        monotone in the smoothed probability (the denominator is constant within a row), so ranks read
        straight off this without normalising.  Row `ncov` holds uncovered current tokens, which fall
        back to the unigram exactly as the bigram-with-backoff does."""
        c = torch.zeros(ncov + 1, V, dtype=torch.float32, device=DEV)
        cu = rows[:, :-1][:, 64:].reshape(-1).long().to(DEV)
        nx = rows[:, 1:][:, 64:].reshape(-1).long().to(DEV)
        r = COV['idmap'][cu]
        r = torch.where(r >= 0, r, torch.full_like(r, ncov))
        c.index_put_((r, nx), torch.ones(r.numel(), device=DEV), accumulate=True)
        loo_state['s'] = c          # RAW counts; every arm adds its own term per batch
        torch.cuda.empty_cache()
        print(f'  eval-row LOO bigram for {ename}: {r.numel()} observations, '
              f'score bank {c.numel() * 4 / 2**30:.2f} GiB', flush=True)

    res = {}
    fr = program_rows(None)
    hooks = [(st, row_hook(fr[st])) for st in sites]
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        build_loo(ev, ename)
        c = audit(ev, hooks, loo_state['s'], ncov)
        del loo_state['s']
        torch.cuda.empty_cache()
        res[ename] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in c.items()}
        print(f'\n  {ename}: n {c["n"]}', flush=True)
        print(f'    §1790 loo_argmax          {c["acc_s1790"]:6.2%}   <- the published arm',
              flush=True)
        print(f'    leak-free argmax          {c["acc_clean"]:6.2%}', flush=True)
        print(f'    program                   {c["acc_prog"]:6.2%}', flush=True)
        print(f'    live                      {c["acc_live"]:6.2%}', flush=True)
        print(f'    §1790 hits held by a TIE: {c["tie_hits"]} of {c["s1790_hits"]} '
              f'({c["tie_share_of_s1790_hits"]:.1%})', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    pa = all(res[e]['acc_s1790'] - res[e]['acc_clean'] >= 0.02 for e in roles)
    pb = all(res[e]['tie_share_of_s1790_hits'] >= 0.20 for e in roles)
    pc = all(res[e]['acc_prog'] > res[e]['acc_clean'] for e in roles)
    fitce = {e: bigram_ce(load(p), COV['fitbg_ce_table']) for e, p, _ in EVAL_SETS
             if e in S1767_FITBIGRAM_CE}
    # LESSON 42: this is the whole run -- the §1790 arm must reproduce §1790's PUBLISHED figure, or I
    # have not re-created the code path I am accusing and no conclusion may be drawn either way.
    pd = (all(abs(res[e]['acc_s1790'] - S1790_LOOBG[e]) <= 0.001
              and abs(res[e]['acc_prog'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['acc_live'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(fitce[e] - v) <= 0.01 for e, v in S1767_FITBIGRAM_CE.items())
          and ncov == NCOV)

    print(f'\n  §1790\'s arm is inflated by >=2pp over a leak-free one -> {pa}  ' + '  '.join(
        f'{e} {res[e]["acc_s1790"]:.2%} vs {res[e]["acc_clean"]:.2%} '
        f'({100*(res[e]["acc_s1790"] - res[e]["acc_clean"]):+.2f}pp)' for e in roles), flush=True)
    print(f'  >=20% of its hits are held by a TIE -> {pb}  ' + '  '.join(
        f'{e} {res[e]["tie_hits"]}/{res[e]["s1790_hits"]} '
        f'({res[e]["tie_share_of_s1790_hits"]:.1%})' for e in roles), flush=True)
    print(f'  leak-free, the PROGRAM beats the bigram -> {pc}  ' + '  '.join(
        f'{e} prog {res[e]["acc_prog"]:.2%} vs bigram {res[e]["acc_clean"]:.2%} '
        f'({100*(res[e]["acc_prog"] - res[e]["acc_clean"]):+.2f}pp)' for e in roles), flush=True)
    print(f'  §1790 arm reproduces its PUBLISHED figure, program/live reproduce §1789, '
          f'fit-bigram CE reproduces §1767 ' + '  '.join(
        f'{e} {v:.5f}' for e, v in fitce.items()) + f', coverage {ncov} -> control {pd}', flush=True)

    r2 = {'config': {'table_ranks': ['full' if r is None else str(r) for r in RANKS],
                     'map_rank': MAP_RANK,
                     'program': 'context-free tables, output-NN fallback with a rank-64 '
                                'embedding->row map -- the settled design of §1780-§1786',
                     'instruments': 'top-1 agreement with the live model, top-1 accuracy against the '
                                    'true next token, and KL(live || program). All three are NEW to '
                                    'this thread, which has been CE-only since §1747.',
                     'ROLE_NOTE': 'DISCOVERY ONLY; a second-class confirmation with a DIFFERENT '
                                  'instrument, not a replication of the same one.'},
          'results': res,
          'fit_bigram_covered_ce': {e: round(v, 5) for e, v in fitce.items()},
          'predictions': {'pred_a_s1790_arm_inflated_by_2pp': bool(pa),
                          'pred_b_ties_hold_a_fifth_of_its_hits': bool(pb),
                          'pred_c_leakfree_program_beats_bigram': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
