# IS THE PROGRAM'S REMAINING STRENGTH BIGRAM-REACHABLE?  -- the question §1789 ended on.
#
# §1789 located the program's accuracy: it keeps 63.5 / 62.9 / 63.4% of the live model on targets seen
# 125+ times in the fit rows, and 2.7 / 6.2 / 3.6% on targets never seen. So whatever the program
# still does, it does on the head of the distribution -- which is exactly where a bigram lives too.
# §1767 already showed the program beats a bigram in CROSS-ENTROPY, but §1788 was the whole lesson
# that CE and accuracy are different instruments, so that does not settle it.
#
# Three reference arms against the same buckets:
#   fit-row bigram  -- add-alpha 0.01 with unigram backoff on the SAME 96x256 fit rows the program was
#                      fitted on, uncovered current tokens falling back to the unigram argmax. This is
#                      the FAIR floor: identical data, no layers.
#   eval-row LOO bigram -- fitted on the very rows it is scored on, leave-one-out exact via top-2
#                      counts. NOT fair; it is an upper bound on what any bigram could reach here.
#   live model      -- as in §1789.
#
# ROLES. skip7000, skip11000, skip1200; full-rank settled program. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, with explicit margins per LESSON 40, read back
# against their own sentences per LESSON 39:
#   pred_a ON FREQUENT TARGETS THE PROGRAM EARNS ITS LAYERS: on the 125+ bucket it beats the fit-row
#          bigram by at least 3 percentage points at every role. If FALSE, the one regime where the
#          program still works is reachable by counting pairs on its own fit data, and 36 sites of
#          position-wise arithmetic buy nothing there.
#   pred_b ... AND BEATS EVEN AN EVAL-FITTED BIGRAM: on the 125+ bucket it beats the leave-one-out
#          eval-row bigram by at least 1 percentage point at every role. Scored separately from
#          pred_a because a fit-row bigram is starved (24,576 tokens over 5,419 types) and could lose
#          on data volume alone; losing to the LOO arm instead would mean the program is genuinely
#          within the bigram class and merely better estimated.
#   pred_c ON UNSEEN TARGETS, STATISTICS BEAT THE PROGRAM: the LOO bigram, which has support there,
#          exceeds the program's accuracy on the fit-count-0 bucket at every role. If FALSE, then even
#          a bigram with access to the eval statistics cannot produce those tokens, and the tail the
#          program loses needs CONTEXT rather than better per-token estimation -- the stronger claim.
#          (The fit-row bigram is structurally near-zero on this bucket by construction, so it is
#          reported but NOT used as evidence here.)
#   pred_d CONTROLS: overall program and live accuracies reproduce §1789's 0.1355 / 0.1425 / 0.1364
#          and 0.3932 / 0.4235 / 0.3888 within 0.001; the fit-row bigram's covered CE reproduces
#          §1767's 7.88804 / 7.90729 within 0.01, confirming it is the same object; the buckets
#          partition every scored position; coverage is exactly 5419 of 50257.
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
OUT = PT + 'ops/bigram_reachable_accuracy_results.json'
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
def compare_by_bucket(rows, hooks, loo_argmax):
    """Top-1 accuracy of live and program, split by the TRUE TARGET token's fit-row frequency.

    The bucket axis is the TARGET, not the current token: the program is keyed on the current token,
    so this asks what it can PRODUCE rather than what it can condition on."""
    ARMS = ('live', 'prog', 'fitbg', 'loobg')
    a = {b: {k: 0 for k in ARMS} | {'n': 0} for b in BUCKETS}
    tot = {k: 0 for k in ARMS} | {'n': 0}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        hit = {'live': forward_logits(idx)[:, 64:].argmax(-1) == tg,
               'prog': forward_logits(idx, hooks)[:, 64:].argmax(-1) == tg,
               'fitbg': COV['fitbg'][idx] == tg,
               'loobg': loo_argmax(idx, tg) == tg}
        f = COV['freq'][tg]
        tot['n'] += int(tg.numel())
        for k in ARMS:
            tot[k] += int(hit[k].sum())
        for b in BUCKETS:
            msk = (f >= b[0]) & (f <= b[1])
            a[b]['n'] += int(msk.sum())
            for k in ARMS:
                a[b][k] += int(hit[k][msk].sum())
    assert sum(a[b]['n'] for b in BUCKETS) == tot['n'], 'buckets do not partition the positions'
    out = {'overall': {f'acc_{k}': tot[k] / tot['n'] for k in ARMS} | {'n': tot['n']}}
    for b in BUCKETS:
        n = max(a[b]['n'], 1)
        out[f'{b[0]}-{b[1]}'] = {f'acc_{k}': a[b][k] / n for k in ARMS} | {'n': a[b]['n']}
    return out


@torch.no_grad()
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
    ev_all = torch.cat([load(p) for _, p, _ in EVAL_SETS], 0)
    loo_cnt = torch.zeros(ncov + 1, V, dtype=torch.float32, device=DEV)
    ec_cur = ev_all[:, :-1][:, 64:].reshape(-1).long()
    ec_nxt = ev_all[:, 1:][:, 64:].reshape(-1).long()
    ec_ri = COV['idmap'][ec_cur.to(DEV)]
    ec_ri = torch.where(ec_ri >= 0, ec_ri, torch.full_like(ec_ri, ncov))
    loo_cnt.index_put_((ec_ri, ec_nxt.to(DEV)),
                       torch.ones(ec_ri.numel(), device=DEV), accumulate=True)
    loo_top = loo_cnt.topk(2, dim=-1)
    del ev_all
    torch.cuda.empty_cache()
    print(f'  eval-row LOO bigram: {ec_ri.numel()} observations', flush=True)

    def loo_argmax(idx, tg):
        """argmax of the eval-row bigram with THIS position's own observation removed."""
        r = COV['idmap'][idx]
        r = torch.where(r >= 0, r, torch.full_like(r, ncov))
        v, k = loo_top.values[r], loo_top.indices[r]
        c1 = v[..., 0] - (k[..., 0] == tg).float()
        return torch.where(c1 >= v[..., 1], k[..., 0], k[..., 1])

    res = {}
    fr = program_rows(None)
    hooks = [(st, row_hook(fr[st])) for st in sites]
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        c = compare_by_bucket(ev, hooks, loo_argmax)
        res[ename] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in c.items()}
        o = c['overall']
        print(f'\n  {ename}: overall live {o["acc_live"]:.2%} prog {o["acc_prog"]:.2%} '
              f'fit-bigram {o["acc_fitbg"]:.2%} LOO-bigram {o["acc_loobg"]:.2%}', flush=True)
        for b in BUCKETS:
            k = f'{b[0]}-{b[1]}'
            x = c[k]
            print(f'    target fit-count {k:12s} n {x["n"]:6d}  live {x["acc_live"]:6.2%}  '
                  f'prog {x["acc_prog"]:6.2%}  fitbg {x["acc_fitbg"]:6.2%}  '
                  f'loobg {x["acc_loobg"]:6.2%}', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'
    pa = all(res[e][top]['acc_prog'] - res[e][top]['acc_fitbg'] >= 0.03 for e in roles)
    pb = all(res[e][top]['acc_prog'] - res[e][top]['acc_loobg'] >= 0.01 for e in roles)
    pc = all(res[e][bot]['acc_loobg'] > res[e][bot]['acc_prog'] for e in roles)
    fitce = {e: bigram_ce(load(p), COV['fitbg_ce_table']) for e, p, _ in EVAL_SETS
             if e in S1767_FITBIGRAM_CE}
    pd = (all(abs(res[e]['overall']['acc_prog'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['overall']['acc_live'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(fitce[e] - v) <= 0.01 for e, v in S1767_FITBIGRAM_CE.items())
          and ncov == NCOV)

    print(f'\n  on frequent targets the program beats the FIT bigram by >=3pp -> {pa}  ' + '  '.join(
        f'{e} {res[e][top]["acc_prog"]:.2%} vs {res[e][top]["acc_fitbg"]:.2%} '
        f'(+{100*(res[e][top]["acc_prog"] - res[e][top]["acc_fitbg"]):.2f}pp)' for e in roles),
        flush=True)
    print(f'  ... and beats the EVAL-ROW LOO bigram by >=1pp -> {pb}  ' + '  '.join(
        f'{e} {res[e][top]["acc_prog"]:.2%} vs {res[e][top]["acc_loobg"]:.2%} '
        f'(+{100*(res[e][top]["acc_prog"] - res[e][top]["acc_loobg"]):.2f}pp)' for e in roles),
        flush=True)
    print(f'  on UNSEEN targets the LOO bigram beats the program -> {pc}  ' + '  '.join(
        f'{e} {res[e][bot]["acc_loobg"]:.2%} vs {res[e][bot]["acc_prog"]:.2%}' for e in roles),
        flush=True)
    print(f'  accuracies reproduce §1789, fit-bigram CE reproduces §1767 ' + '  '.join(
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
          'predictions': {'pred_a_beats_fit_bigram_on_frequent_targets': bool(pa),
                          'pred_b_beats_loo_bigram_on_frequent_targets': bool(pb),
                          'pred_c_loo_bigram_beats_program_on_unseen': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
