# WHERE IN THE RANKING DOES THE CROSSOVER HAPPEN?  -- v2, after §1792 voided v1.
#
# v1 (§1792) failed its own cross-run control: its "LOO bigram" ranked by the §1767 smoothed score
# `counts + alpha*V*back` at alpha=0.01, where alpha*V = 503.04 multiplies a unigram distribution
# summing to 1 while the actual LOO counts are 1-3 (36,864 eval observations over 5,419 covered
# current types).  The backoff swamped the counts, the arm was a UNIGRAM at the top of its ranking,
# and its top-1 came out 10.31 / 10.84 / 10.04% against the 15.97 / 16.63 / 18.00% the same nominal
# object scored in §1790.  Every internal control passed -- monotonicity, partition, coverage, and a
# CE control on a different table.  Only the cross-run figure caught it.  That is LESSON 42.
#
# WHAT CHANGED.  The ranking arm `loobg` now uses RAW LOO counts with the unigram used only to break
# ties (0.5*back < 0.5 can never reorder distinct integer counts).  §1790's exact object -- the argmax
# of the raw LOO counts -- is carried as a separate arm `loobg_argmax_top1` and is what pred_d checks,
# so the control compares like with like.  The alpha=0.01 arm is retained as `loobg_smoothed` so
# §1792's finding stays on the record as a measured number rather than as prose.
#
# ONE TIE-BREAKING CONVENTION FOR EVERY ARM: rank = 1 + #{tokens scored STRICTLY ABOVE the target}.
# Counts tie heavily even after the unigram tie-break, so this is optimistic for the bigram and
# biases AGAINST the program: a program win here is stronger than it looks, a loss weaker.
# Registered before the run so it cannot be adduced afterwards.
#
# ROLES. skip7000, skip11000, skip1200; full-rank settled program; LOO bigram built PER ROLE, exact
# leave-one-out by decrementing the target's own count. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39:
#   pred_a THE CROSSOVER IS EARLY: the program beats the LOO bigram on TOP-5 accuracy overall by at
#          least 2 percentage points, at every role. If FALSE, the program's CE advantage is not
#          visible even a few ranks in, locating the whole of its CE win in the deep tail of the
#          distribution -- mass assigned to tokens no ranking metric ever reaches.
#   pred_b ... AND IT IS VISIBLE IN THE WHOLE RANKING: the program's MRR exceeds the LOO bigram's
#          overall by at least 0.005, at every role. Scored separately because top-5 is another single
#          point; MRR integrates the rank and could go the other way if the program is better at k=5
#          and worse everywhere below.
#   pred_c ... AND ON THE HEAD, NOT ONLY IN THE TAIL: on the 125+ bucket the program beats the LOO
#          bigram on top-5 by at least 2pp. If FALSE, the program's ranking advantage lives entirely
#          on rare targets and the head is bigram territory at every depth, not just at k=1.
#   pred_d CONTROLS, cross-run per LESSON 42: the raw-count argmax arm reproduces §1790's PUBLISHED
#          0.1597 / 0.1663 / 0.1800 within 0.001 -- the check v1 failed; program and live rank-1
#          reproduce §1789's 0.1355 / 0.1425 / 0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001;
#          top-k is monotone in k for every arm and bucket; the fit-row bigram's covered CE reproduces
#          §1767's 7.88804 / 7.90729 within 0.01; buckets partition; coverage is 5419 of 50257.
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
OUT = PT + 'ops/rank_crossover_v2_results.json'
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
def rank_stats(rows, hooks, loo_scores, ncov):
    """Rank of the TRUE target under each arm, and the top-k curve that falls out of it.

    One convention for every arm: rank = 1 + #{tokens scored STRICTLY ABOVE the target}. Counts tie
    heavily, so this is optimistic for the bigram and pessimistic for nothing -- it biases AGAINST the
    program, which is the direction that makes a program win meaningful and a program loss weak."""
    ARMS = ('live', 'prog', 'loobg', 'loobg_smoothed')
    acc = {b: {a: {'topk': {k: 0 for k in KS}, 'rr': 0.0} for a in ARMS}
           | {'n': 0, 'argmax_raw_hits': 0} for b in BUCKETS}
    tot = {a: {'topk': {k: 0 for k in KS}, 'rr': 0.0} for a in ARMS} | {'n': 0,
                                                                        'argmax_raw_hits': 0}
    for i in range(0, rows.shape[0], RB):
        bb = rows[i:i + RB]
        idx = bb[:, :-1].to(DEV).contiguous()
        cur = idx[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        r = COV['idmap'][cur]
        r = torch.where(r >= 0, r, torch.full_like(r, ncov))
        cnt = loo_scores[r]                                   # [B, P, V] RAW counts (a copy)
        own = cnt.gather(-1, tg.unsqueeze(-1))
        cnt.scatter_(-1, tg.unsqueeze(-1), own - 1.0)         # exact leave-one-out
        # §1790's exact object: argmax of the raw LOO counts, for the cross-run control.
        argmax_raw = cnt.argmax(-1)
        rank = {}
        for name, sc in (('live', forward_logits(idx)[:, 64:].float()),
                         ('prog', forward_logits(idx, hooks)[:, 64:].float()),
                         # raw counts, unigram ONLY as a tie-break: back is a distribution summing to
                         # 1, so 0.5*back < 0.5 can never reorder distinct integer counts.
                         ('loobg', cnt + 0.5 * COV['back']),
                         # the §1767 CE-selected smoothing, kept so §1792's finding stays a number.
                         ('loobg_smoothed', cnt + ALPHA * V * COV['back'])):
            t = sc.gather(-1, tg.unsqueeze(-1))
            rank[name] = 1 + (sc > t).sum(-1)
            del sc
        del cnt
        f = COV['freq'][tg]
        raw_hit = argmax_raw == tg
        tot['n'] += int(tg.numel())
        tot['argmax_raw_hits'] += int(raw_hit.sum())
        for b in BUCKETS:
            msk = (f >= b[0]) & (f <= b[1])
            acc[b]['n'] += int(msk.sum())
            acc[b]['argmax_raw_hits'] += int(raw_hit[msk].sum())
            for a in ARMS:
                rr = (1.0 / rank[a].double())
                tot[a]['rr'] += float(rr.sum()) if b == BUCKETS[0] else 0.0
                acc[b][a]['rr'] += float(rr[msk].sum())
                for k in KS:
                    hit = rank[a] <= k
                    if b == BUCKETS[0]:
                        tot[a]['topk'][k] += int(hit.sum())
                    acc[b][a]['topk'][k] += int(hit[msk].sum())
        torch.cuda.empty_cache()
    assert sum(acc[b]['n'] for b in BUCKETS) == tot['n'], 'buckets do not partition the positions'
    out = {}
    for label, d, n in [('overall', tot, tot['n'])] + \
            [(f'{b[0]}-{b[1]}', acc[b], max(acc[b]['n'], 1)) for b in BUCKETS]:
        o = {'n': (tot['n'] if label == 'overall' else acc[
            [b for b in BUCKETS if f'{b[0]}-{b[1]}' == label][0]]['n'])}
        for a in ARMS:
            o[a] = {f'top{k}': d[a]['topk'][k] / n for k in KS} | {'mrr': d[a]['rr'] / n}
        o['loobg_argmax_top1'] = d['argmax_raw_hits'] / n
        out[label] = o
    return out


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
        c = rank_stats(ev, hooks, loo_state['s'], ncov)
        del loo_state['s']
        torch.cuda.empty_cache()
        res[ename] = {kb: {ka: ({kk: round(vv, 5) for kk, vv in va.items()}
                                if isinstance(va, dict) else va)
                           for ka, va in vb.items()} for kb, vb in c.items()}
        print(f'\n  {ename}:', flush=True)
        for label in ('overall', f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}'):
            x = c[label]
            print(f'    {label:12s} n {x["n"]:6d}', flush=True)
            print(f'      §1790 raw-count argmax top1 {x["loobg_argmax_top1"]:.2%}', flush=True)
            for a in ('live', 'prog', 'loobg', 'loobg_smoothed'):
                print(f'      {a:14s} ' + '  '.join(f'top{k} {x[a][f"top{k}"]:6.2%}' for k in KS)
                      + f'  MRR {x[a]["mrr"]:.5f}', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'
    pa = all(res[e]['overall']['prog']['top5'] - res[e]['overall']['loobg']['top5'] >= 0.02
             for e in roles)
    pb = all(res[e]['overall']['prog']['mrr'] - res[e]['overall']['loobg']['mrr'] >= 0.005
             for e in roles)
    pc = all(res[e][top]['prog']['top5'] - res[e][top]['loobg']['top5'] >= 0.02 for e in roles)
    fitce = {e: bigram_ce(load(p), COV['fitbg_ce_table']) for e, p, _ in EVAL_SETS
             if e in S1767_FITBIGRAM_CE}
    mono = all(res[e][b][a][f'top{k2}'] >= res[e][b][a][f'top{k1}'] - 1e-9
               for e in roles for b in res[e]
               for a in ('live', 'prog', 'loobg', 'loobg_smoothed')
               for k1, k2 in zip(KS, KS[1:]))
    # LESSON 42: the control that matters names a figure PUBLISHED BY AN EARLIER RUN of different
    # code -- here §1790's raw-count argmax -- not one re-derived inside this script.
    pd = (all(abs(res[e]['overall']['prog']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['overall']['live']['top1'] - S1789_LIVE[e]) <= 0.001
              and abs(res[e]['overall']['loobg_argmax_top1'] - S1790_LOOBG[e]) <= 0.001
              for e in roles)
          and all(abs(fitce[e] - v) <= 0.01 for e, v in S1767_FITBIGRAM_CE.items())
          and mono and ncov == NCOV)

    print(f'\n  the program beats the LOO bigram on TOP-5 overall by >=2pp -> {pa}  ' + '  '.join(
        f'{e} {res[e]["overall"]["prog"]["top5"]:.2%} vs {res[e]["overall"]["loobg"]["top5"]:.2%} '
        f'({100*(res[e]["overall"]["prog"]["top5"] - res[e]["overall"]["loobg"]["top5"]):+.2f}pp)'
        for e in roles), flush=True)
    print(f'  ... and on MRR overall by >=0.005 -> {pb}  ' + '  '.join(
        f'{e} {res[e]["overall"]["prog"]["mrr"]:.5f} vs {res[e]["overall"]["loobg"]["mrr"]:.5f} '
        f'({res[e]["overall"]["prog"]["mrr"] - res[e]["overall"]["loobg"]["mrr"]:+.5f})'
        for e in roles), flush=True)
    print(f'  ... and on TOP-5 on the head by >=2pp -> {pc}  ' + '  '.join(
        f'{e} {res[e][top]["prog"]["top5"]:.2%} vs {res[e][top]["loobg"]["top5"]:.2%} '
        f'({100*(res[e][top]["prog"]["top5"] - res[e][top]["loobg"]["top5"]):+.2f}pp)'
        for e in roles), flush=True)
    kcross = {}
    for e in roles:
        kcross[e] = next((k for k in KS
                          if res[e]['overall']['prog'][f'top{k}']
                          > res[e]['overall']['loobg'][f'top{k}']), None)
    print(f'  CROSSOVER: smallest k in {KS} where the program overtakes the bigram -> ' + '  '.join(
        f'{e} {kcross[e]}' for e in roles), flush=True)
    print(f'  §1792 smoothing check: alpha={ALPHA} arm top1 ' + '  '.join(
        f'{e} {res[e]["overall"]["loobg_smoothed"]["top1"]:.2%}' for e in roles)
        + ' vs raw-count ' + '  '.join(
            f'{res[e]["overall"]["loobg_argmax_top1"]:.2%}' for e in roles), flush=True)
    print(f'  top-1 reproduces §1789/§1790, top-k monotone, fit-bigram CE reproduces §1767 ' + '  '.join(
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
          'crossover_k': kcross,
          'predictions': {'pred_a_program_beats_bigram_at_top5': bool(pa),
                          'pred_b_program_beats_bigram_on_mrr': bool(pb),
                          'pred_c_program_beats_bigram_at_top5_on_the_head': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
