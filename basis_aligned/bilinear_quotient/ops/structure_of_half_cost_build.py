# DOES THE HALF-COST BUILD KEEP THE DEPLOYED BUILD'S ACCURACY STRUCTURE?
#
# §1882 found that at 16,110 covered types, table rank 512 with a rank-512 map costs 360.724M against
# the deployed DESIGN's 673.464M -- a 46% cut -- for +0.00502 / +0.00408 all-position nats on two roles
# and -0.00352 (BETTER) on the third. That is the largest cost result in this arc.
#
# It is also a MEAN. §1789 showed the deployed build's accuracy is sharply non-uniform in the true
# target's fit-row count: it keeps ~63% of the live model's top-1 on targets seen 125+ times and 2.7 /
# 6.2 / 3.6% on targets never seen, while the live model itself is nearly flat (2.4x across the range).
# A CE that moves by five thousandths can hide a large REDISTRIBUTION across those buckets, and
# truncating a table from full rank to 512 is exactly the operation that should hurt the rare end first:
# rare tokens contribute least to the covered block's variance, so they are what an SVD truncation drops.
#
# This applies §1789's instrument to both builds at 16,110 coverage. It is a comparison, not a
# certification -- §1789 was itself a discovery run.
#
# ROLES. skip7000, skip11000, skip1200; top-1 by target-frequency bucket. DISCOVERY ONLY.
# Rung 3: §1882's open question.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# ANCHORS: §1789's figures are at 5,419 types and are NOT used as bars -- at 16,110 the buckets hold
# different populations, the trap that cost §1882 a launch. They are printed for context only.
#   pred_a THE STRUCTURE SURVIVES AT THE COMMON END: the half-cost build's kept-fraction in the 125+
#          bucket is within 2 percentage points of the deployed design's, on all three roles. This is
#          where 82% of the program's correct predictions come from (§1789), so a loss here would mean
#          the 46% saving is paid for in the only bucket that carries the program.
#   pred_b AND AT THE RARE END: its kept-fraction in the fit-count-0 bucket is within 1 percentage point
#          of the deployed design's, on all three roles. Tighter in absolute terms because the bucket is
#          small (2.7-6.2% kept at 5,419), and this is the bucket an SVD truncation should damage first.
#          If FALSE the mean CE was hiding a redistribution and §1882's headline needs the caveat
#          attached wherever it is quoted -- which is exactly why this runs before anything is certified.
#   pred_c AND THE OVERALL TOP-1 GAP IS SMALL: the two builds' overall top-1 differ by less than 1
#          percentage point on all three roles, consistent with a 0.005-nat CE gap. If FALSE, top-1 and
#          CE disagree about these builds and the cost result needs restating in whichever unit the
#          program is actually judged on.
#   pred_d CONTROLS: coverage is exactly 16,110; the five buckets partition every scored position in
#          both arms; and the LIVE model's per-bucket accuracy is identical between arms, since no arm
#          touches it -- a fourteenth known-answer check.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None, 512)          # deployed DESIGN vs §1882's half-cost build
MAPRANK_OF = {None: 64, 512: 512}   # §1880/§1881: map_rank >= table_rank + 1
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/structure_of_half_cost_build_results.json'
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'   # 16,110 types, §1882's coverage
H = m.transformer.h
NCOV = 16110      # §1882's coverage. §1834's 5419 is S1789_COV; §1788/§1789's accuracy figures
S1789_COV = 5419  # below are AT 5419 and are printed for context, never used as bars (§1882's trap)
S1788_ACC = {'skip7000': {'prog': 0.1355, 'live': 0.3932},
             'skip11000': {'prog': 0.1425, 'live': 0.4235},
             'skip1200': {'prog': 0.1364, 'live': 0.3888}}
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
def compare_by_bucket(rows, hooks):
    """Top-1 accuracy of live and program, split by the TRUE TARGET token's fit-row frequency.

    The bucket axis is the TARGET, not the current token: the program is keyed on the current token,
    so this asks what it can PRODUCE rather than what it can condition on."""
    a = {b: {'acc_l': 0, 'acc_p': 0, 'n': 0} for b in BUCKETS}
    tot = {'acc_l': 0, 'acc_p': 0, 'n': 0}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        al = forward_logits(idx)[:, 64:].argmax(-1)
        ap = forward_logits(idx, hooks)[:, 64:].argmax(-1)
        cl, cp = (al == tg), (ap == tg)
        f = COV['freq'][tg]
        tot['acc_l'] += int(cl.sum()); tot['acc_p'] += int(cp.sum()); tot['n'] += int(tg.numel())
        for b in BUCKETS:
            msk = (f >= b[0]) & (f <= b[1])
            a[b]['acc_l'] += int(cl[msk].sum()); a[b]['acc_p'] += int(cp[msk].sum())
            a[b]['n'] += int(msk.sum())
    assert sum(a[b]['n'] for b in BUCKETS) == tot['n'], 'buckets do not partition the positions'
    out = {'overall': {'top1_acc_live': tot['acc_l'] / tot['n'],
                       'top1_acc_prog': tot['acc_p'] / tot['n'], 'n': tot['n']}}
    for b in BUCKETS:
        n = max(a[b]['n'], 1)
        out[f'{b[0]}-{b[1]}'] = {'top1_acc_live': a[b]['acc_l'] / n,
                                 'top1_acc_prog': a[b]['acc_p'] / n,
                                 'kept_fraction': (a[b]['acc_p'] / n) / max(a[b]['acc_l'] / n, 1e-9),
                                 'n': a[b]['n']}
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
    print(f'STRUCTURE OF THE HALF-COST BUILD | buckets {BUCKETS} on the fit-row count of the TRUE '
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
            mr = MAPRANK_OF[r]
            mp = (U[:, :mr] * S[:mr]) @ Vh[:mr]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tc[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        return out

    res = {}
    for r in RANKS:
        key = 'full' if r is None else str(r)
        fr = program_rows(r)
        hooks = [(st, row_hook(fr[st])) for st in sites]
        res[key] = {}
        print(f'\n  === table rank {key}, map rank {MAPRANK_OF[r]} ===', flush=True)
        for ename, epath, ce_ref in EVAL_SETS:
            ev = load(epath)
            c = compare_by_bucket(ev, hooks)
            res[key][ename] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                                   for kk, vv in v.items()} for k, v in c.items()}
            print(f'  {ename}: overall live {c["overall"]["top1_acc_live"]:.2%} '
                  f'prog {c["overall"]["top1_acc_prog"]:.2%}', flush=True)
            for b in BUCKETS:
                k = f'{b[0]}-{b[1]}'
                x = c[k]
                print(f'    target fit-count {k:12s} n {x["n"]:6d}  live {x["top1_acc_live"]:6.2%}  '
                      f'prog {x["top1_acc_prog"]:6.2%}  kept {x["kept_fraction"]:6.1%}', flush=True)
            ev = None
            torch.cuda.empty_cache()
        fr, hooks = None, None
        torch.cuda.empty_cache()
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'
    A, B = 'full', '512'          # the deployed DESIGN and §1882's half-cost build

    def kept(arm, e, b):
        return res[arm][e][b]['kept_fraction']
    dtop = {e: abs(kept(B, e, top) - kept(A, e, top)) for e in roles}
    dbot = {e: abs(kept(B, e, bot) - kept(A, e, bot)) for e in roles}
    dovr = {e: abs(res[B][e]['overall']['top1_acc_prog']
                   - res[A][e]['overall']['top1_acc_prog']) for e in roles}
    livespread = max(abs(res[B][e][b]['top1_acc_live'] - res[A][e][b]['top1_acc_live'])
                     for e in roles for b in [f'{x}-{y}' for x, y in BUCKETS] + ['overall'])
    partition = all(sum(res[arm][e][f'{x}-{y}']['n'] for x, y in BUCKETS)
                    == res[arm][e]['overall']['n'] for arm in (A, B) for e in roles)
    pa = all(dtop[e] <= 0.02 for e in roles)
    pb = all(dbot[e] <= 0.01 for e in roles)
    pc = all(dovr[e] <= 0.01 for e in roles)
    pd = ncov == NCOV and partition and livespread <= 1e-9

    print(f'\n  STRUCTURE AT THE COMMON END (125+ bucket, kept-fraction) -> {pa}  ' + '  '.join(
        f'{e} deployed {kept(A, e, top):.1%} vs half-cost {kept(B, e, top):.1%} '
        f'(d {dtop[e] * 100:.2f}pp)' for e in roles), flush=True)
    print(f'  AT THE RARE END (fit-count 0 bucket) -> {pb}  ' + '  '.join(
        f'{e} deployed {kept(A, e, bot):.1%} vs half-cost {kept(B, e, bot):.1%} '
        f'(d {dbot[e] * 100:.2f}pp)' for e in roles), flush=True)
    print(f'  OVERALL top-1 gap under 1pp -> {pc}  ' + '  '.join(
        f'{e} deployed {res[A][e]["overall"]["top1_acc_prog"]:.2%} vs half-cost '
        f'{res[B][e]["overall"]["top1_acc_prog"]:.2%} (d {dovr[e] * 100:.2f}pp)'
        for e in roles), flush=True)
    print(f'  coverage {ncov}, buckets partition {partition}, LIVE identical between arms '
          f'(spread {livespread:.2e}) -> control {pd}', flush=True)

    r2 = {'config': {'table_ranks': ['full' if r is None else str(r) for r in RANKS],
                     'map_rank_of_table_rank': {str(k): v for k, v in MAPRANK_OF.items()},
                     'program': 'context-free tables, output-NN fallback with a rank-64 '
                                'embedding->row map -- the settled design of §1780-§1786',
                     'instruments': 'top-1 agreement with the live model, top-1 accuracy against the '
                                    'true next token, and KL(live || program). All three are NEW to '
                                    'this thread, which has been CE-only since §1747.',
                     'ROLE_NOTE': 'DISCOVERY ONLY; a second-class confirmation with a DIFFERENT '
                                  'instrument, not a replication of the same one.'},
          'results': res,
          'predictions': {'pred_a_program_concentrated_on_frequent_targets': bool(pa),
                          'pred_b_more_concentrated_than_live': bool(pb),
                          'pred_c_keeps_60pc_on_the_top_bucket': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
