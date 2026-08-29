# IS THE REDISTRIBUTION A RANK EFFECT? -- §1932's open question, measured rather than guessed.
#
# §1932 found that at the DEPLOYED 5,419 coverage the combined build (attn256/mlp768 + rank-512 map) is a
# redistribution rather than a strict win: 29% cheaper, better overall top-1 and better on UNSEEN targets
# on all three roles, but WORSE on the 125+ bucket on all three by 1.20 / 0.93 / 0.65pp. At 16,110 both
# of those signs were the other way round.
#
# §1932 offered an account and explicitly declined to claim it: the tables are relatively richer at the
# smaller covered set, so the deployed design's full-rank advantage on frequent targets should be larger
# there. If that is right, the redistribution is a TABLE-RANK effect and should appear in a plain uniform
# rank sweep with the map held fixed -- no allocation needed. If it does not appear, the effect belongs to
# the allocation or the map and the account is wrong.
#
# The sweep holds the map at rank 512 for the four table-rank arms so the map cannot confound, and carries
# the deployed design (full rank + rank-64 map) as the published anchor. §1789's instrument throughout.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY, 5,419 coverage. Rung 3: §1932's named question.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
#   pred_a THE COMMON BUCKET FALLS WITH TABLE RANK: with the map fixed at 512, the 125+ kept-fraction at
#          table rank 256 is below that at full rank, on at least 2 of 3 roles. If FALSE the common-target
#          loss §1932 measured is not a rank effect and its account is wrong -- which I would say in
#          §1932, where it is already labelled unmeasured.
#   pred_b AND THE UNSEEN BUCKET RISES: the fit-count-0 kept-fraction at table rank 256 is ABOVE that at
#          full rank, on at least 2 of 3 roles. This is the other half of a redistribution: if the common
#          bucket falls and the rare one does not rise, table rank is simply losing accuracy rather than
#          moving it, and §1932's framing as a trade is wrong.
#   pred_c AND THE MAP IS NOT THE CAUSE: going from the rank-64 to the rank-512 map at FULL table rank
#          moves both buckets by less than 1.5 percentage points on all three roles. This isolates the
#          effect to the table axis. If FALSE the map rank moves the accuracy structure too and §1932's
#          comparison confounded two levers -- worth knowing, since §1870/§1877/§1880 priced map rank
#          purely in CE and never looked at buckets.
#   pred_d CONTROLS: coverage is exactly 5,419; the deployed arm (full rank + rank-64 map) reproduces
#          §1932's PUBLISHED 125+ figures 63.5 / 62.9 / 63.4% and unseen figures 2.7 / 6.2 / 3.6% within
#          0.5pp; the buckets partition; and the LIVE per-bucket accuracy is identical across all arms.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = ('deployed', 'full512', 'r768', 'r512', 'r256')
MAPRANK_OF = {'deployed': 64, 'full512': 512, 'r768': 512, 'r512': 512, 'r256': 512}
# UNIFORM ranks only -- the question is about the table axis, not the allocation.
ALLOC = {'deployed': None, 'full512': None,
         'r768': {'attn': 768, 'mlp': 768}, 'r512': {'attn': 512, 'mlp': 512},
         'r256': {'attn': 256, 'mlp': 256}}
S1932_TOP = {'skip7000': 0.635, 'skip11000': 0.629, 'skip1200': 0.634}   # §1932 deployed, 125+
S1932_BOT = {'skip7000': 0.027, 'skip11000': 0.062, 'skip1200': 0.036}   # §1932 deployed, unseen

RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/redistribution_is_rank_results.json'
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'   # 5,419 types at T=256 -- the DEPLOYED coverage
H = m.transformer.h
NCOV = 5419       # §1834's deployed coverage; the 3.29205 live anchors below are ITS population
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
    print(f'IS THE REDISTRIBUTION A RANK EFFECT | buckets {BUCKETS} on the fit-row count of the TRUE '
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
        a = ALLOC[r]
        if a is None:
            tc = tables
        else:
            tc = {}
            for st, tbl in tables.items():
                rk = a[st[0]]
                b = tbl.double()
                mu = b.mean(0, keepdim=True)
                U, S, Vh = torch.linalg.svd(b - mu, full_matrices=False)
                tc[st] = (mu + (U[:, :rk] * S[:rk]) @ Vh[:rk]).float()
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
        key = str(r)
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
    def kept(arm, e, b):
        return res[arm][e][b]['kept_fraction']
    nroles = len(roles)
    fall = sum(1 for e in roles if kept('r256', e, top) < kept('full512', e, top))
    rise = sum(1 for e in roles if kept('r256', e, bot) > kept('full512', e, bot))
    mapmv = max(max(abs(kept('full512', e, b) - kept('deployed', e, b)) for b in (top, bot))
                for e in roles)
    livespread = max(abs(res[a2][e][b]['top1_acc_live'] - res['deployed'][e][b]['top1_acc_live'])
                     for a2 in RANKS for e in roles
                     for b in [f'{x}-{y}' for x, y in BUCKETS] + ['overall'])
    partition = all(sum(res[a2][e][f'{x}-{y}']['n'] for x, y in BUCKETS)
                    == res[a2][e]['overall']['n'] for a2 in RANKS for e in roles)
    pa = fall >= 2
    pb = rise >= 2
    pc = mapmv < 0.015
    pd = (ncov == NCOV and partition and livespread <= 1e-9
          and all(abs(kept('deployed', e, top) - S1932_TOP[e]) <= 0.005 for e in roles)
          and all(abs(kept('deployed', e, bot) - S1932_BOT[e]) <= 0.005 for e in roles))

    print(f'\n  kept-fraction by table rank (map 512 except the deployed arm), 5,419 types:', flush=True)
    for e in roles:
        print(f'    {e:10s} 125+   ' + '  '.join(
            f'{a2} {kept(a2, e, top):.1%}' for a2 in RANKS), flush=True)
        print(f'    {"":10s} unseen ' + '  '.join(
            f'{a2} {kept(a2, e, bot):.1%}' for a2 in RANKS), flush=True)
    print(f'\n  the COMMON bucket falls with table rank (>=2 roles) -> {pa}  {fall}/3', flush=True)
    print(f'  and the UNSEEN bucket rises (>=2 roles) -> {pb}  {rise}/3', flush=True)
    print(f'  and the MAP is not the cause (map64->512 at full rank moves < 1.5pp) -> {pc}  '
          f'max {mapmv * 100:.2f}pp', flush=True)
    print(f'  coverage {ncov}, deployed arm reproduces §1932, partitions, LIVE identical '
          f'-> control {pd}', flush=True)

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
          'predictions': {'pred_a_common_falls_with_rank_program_concentrated_on_frequent_targets': bool(pa),
                          'pred_b_unseen_rises_more_concentrated_than_live': bool(pb),
                          'pred_c_map_not_the_cause_keeps_60pc_on_the_top_bucket': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
