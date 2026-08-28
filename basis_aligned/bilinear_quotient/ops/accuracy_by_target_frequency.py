# WHERE DOES THE PROGRAM'S ACCURACY GO? -- top-1 by the TARGET token's frequency.
#
# §1788: the settled standalone program keeps only about a THIRD of the live model's top-1 accuracy
# (13.55 / 14.25 / 13.64% against 39.32 / 42.35 / 38.88%), while recovering 32.4% of the CE stake. The
# ledger has been quoting the CE figure alone since §1747 and both now sit together. What it does not
# say is whether the lost accuracy is spread evenly or concentrated.
#
# The obvious split is by the TRUE TARGET token's frequency in the fit rows. Note the axis: the
# program is keyed on the CURRENT token, so bucketing by the TARGET asks what it can PRODUCE rather
# than what it can condition on. Buckets are fit-row counts 0, 1-4, 5-24, 25-124, 125+, and the run
# asserts they partition the scored positions.
#
# ROLES. skip7000, skip11000, skip1200; full-rank settled program. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, with margins per LESSON 40, read back per
# LESSON 39:
#   pred_a THE PROGRAM IS CONCENTRATED ON FREQUENT TARGETS: its accuracy on the 125+ bucket is at
#          least 3x its accuracy on the count-0 bucket, at every role. If FALSE the loss is spread
#          evenly across target frequencies, which would mean a per-token program degrades the model
#          uniformly rather than collapsing onto the head of the distribution.
#   pred_b IT IS MORE CONCENTRATED THAN THE LIVE MODEL: the program's top/bottom accuracy ratio
#          exceeds the live model's at every role. Scored independently of pred_a, since both could
#          be concentrated to the same degree -- which would say the concentration is a property of
#          the TASK rather than of the program.
#   pred_c ON FREQUENT TARGETS IT NEARLY KEEPS UP: on the 125+ bucket the program retains at least
#          60% of the live model's accuracy, against ~34% overall. If FALSE the deficit is not a
#          head/tail story at all and the program is worse everywhere.
#   pred_d CONTROLS: the overall top-1 accuracies reproduce §1788's 0.1355 / 0.1425 / 0.1364 and
#          0.3932 / 0.4235 / 0.3888 within 0.001, the buckets partition every scored position, and
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
OUT = PT + 'ops/accuracy_by_target_frequency_results.json'
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
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

    res = {}
    fr = program_rows(None)
    hooks = [(st, row_hook(fr[st])) for st in sites]
    for ename, epath, ce_ref in EVAL_SETS:
        ev = load(epath)
        c = compare_by_bucket(ev, hooks)
        res[ename] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in c.items()}
        print(f'\n  {ename}: overall live {c["overall"]["top1_acc_live"]:.2%} '
              f'prog {c["overall"]["top1_acc_prog"]:.2%}', flush=True)
        for b in BUCKETS:
            k = f'{b[0]}-{b[1]}'
            x = c[k]
            print(f'    target fit-count {k:12s} n {x["n"]:6d}  live {x["top1_acc_live"]:6.2%}  '
                  f'prog {x["top1_acc_prog"]:6.2%}  kept {x["kept_fraction"]:6.1%}', flush=True)
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'

    def ratio(e, who):
        b = res[e][bot][f'top1_acc_{who}']
        return res[e][top][f'top1_acc_{who}'] / b if b > 1e-9 else float('inf')
    pa = all(ratio(e, 'prog') >= 3.0 for e in roles)
    pb = all(ratio(e, 'prog') > ratio(e, 'live') for e in roles)
    pc = all(res[e][top]['kept_fraction'] >= 0.60 for e in roles)
    pd = (all(abs(res[e]['overall']['top1_acc_prog'] - v['prog']) <= 0.001
              and abs(res[e]['overall']['top1_acc_live'] - v['live']) <= 0.001
              for e, v in S1788_ACC.items()) and ncov == NCOV)

    print(f'\n  the program is >=3x more accurate on the top bucket than the bottom -> {pa}  '
          + '  '.join(f'{e} {ratio(e, "prog"):.2f}x' for e in roles), flush=True)
    print(f'  it is MORE concentrated than the live model -> {pb}  ' + '  '.join(
        f'{e} prog {ratio(e, "prog"):.2f}x live {ratio(e, "live"):.2f}x' for e in roles), flush=True)
    print(f'  on the top bucket it keeps >=60% of live accuracy -> {pc}  ' + '  '.join(
        f'{e} {res[e][top]["kept_fraction"]:.1%}' for e in roles), flush=True)
    print(f'  overall accuracies reproduce §1788 + coverage {ncov} -> control {pd}', flush=True)

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
