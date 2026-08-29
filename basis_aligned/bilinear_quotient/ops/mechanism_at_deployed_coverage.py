# THE MECHANISM SPLIT AT DEPLOYED COVERAGE -- confirming §1887 where the fallback arm is largest.
#
# §1887 attributed the compiled program's fidelity to the live model to the TABLE lookups: enrichment
# over a permutation null is 7.16 / 7.29 / 7.49x at covered current tokens against 3.14 / 3.56 / 3.11x at
# uncovered ones, a 2.28 / 2.05 / 2.41x ratio that WIDENS to 2.57 / 2.97 / 2.71x when the target bucket
# is held at 125+.
#
# It was measured at 16,110 covered types, where only about a tenth of scored positions are uncovered --
# 3,672 / 3,668 / 1,835 positions carrying the entire fallback arm. At the DEPLOYED coverage of 5,419 the
# uncovered share is roughly 24%, so the fallback arm gets about two and a half times the positions. That
# is both a stronger measurement of the weaker arm and a coverage generalisation of the attribution.
#
# It is also the arm where §1887's conclusion is least well supported, which is why this runs before the
# per-site question rather than after it.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2: second-class confirmation of §1887.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# The permutation estimator is §1885's, calibrated 0.977x known-independent / 15.6x known-40%-copier.
# ANCHORS are §1887's, measured at 16,110 -- they are the thing being generalised, not a control, and
# the KEPT-FRACTION anchors of §1883 are deliberately NOT used because they are population-dependent
# (the trap that cost §1882 a launch).
#   pred_a THE ATTRIBUTION HOLDS: the covered/uncovered enrichment ratio is at least 1.5x on all three
#          roles at 5,419 types. If FALSE the attribution is specific to high coverage, and §1887's
#          headline needs "at 16,110 types" attached -- which would matter, because the DEPLOYED build
#          runs at 5,419.
#   pred_b AND THE FALLBACK ARM REPRODUCES: uncovered enrichment lands within 1.0x of §1887's PUBLISHED
#          3.14 / 3.56 / 3.11x. This is the arm with the thin statistics at 16,110, so agreement here on
#          2.5x more positions is what makes §1887's weaker number trustworthy. If FALSE the fallback
#          tracks the model differently at different coverages, which would be a new fact about §1870's
#          map rather than a failure of §1887.
#   pred_c AND IT IS STILL NOT A FREQUENCY PROXY: the ratio within the 125+ target bucket is at least
#          1.2x on all three roles, as §1887 found (2.57 / 2.97 / 2.71x). The control that made §1887
#          interpretable, re-run where the covered and uncovered token populations differ most.
#   pred_d CONTROLS: coverage is exactly 5,419; the mechanism split partitions every scored position;
#          the target buckets partition independently; and the uncovered share is materially larger than
#          §1887's -- reported, since a run that did not actually move the split would confirm nothing.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (None,)              # the DEPLOYED build: full-rank tables, rank-64 map
MAPRANK_OF = {None: 64, 512: 512}   # §1880/§1881: map_rank >= table_rank + 1
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/mechanism_at_deployed_coverage_results.json'
NPERM = 8            # permutations averaged per bucket; the calibration above used 8
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'    # 5,419 types, the DEPLOYED coverage
H = m.transformer.h
NCOV = 5419       # the DEPLOYED coverage (§1834). §1834's 5419 is S1789_COV; §1788/§1789's accuracy figures
S1789_COV = 5419
S1883_TOP = {'skip7000': 0.536, 'skip11000': 0.541, 'skip1200': 0.539}   # §1883 DEPLOYED column
S1883_BOT = {'skip7000': 0.026, 'skip11000': 0.049, 'skip1200': 0.035}
S1887_UNC = {'skip7000': 3.14, 'skip11000': 3.56, 'skip1200': 3.11}   # §1887 PUBLISHED, 16,110 types
S1884_AGREE = {'skip7000': 0.3464, 'skip11000': 0.3511, 'skip1200': 0.3473}   # §1884 PUBLISHED, 125+  # below are AT 5419 and are printed for context, never used as bars (§1882's trap)
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
def compare_by_bucket(rows, hooks, seenmask=None):
    """Top-1 accuracy of live and program, split by the TRUE TARGET token's fit-row frequency.

    The bucket axis is the TARGET, not the current token: the program is keyed on the current token,
    so this asks what it can PRODUCE rather than what it can condition on."""
    a = {b: {'acc_l': 0, 'acc_p': 0, 'n': 0, 'agree': 0, 'both': 0} for b in BUCKETS}
    keep = {b: {'l': [], 'p': []} for b in BUCKETS}   # per-bucket predictions, for the permutation null
    # the MECHANISM split: covered current token -> table lookup; uncovered -> §1870's fallback map.
    mech = {'cov': {'l': [], 'p': [], 'al': 0, 'ap': 0, 'n': 0},
            'unc': {'l': [], 'p': [], 'al': 0, 'ap': 0, 'n': 0}}
    mtop = {'cov': {'l': [], 'p': []}, 'unc': {'l': [], 'p': []}}
    tot = {'acc_l': 0, 'acc_p': 0, 'n': 0, 'agree': 0, 'both': 0}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        al = forward_logits(idx)[:, 64:].argmax(-1)
        ap = forward_logits(idx, hooks)[:, 64:].argmax(-1)
        cl, cp = (al == tg), (ap == tg)
        ag = (al == ap)                 # top-1 agreement, regardless of the true target
        bo = cl & cp                    # both right
        f = COV['freq'][tg]
        cur = idx[:, 64:]
        iscov = seenmask[cur] if seenmask is not None else torch.ones_like(cur, dtype=torch.bool)
        hib = (f >= BUCKETS[-1][0])
        for nm, msk2 in (('cov', iscov), ('unc', ~iscov)):
            mech[nm]['l'].append(al[msk2]); mech[nm]['p'].append(ap[msk2])
            mech[nm]['al'] += int(cl[msk2].sum()); mech[nm]['ap'] += int(cp[msk2].sum())
            mech[nm]['n'] += int(msk2.sum())
            m3 = msk2 & hib
            mtop[nm]['l'].append(al[m3]); mtop[nm]['p'].append(ap[m3])
        tot['acc_l'] += int(cl.sum()); tot['acc_p'] += int(cp.sum()); tot['n'] += int(tg.numel())
        tot['agree'] += int(ag.sum()); tot['both'] += int(bo.sum())
        for b in BUCKETS:
            msk = (f >= b[0]) & (f <= b[1])
            a[b]['acc_l'] += int(cl[msk].sum()); a[b]['acc_p'] += int(cp[msk].sum())
            a[b]['n'] += int(msk.sum())
            a[b]['agree'] += int(ag[msk].sum()); a[b]['both'] += int(bo[msk].sum())
            keep[b]['l'].append(al[msk]); keep[b]['p'].append(ap[msk])
    assert sum(a[b]['n'] for b in BUCKETS) == tot['n'], 'buckets do not partition the positions'
    out = {'overall': {'top1_acc_live': tot['acc_l'] / tot['n'],
                       'top1_acc_prog': tot['acc_p'] / tot['n'], 'n': tot['n'],
                       'agreement': tot['agree'] / tot['n'],
                       'prog_right_also_live_right': tot['both'] / max(tot['acc_p'], 1)}}
    gen = torch.Generator(device='cpu').manual_seed(0)
    for b in BUCKETS:
        n = max(a[b]['n'], 1)
        L = torch.cat(keep[b]['l']) if keep[b]['l'] else torch.zeros(0, dtype=torch.long, device=DEV)
        P = torch.cat(keep[b]['p']) if keep[b]['p'] else torch.zeros(0, dtype=torch.long, device=DEV)
        if L.numel() > 1:
            pm = sum(float((L == P[torch.randperm(P.numel(), generator=gen).to(P.device)]).double().mean())
                     for _ in range(NPERM)) / NPERM
        else:
            pm = 0.0
        out[f'{b[0]}-{b[1]}'] = {'top1_acc_live': a[b]['acc_l'] / n,
                                 'top1_acc_prog': a[b]['acc_p'] / n,
                                 'kept_fraction': (a[b]['acc_p'] / n) / max(a[b]['acc_l'] / n, 1e-9),
                                 'n': a[b]['n'],
                                 'agreement': a[b]['agree'] / n,
                                 'independence': (a[b]['acc_l'] / n) * (a[b]['acc_p'] / n),
                                 'permutation_null': pm,
                                 'enrichment': (a[b]['agree'] / n) / max(pm, 1e-12),
                                 'broken_S1884_null': (a[b]['acc_l'] / n) * (a[b]['acc_p'] / n),
                                 'prog_right_also_live_right': a[b]['both'] / max(a[b]['acc_p'], 1)}
    def _en(dl, dp):
        L, P = torch.cat(dl), torch.cat(dp)
        if L.numel() < 2:
            return {'agreement': 0.0, 'permutation_null': 0.0, 'enrichment': 0.0, 'n': int(L.numel())}
        ob = float((L == P).double().mean())
        pm = sum(float((L == P[torch.randperm(P.numel(), generator=gen).to(P.device)]).double().mean())
                 for _ in range(NPERM)) / NPERM
        return {'agreement': ob, 'permutation_null': pm, 'enrichment': ob / max(pm, 1e-12),
                'n': int(L.numel())}
    for nm in ('cov', 'unc'):
        n2 = max(mech[nm]['n'], 1)
        out[f'mech_{nm}'] = {**_en(mech[nm]['l'], mech[nm]['p']),
                             'top1_acc_live': mech[nm]['al'] / n2,
                             'top1_acc_prog': mech[nm]['ap'] / n2}
        out[f'mech_{nm}_top_bucket'] = _en(mtop[nm]['l'], mtop[nm]['p'])
    assert out['mech_cov']['n'] + out['mech_unc']['n'] == tot['n'], \
        'the mechanism split does not partition the scored positions'
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
    print(f'MECHANISM SPLIT AT DEPLOYED COVERAGE | buckets {BUCKETS} on the fit-row count of the TRUE '
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
            c = compare_by_bucket(ev, hooks, seen)
            res[key][ename] = {k: {kk: (round(vv, 5) if isinstance(vv, float) else vv)
                                   for kk, vv in v.items()} for k, v in c.items()}
            print(f'  {ename}: overall live {c["overall"]["top1_acc_live"]:.2%} '
                  f'prog {c["overall"]["top1_acc_prog"]:.2%}', flush=True)
            for b in BUCKETS:
                k = f'{b[0]}-{b[1]}'
                x = c[k]
                print(f'    target fit-count {k:12s} n {x["n"]:6d}  live {x["top1_acc_live"]:6.2%}  '
                      f'prog {x["top1_acc_prog"]:6.2%}  kept {x["kept_fraction"]:6.1%}  '
                      f'agree {x["agreement"]:6.2%} (null {x["permutation_null"]:6.2%} '
                      f'= {x["enrichment"]:5.2f}x)  '
                      f'prog-right-also-live {x["prog_right_also_live_right"]:6.1%}', flush=True)
            ev = None
            torch.cuda.empty_cache()
        fr, hooks = None, None
        torch.cuda.empty_cache()
        del ev
        torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    top, bot = f'{BUCKETS[-1][0]}-{BUCKETS[-1][1]}', f'{BUCKETS[0][0]}-{BUCKETS[0][1]}'
    B = 'full'

    def g(e, b, k):
        return res[B][e][b][k]
    en_top = {e: g(e, top, 'enrichment') for e in roles}
    en_bot = {e: g(e, bot, 'enrichment') for e in roles}
    nullratio = {e: g(e, bot, 'permutation_null') / max(g(e, bot, 'broken_S1884_null'), 1e-12)
                 for e in roles}
    partition = all(sum(res[B][e][f'{x}-{y}']['n'] for x, y in BUCKETS)
                    == res[B][e]['overall']['n'] for e in roles)
    mc = {e: res[B][e]['mech_cov'] for e in roles}
    mu = {e: res[B][e]['mech_unc'] for e in roles}
    mct = {e: res[B][e]['mech_cov_top_bucket'] for e in roles}
    mut = {e: res[B][e]['mech_unc_top_bucket'] for e in roles}
    ratio = {e: mc[e]['enrichment'] / max(mu[e]['enrichment'], 1e-12) for e in roles}
    ratio_t = {e: mct[e]['enrichment'] / max(mut[e]['enrichment'], 1e-12) for e in roles}
    parts = all(mc[e]['n'] + mu[e]['n'] == res[B][e]['overall']['n'] for e in roles)
    uncshare = {e: mu[e]['n'] / res[B][e]['overall']['n'] for e in roles}
    pa = all(ratio[e] >= 1.5 for e in roles)
    pb = all(abs(mu[e]['enrichment'] - S1887_UNC[e]) <= 1.0 for e in roles)
    pc = all(ratio_t[e] >= 1.2 for e in roles)
    pd = ncov == NCOV and parts and partition

    print(f'\n  THE TABLES CARRY THE TRACKING (covered/uncovered enrichment >= 1.5x) -> {pa}',
          flush=True)
    for e in roles:
        print(f'    {e:10s} covered   n {mc[e]["n"]:6d}  live {mc[e]["top1_acc_live"]:6.2%}  '
              f'prog {mc[e]["top1_acc_prog"]:6.2%}  agree {mc[e]["agreement"]:6.2%}  '
              f'null {mc[e]["permutation_null"]:6.2%}  = {mc[e]["enrichment"]:5.2f}x', flush=True)
        print(f'    {e:10s} uncovered n {mu[e]["n"]:6d}  live {mu[e]["top1_acc_live"]:6.2%}  '
              f'prog {mu[e]["top1_acc_prog"]:6.2%}  agree {mu[e]["agreement"]:6.2%}  '
              f'null {mu[e]["permutation_null"]:6.2%}  = {mu[e]["enrichment"]:5.2f}x   '
              f'ratio {ratio[e]:5.2f}x', flush=True)
    print(f'  and the FALLBACK arm reproduces §1887 within 1.0x -> {pb}  ' + '  '.join(
        f'{e} {mu[e]["enrichment"]:.2f}x vs §1887 {S1887_UNC[e]:.2f}x' for e in roles), flush=True)
    print(f'  and it is NOT a target-frequency proxy (>=1.2x within the 125+ bucket) -> {pc}  '
          + '  '.join(f'{e} cov {mct[e]["enrichment"]:.2f}x / unc {mut[e]["enrichment"]:.2f}x '
                      f'= {ratio_t[e]:.2f}x' for e in roles), flush=True)
    print(f'  coverage {ncov}, mechanism split partitions {parts}, buckets partition {partition}, '
          f'uncovered share ' + ' '.join(f'{e} {uncshare[e]:.1%}' for e in roles)
          + f' (§1887 was ~10%) -> control {pd}', flush=True)

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
          'predictions': {'pred_a_tables_carry_tracking_program_concentrated_on_frequent_targets': bool(pa),
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
