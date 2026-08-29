# AGREEMENT AGAINST A PERMUTATION NULL -- redoing §1884 with an instrument that works.
#
# §1884 asked whether §1882's half-cost build REPRODUCES the live model's computation on the bucket it
# keeps, or merely shares a frequency default with it. Its one clean figure stands: 81.4 / 82.5 / 80.7%
# of the program's correct top-1 predictions on the 125+ bucket are positions the model also gets right,
# so one in five are positions the MODEL gets wrong.
#
# Its two companion predicates do not stand, and the fault was mine. I used `acc_live * acc_prog` as the
# chance baseline for top-1 AGREEMENT. That is the probability both are RIGHT, not the probability they
# AGREE; it implicitly assumes two disagreeing predictors scatter wrong answers uniformly over 50,257
# tokens, when both in fact concentrate on frequent ones. The null was orders of magnitude too low, worst
# where accuracy is lowest, which manufactured a 97x "enrichment" on the rare bucket and a spurious
# inversion. Both were struck.
#
# The correct null is a PERMUTATION control: agreement between the live model's predictions and the
# program's predictions SHUFFLED across positions within the same bucket. That preserves each predictor's
# marginal distribution over tokens exactly and destroys only the position pairing, which is the thing
# under test.
#
# CALIBRATED BEFORE REGISTERING, per LESSON 69 -- the discipline whose absence caused §1884 and §1879.
# On 16,000 synthetic draws from Zipf-ish marginals: two INDEPENDENT predictors give observed 0.02738,
# permutation null 0.02802, enrichment 0.977x (want ~1.0); a predictor that copies the other 40% of the
# time gives 0.41969 / 0.02693 = 15.6x (want >>1). The estimator returns the known answer in both
# directions before it is trusted on real data.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3: §1884's open question.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
#   pred_a THE PAIRING IS REAL: on the 125+ bucket, observed agreement is at least 1.5x the permutation
#          null, on all three roles. The calibration above puts a genuinely independent pair at 0.98x, so
#          1.5x is comfortably outside it. If FALSE the program's top-1 on frequent targets is explained
#          by shared token marginals alone, and "keeps 54% of the model" means the two agree no more than
#          any two predictors with these output distributions would -- the strongest available deflation
#          of every accuracy claim in this thread.
#   pred_b AND THE RARE END IMITATES LESS: the fit-count-0 bucket's enrichment is BELOW the 125+ bucket's
#          on all three roles. This is §1884's pred_c re-asked with a valid null. §1884's answer (a 45x
#          inversion) was an artifact and carries no information, so this is genuinely open in both
#          directions.
#   pred_c AND THE STRIKE WAS CORRECT, NOT CONVENIENT: on the fit-count-0 bucket the PERMUTATION null
#          exceeds §1884's broken acc_live*acc_prog baseline by at least 10x. A KNOWN-ANSWER check on my
#          own correction -- if FALSE, the old baseline was not badly wrong there and I struck two
#          predicates I should have kept, which I would then say.
#   pred_d CONTROLS: raw agreement reproduces §1884's PUBLISHED 34.6 / 35.1 / 34.7% on the 125+ bucket
#          within 0.2pp (same build, same positions, extra instrument only); kept-fractions reproduce
#          §1883 within 0.2pp; coverage is exactly 16,110; the buckets partition every scored position.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; V = 50257; W = 50304
RANKS = (512,)               # §1882's half-cost build only; §1883 settled the comparison
MAPRANK_OF = {None: 64, 512: 512}   # §1880/§1881: map_rank >= table_rank + 1
RIDGE = 1e-2
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/permutation_null_agreement_results.json'
NPERM = 8            # permutations averaged per bucket; the calibration above used 8
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'   # 16,110 types, §1882's coverage
H = m.transformer.h
NCOV = 16110      # §1882's coverage. §1834's 5419 is S1789_COV; §1788/§1789's accuracy figures
S1789_COV = 5419
S1883_TOP = {'skip7000': 0.527, 'skip11000': 0.537, 'skip1200': 0.531}   # §1883 PUBLISHED, 16,110
S1883_BOT = {'skip7000': 0.024, 'skip11000': 0.046, 'skip1200': 0.024}
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
def compare_by_bucket(rows, hooks):
    """Top-1 accuracy of live and program, split by the TRUE TARGET token's fit-row frequency.

    The bucket axis is the TARGET, not the current token: the program is keyed on the current token,
    so this asks what it can PRODUCE rather than what it can condition on."""
    a = {b: {'acc_l': 0, 'acc_p': 0, 'n': 0, 'agree': 0, 'both': 0} for b in BUCKETS}
    keep = {b: {'l': [], 'p': []} for b in BUCKETS}   # per-bucket predictions, for the permutation null
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
    print(f'PERMUTATION-NULL AGREEMENT | buckets {BUCKETS} on the fit-row count of the TRUE '
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
    B = '512'

    def g(e, b, k):
        return res[B][e][b][k]
    en_top = {e: g(e, top, 'enrichment') for e in roles}
    en_bot = {e: g(e, bot, 'enrichment') for e in roles}
    nullratio = {e: g(e, bot, 'permutation_null') / max(g(e, bot, 'broken_S1884_null'), 1e-12)
                 for e in roles}
    partition = all(sum(res[B][e][f'{x}-{y}']['n'] for x, y in BUCKETS)
                    == res[B][e]['overall']['n'] for e in roles)
    pa = all(en_top[e] >= 1.5 for e in roles)
    pb = all(en_bot[e] < en_top[e] for e in roles)
    pc = all(nullratio[e] >= 10.0 for e in roles)
    pd = (ncov == NCOV and partition
          and all(abs(g(e, top, 'agreement') - S1884_AGREE[e]) <= 0.002 for e in roles)
          and all(abs(g(e, top, 'kept_fraction') - S1883_TOP[e]) <= 0.002 for e in roles))

    print(f'\n  THE PAIRING IS REAL (125+ enrichment >= 1.5x) -> {pa}  ' + '  '.join(
        f'{e} agree {g(e, top, "agreement"):.2%} null {g(e, top, "permutation_null"):.2%} '
        f'= {en_top[e]:.2f}x' for e in roles), flush=True)
    print(f'  and the RARE end imitates LESS -> {pb}  ' + '  '.join(
        f'{e} rare {en_bot[e]:.2f}x vs common {en_top[e]:.2f}x' for e in roles), flush=True)
    print(f'  and §1884\'s STRIKE was correct (perm null >= 10x the broken one, rare bucket) -> {pc}  '
          + '  '.join(f'{e} {g(e, bot, "permutation_null"):.4f} vs '
                      f'{g(e, bot, "broken_S1884_null"):.6f} = {nullratio[e]:.0f}x'
                      for e in roles), flush=True)
    print(f'  §1884 agreement and §1883 kept-fractions reproduce, coverage {ncov}, '
          f'partition {partition} -> control {pd}', flush=True)

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
          'predictions': {'pred_a_imitates_not_coincides_program_concentrated_on_frequent_targets': bool(pa),
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
