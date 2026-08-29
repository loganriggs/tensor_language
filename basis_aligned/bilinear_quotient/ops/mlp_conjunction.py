# DOES THE CONJUNCTION GOVERN THE MLP SITES TOO? -- §1908's open question.
#
# §1908 established, by a designed falsification test, that a restored ATTENTION site destroys the
# compiled program when it is simultaneously near-parallel to its own table row and far larger than it:
# attn17 (cosine +0.9472) amplified collapses 7.53 -> 2.74x at an effective 149.3x, matching attn5's
# natural 144.0x within 4%, while attn3 (cosine +0.7001) amplified to 415.9x climbs to 12.17x. Same
# multipliers, opposite outcomes, cosine the only difference.
#
# Nothing has measured the MLP sites on either axis. §1903 showed MLP restoration is a no-op at COVERED
# current tokens -- live output matches the table row to 3.232e-07 (§1901) -- and moves ~35% of
# predictions at UNCOVERED ones, where the substituted row is §1870's fallback map output rather than the
# model's length-1 output. §1898 measured the per-MLP change rate rising from 0.003% at mlp0 to 3.58% at
# mlp16 with a dip at mlp7-mlp8.
#
# If §1908's conjunction is a property of the PROGRAM rather than of attention, it should predict that
# uncovered-token profile: sites whose fallback row is both nearly parallel to the live output and far
# from it in scale should be the ones that move the most predictions.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3: §1908's open question.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# Every quantity NAMES ITS POPULATION (LESSON 71) -- covered and uncovered are reported separately
# throughout, since conflating them is what cost §1898-§1903 six sections.
#   pred_a THE COVERED SIDE IS THE KNOWN ANSWER: at covered current tokens every one of the 18 MLP sites
#          has median cosine above 0.999, since §1901 measured live-vs-table at 3.232e-07 there. If
#          FALSE the instrument disagrees with §1901 on a settled quantity and nothing else in the run
#          can be read. §1889's lesson: a measurement that cannot return a known answer is not measuring.
#   pred_b AND THE UNCOVERED SIDE IS FAR FROM ALIGNED: the median uncovered cosine over the 18 sites is
#          below 0.80, on all three roles. This is what §1903 implies without ever having measured it --
#          the fallback row is a rank-64 map output and has no reason to point where the model does. If
#          FALSE the fallback rows ARE well aligned and §1903's 35% prediction changes come from scale
#          alone, which would be a different and simpler story than §1908's.
#   pred_c AND THE CONJUNCTION PREDICTS THE §1898 PROFILE: Spearman between each MLP's uncovered
#          conjunction score (median cosine x live/table norm ratio) and §1898's PUBLISHED per-site
#          change rate (0.003 / 0.13 / 0.13 / 0.20 / 1.21 / 1.61 / 1.60 / 0.84 / 0.81 / 1.27 / 1.07 /
#          1.53 / 1.33 / 1.40 / 1.34 / 2.32 / 3.58 / 3.15 %) is at least +0.6. If TRUE, §1908's law is a
#          property of the compiled program and not of attention, and it explains a profile measured two
#          days of sections earlier by a different instrument. If FALSE the law is attention-specific,
#          which bounds it and is worth knowing before anyone generalises it.
#   pred_d CONTROLS: coverage is exactly 16,110; covered and uncovered populations are disjoint and sum
#          to every scored position; and the covered-arm baseline is reported as an ENRICHMENT, not
#          anchored (§1905's pred_d failed on a cross-coverage anchor; §1907 printed a position count
#          mislabelled as one).
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
OUT = PT + 'ops/mlp_conjunction_results.json'
NPERM = 8            # permutations averaged per bucket; the calibration above used 8
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'   # 16,110 types
H = m.transformer.h
NCOV = 16110      # high coverage; §1834's 5419 is S1789_COV; §1788/§1789's accuracy figures
S1789_COV = 5419
S1883_TOP = {'skip7000': 0.536, 'skip11000': 0.541, 'skip1200': 0.539}   # §1883 DEPLOYED column
S1883_BOT = {'skip7000': 0.026, 'skip11000': 0.049, 'skip1200': 0.035}
S1887_UNC = {'skip7000': 3.14, 'skip11000': 3.56, 'skip1200': 3.11}   # §1887 PUBLISHED, 16,110 types
S1888_COV = {'skip7000': 7.19, 'skip11000': 7.29, 'skip1200': 7.64}   # §1888 PUBLISHED, covered arm
S1898_MLP = [0.003, 0.13, 0.13, 0.20, 1.21, 1.61, 1.60, 0.84, 0.81, 1.27,
             1.07, 1.53, 1.33, 1.40, 1.34, 2.32, 3.58, 3.15]   # §1898 PUBLISHED, skip7000
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
        return {'agreement': ob, 'permutation_null': pm,
                'enrichment': (ob / max(pm, 1e-12)) if L.numel() >= 100 else None,
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
    print(f'DOES THE CONJUNCTION GOVERN THE MLPs? | buckets {BUCKETS} on the fit-row count of the TRUE '
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
    fr = program_rows(None)
    evs = {}
    for ename, epath, _ce in EVAL_SETS:
        evs[ename] = load(epath)
    allh = [(st, row_hook(fr[st])) for st in sites]

    @torch.no_grad()
    def preds(rows, hooks):
        """argmax predictions at every scored position -- the object the derivation is about."""
        out2 = []
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            out2.append(forward_logits(idx, hooks)[:, 64:].argmax(-1).reshape(-1))
        return torch.cat(out2)

    MSITES = [('mlp', L2) for L2 in range(18)]
    S1898_RATE = [0.003, 0.13, 0.13, 0.20, 1.21, 1.61, 1.60, 0.84, 0.81, 1.27,
                  1.07, 1.53, 1.33, 1.40, 1.34, 2.32, 3.58, 3.15]   # §1898 PUBLISHED, skip7000, mlp0..17
    capo = {}

    def mk_out(st):
        def hook(mod, args, out_):
            capo[st] = (out_[0] if isinstance(out_, tuple) else out_).detach().float()
            return None
        return hook

    div = {}
    for s in MSITES:
        num, den, nn, sq = 0.0, 0.0, 0, []
        acc2 = {p2: {'cos': [], 'lv': 0.0, 'tb': 0.0, 'n': 0} for p2 in ('cov', 'unc')}
        for ename in evs:
            ev1 = evs[ename]
            for i in range(0, ev1.shape[0], 4):
                bb = ev1[i:i + 4]
                idx = bb[:, :-1].to(DEV).contiguous()
                cur = idx[:, 64:]
                # the RESTORED configuration: 35 compiled, s LIVE, capture what s actually emits
                forward_logits(idx, [(st, row_hook(fr[st])) for st in sites if st != s]
                               + [(s, mk_out(s))])
                live = capo[s][:, 64:].reshape(-1, D)
                flat = cur.reshape(-1)
                tab = fr[s][flat]                    # the row the all-compiled arm substitutes
                msk = seen[flat]
                for _nm, _m6 in (('cov', msk), ('unc', ~msk)):
                    if int(_m6.sum()) == 0:
                        continue
                    acc2[_nm]['cos'].append(
                        torch.nn.functional.cosine_similarity(live[_m6], tab[_m6], dim=-1))
                    acc2[_nm]['lv'] += float(live[_m6].norm(dim=-1).sum())
                    acc2[_nm]['tb'] += float(tab[_m6].norm(dim=-1).sum())
                    acc2[_nm]['n'] += int(_m6.sum())
                nn += int(msk.sum())
        rec = {}
        for p2 in ('cov', 'unc'):
            a2 = acc2[p2]
            c2 = torch.cat(a2['cos']) if a2['cos'] else torch.zeros(1, device=DEV)
            rec[p2] = {'median_cosine': float(c2.median()),
                       'norm_ratio': a2['lv'] / max(a2['tb'], 1e-9), 'n': a2['n']}
            rec[p2]['conjunction'] = rec[p2]['median_cosine'] * rec[p2]['norm_ratio']
        div[f'{s[0]}{s[1]}'] = rec
        print(f'    {s[0]}{s[1]:<2d}  COVERED cos {rec["cov"]["median_cosine"]:+.4f} ratio '
              f'{rec["cov"]["norm_ratio"]:6.2f}x (n {rec["cov"]["n"]})  |  UNCOVERED cos '
              f'{rec["unc"]["median_cosine"]:+.4f} ratio {rec["unc"]["norm_ratio"]:7.2f}x '
              f'conj {rec["unc"]["conjunction"]:7.2f} (n {rec["unc"]["n"]})', flush=True)
        sq, acc2 = None, None
        torch.cuda.empty_cache()
    res['stream_divergence'] = div

    basep = {ename: preds(evs[ename], allh) for ename in evs}
    res['baseline_positions'] = {e: int(basep[e].numel()) for e in basep}
    # the covered-arm ENRICHMENT, computed here rather than inherited: §1907 printed the position
    # count under a '...x' label because this lineage stored a count in res['baseline'].
    basecov = {}
    for ename in evs:
        basecov[ename] = compare_by_bucket(evs[ename], allh, seen)['mech_cov']['enrichment']
    res['baseline'] = basecov
    selfchk = max(int((basep[e] != preds(evs[e], allh)).sum()) for e in basep)
    res['self_check'] = selfchk
    res['sites'] = {}
    basep = None
    torch.cuda.empty_cache()

    fr, allh, evs = None, None, None
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    MK2 = [f'mlp{L2}' for L2 in range(18)]
    D2 = res['stream_divergence']
    covc = [D2[k2]['cov']['median_cosine'] for k2 in MK2]
    uncc = [D2[k2]['unc']['median_cosine'] for k2 in MK2]
    conj = [D2[k2]['unc']['conjunction'] for k2 in MK2]

    def spearman(a, b):
        def rk(v):
            o = sorted(range(len(v)), key=lambda i2: v[i2]); r = [0] * len(v)
            for pos, i2 in enumerate(o): r[i2] = pos
            return r
        ra, rb, n2 = rk(a), rk(b), len(a)
        return 1 - 6 * sum((ra[i2] - rb[i2]) ** 2 for i2 in range(n2)) / (n2 * (n2 * n2 - 1))
    rho = spearman(conj, S1898_RATE)
    medunc = sorted(uncc)[len(uncc) // 2]
    pa = all(c > 0.999 for c in covc)
    pb = medunc < 0.80
    pc = rho >= 0.6
    pd = (ncov == NCOV
          and all(D2[k2]['cov']['n'] + D2[k2]['unc']['n'] > 0 for k2 in MK2)
          and len(MK2) == 18)

    print(f'\n  per-MLP: UNCOVERED cosine, ratio, conjunction, vs §1898 change rate:', flush=True)
    for i2, k2 in enumerate(MK2):
        print(f'    {k2:<6s} cos {uncc[i2]:+.4f}  ratio {D2[k2]["unc"]["norm_ratio"]:7.2f}x  '
              f'conj {conj[i2]:7.2f}   §1898 rate {S1898_RATE[i2]:5.2f}%', flush=True)
    print(f'\n  COVERED side is the known answer (all cos > 0.999) -> {pa}  min {min(covc):+.6f}',
          flush=True)
    print(f'  and the UNCOVERED side is far from aligned (median < 0.80) -> {pb}  '
          f'median {medunc:+.4f}', flush=True)
    print(f'  and the conjunction predicts §1898\'s profile (Spearman >= +0.6) -> {pc}  rho {rho:+.3f}',
          flush=True)
    print(f'  coverage {ncov}, 18 sites -> control {pd}; covered-arm baseline ENRICHMENT ' + '  '.join(
        f'{e} {res["baseline"][e]:.2f}x' for e in roles), flush=True)

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
          'predictions': {'pred_a_covered_known_answer': bool(pa),
                          'pred_b_uncovered_unaligned': bool(pb),
                          'pred_c_conjunction_predicts_profile': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
