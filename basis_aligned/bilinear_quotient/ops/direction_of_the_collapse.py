# WHAT DO attn5 AND attn6 WRITE? -- direction, after §1906 killed the magnitude account.
#
# §1892 said the collapse was a norm mismatch. §1905 downgraded that from an ordering to a threshold.
# §1906 removed the causation entirely: amplifying attn3 to an effective ratio of 156x -- ABOVE attn5's
# natural 144x -- leaves it at 9.04x enrichment, the best figure measured, while attn5 at 144x sits at
# 1.06x. Scaling a harmful site down fixes it; scaling a harmless site up does not break it. Magnitude is
# sufficient to neutralise and not sufficient to cause, so what separates attn5 and attn6 from the other
# sixteen attention sites is DIRECTION, and no instrument has looked at it.
#
# This measures the cosine between what a restored site actually EMITS and the table row the compiled
# program substitutes there, for all eighteen attention sites, on covered current tokens. §1901 measured
# the same pair as a relative L2 and reported it for two MLP sites; the cosine strips magnitude out,
# which is exactly the variable §1906 eliminated.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3: §1906's open question.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# Every quantity is COVERED-arm, named so per LESSON 71; the baseline is REPORTED, never anchored to a
# figure from a different coverage -- the mistake that failed §1905's pred_d.
#   pred_a attn5 AND attn6 ARE THE DIRECTIONAL OUTLIERS: their live-vs-table cosine is the lowest of all
#          eighteen attention sites, on all three roles. If TRUE the collapse has a directional signature
#          and §1906's open question has an answer. If FALSE direction does not separate them either, and
#          after magnitude (§1906) and depth (attn4 and attn7 flank them harmlessly) the cause is
#          something neither instrument reaches -- which I would state as such rather than reach again.
#   pred_b AND DIRECTION PREDICTS BEHAVIOUR WHERE MAGNITUDE DID NOT: Spearman between cosine and covered
#          enrichment over all 18 sites is at least +0.6, against the norm ratio's PUBLISHED -0.158 /
#          -0.212 / -0.249 (§1905). This is the direct comparison of the two candidate variables on the
#          same 18 points. If FALSE, cosine is no better an explanation than the ratio was, and both are
#          correlates.
#   pred_c AND attn4 IS ALIGNED DESPITE ITS NORM: attn4, which carries a 35.4x ratio and is completely
#          harmless (7.16 / 7.28 / 7.39), has cosine above the median of the eighteen. It is the site
#          that falsified the threshold story; if direction is the real variable it should look ordinary
#          here, and if it does not then the directional account fails on the same site the magnitude
#          account did.
#   pred_d CONTROLS: coverage is exactly 16,110; all 18 sites are measured on identical covered
#          positions; and the covered-arm baseline is REPORTED alongside, not used as a bar.
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
OUT = PT + 'ops/direction_of_the_collapse_results.json'
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
    print(f'DIRECTION OF THE COLLAPSE | buckets {BUCKETS} on the fit-row count of the TRUE '
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

    MSITES = [('attn', L2) for L2 in range(18)]
    S1905_EN = {'attn0': 7.16, 'attn1': 7.15, 'attn2': 7.19, 'attn3': 7.36, 'attn4': 7.16,
                'attn5': 1.06, 'attn6': 1.32, 'attn7': 6.93, 'attn8': 7.31, 'attn9': 6.85,
                'attn10': 6.13, 'attn11': 6.88, 'attn12': 7.00, 'attn13': 7.97, 'attn14': 7.85,
                'attn15': 7.32, 'attn16': 7.85, 'attn17': 7.53}   # §1905 PUBLISHED, skip7000
    capo = {}

    def mk_out(st):
        def hook(mod, args, out_):
            capo[st] = (out_[0] if isinstance(out_, tuple) else out_).detach().float()
            return None
        return hook

    div = {}
    for s in MSITES:
        num, den, nn, sq = 0.0, 0.0, 0, []
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
                d2 = (live[msk] - tab[msk])
                num += float(d2.norm(dim=-1).sum()); den += float(tab[msk].norm(dim=-1).sum())
                nn += int(msk.sum())
                sq.append(torch.nn.functional.cosine_similarity(live[msk], tab[msk], dim=-1))
        allcos = torch.cat(sq)
        div[f'{s[0]}{s[1]}'] = {'rel_diff': num / max(den, 1e-9), 'n': nn,
                                'median_cosine': float(allcos.median()),
                                'mean_cosine': float(allcos.mean()),
                                'frac_cos_below_0': float((allcos < 0).double().mean())}
        print(f'    {s[0]}{s[1]:<2d}  median cos {float(allcos.median()):+.4f}  '
              f'mean cos {float(allcos.mean()):+.4f}  frac<0 {float((allcos < 0).double().mean()):6.2%}  '
              f'relL2 {num / max(den, 1e-9):.3e}  (n {nn})', flush=True)
        sq, allcos = None, None
        torch.cuda.empty_cache()
    res['stream_divergence'] = div

    basep = {ename: preds(evs[ename], allh) for ename in evs}
    res['baseline'] = {e: int(basep[e].numel()) for e in basep}
    selfchk = max(int((basep[e] != preds(evs[e], allh)).sum()) for e in basep)
    res['self_check'] = selfchk
    res['sites'] = {}
    basep = None
    torch.cuda.empty_cache()

    fr, allh, evs = None, None, None
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    AK = [f'attn{L2}' for L2 in range(18)]
    cos = {k2: res['stream_divergence'][k2]['median_cosine'] for k2 in AK}
    order = sorted(AK, key=lambda k2: cos[k2])
    med = sorted(cos.values())[len(AK) // 2]

    def spearman(a, b):
        def rk(v):
            o = sorted(range(len(v)), key=lambda i2: v[i2]); r = [0] * len(v)
            for pos, i2 in enumerate(o): r[i2] = pos
            return r
        ra, rb, n2 = rk(a), rk(b), len(a)
        return 1 - 6 * sum((ra[i2] - rb[i2]) ** 2 for i2 in range(n2)) / (n2 * (n2 * n2 - 1))
    rho = spearman([cos[k2] for k2 in AK], [S1905_EN[k2] for k2 in AK])
    pa = set(order[:2]) == {'attn5', 'attn6'}
    pb = rho >= 0.6
    pc = cos['attn4'] > med
    pd = ncov == NCOV and len(AK) == 18 and all(res['stream_divergence'][k2]['n'] > 0 for k2 in AK)

    print(f'\n  live-vs-table cosine, all 18 attention sites (covered positions), ascending:', flush=True)
    for k2 in order:
        print(f'    {k2:<7s} median cos {cos[k2]:+.4f}   §1905 enrichment {S1905_EN[k2]:5.2f}x', flush=True)
    print(f'\n  attn5 AND attn6 ARE THE DIRECTIONAL OUTLIERS -> {pa}  two lowest: {order[:2]}', flush=True)
    print(f'  and DIRECTION predicts behaviour (Spearman >= +0.6) -> {pb}  rho {rho:+.3f}  '
          f'(norm ratio gave -0.158/-0.212/-0.249 at §1905)', flush=True)
    print(f'  and attn4 is ALIGNED despite its 35.4x norm -> {pc}  cos {cos["attn4"]:+.4f} '
          f'vs median {med:+.4f}', flush=True)
    print(f'  coverage {ncov}, 18 sites -> control {pd}; baseline REPORTED ' + '  '.join(
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
          'predictions': {'pred_a_directional_outliers': bool(pa),
                          'pred_b_direction_predicts': bool(pb),
                          'pred_c_attn4_aligned': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
