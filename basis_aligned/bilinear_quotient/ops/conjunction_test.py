# THE CONJUNCTION TEST -- amplify a HIGH-COSINE harmless site, which §1906 never did.
#
# §1907 found that attn5 and attn6, the only two sites that collapse the compiled program, are the two
# whose live output is most nearly PARALLEL to their own table row: median cosine +0.9992 and +0.9866,
# against a spread from +0.1312 (attn16) upward. Spearman between cosine and enrichment over all 18 sites
# is -0.742, three to four times stronger than the norm ratio's -0.158 / -0.212 / -0.249 (§1905).
#
# That reconciled §1906. §1906 amplified attn3 to an effective 156x ratio -- above attn5's natural 144x --
# and it did no harm; but attn3 sits at cosine +0.7001, so amplifying it does not reproduce attn5's
# condition. The two experiments were never testing the same thing.
#
# So the measured statement from §1907 is a CONJUNCTION: the collapsing sites are both nearly parallel to
# their table row AND enormously larger than it, and no healthy site has both. This tests it. attn17 sits
# at cosine +0.9472 -- nearly as aligned as attn6 -- and a natural ratio of only 9.3x, and is completely
# healthy at 7.53x. Amplifying it walks it into attn5's corner along the one axis it lacks.
#
# attn3 (+0.7001) is carried in the same run at the same multipliers as the contrast, so §1906's negative
# result and this one are measured against each other rather than across builds.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3: §1907's named next experiment.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# Baseline REPORTED, not anchored (LESSON 71) -- and reported as an ENRICHMENT this time; §1907's print
# carried a position count mislabelled as one, inherited from §1901.
#   pred_a THE CONJUNCTION IS SUFFICIENT: attn17 falls below 3.0x enrichment at some multiplier in the
#          sweep, on all three roles. If TRUE, high alignment plus large magnitude is enough to destroy
#          the program, no healthy site has both, and §1907's conjunction is the mechanism rather than a
#          coincidence of two sites. If FALSE, alignment plus magnitude is NOT sufficient, attn5 and
#          attn6 have some third property, and I will have spent §1892 through §1908 narrowing a
#          correlate -- which I would say plainly, since I have already corrected this line three times.
#   pred_b AND THE MID-COSINE CONTRAST HOLDS: attn3 stays above 5.0x at every multiplier, reproducing
#          §1906 inside this run. If FALSE, §1906's result does not replicate and the contrast between
#          the two sites is not what §1907 read it as.
#   pred_c AND THE ONSET IS NEAR attn5's RATIO: the smallest attn17 multiplier below 3.0x, times its
#          natural 9.3x, lands within a factor of 3 of attn5's PUBLISHED natural 144x -- i.e. between 48x
#          and 432x. A cheap quantitative check that the constructed condition matches the natural one
#          rather than merely being extreme.
#   pred_d CONTROLS: coverage is exactly 16,110; multiplier 1 reproduces §1905's PUBLISHED natural
#          enrichments (attn17 7.53, attn3 7.36 on skip7000) within 0.05; and the covered-arm baseline is
#          reported as an enrichment alongside, not used as a bar.
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
OUT = PT + 'ops/conjunction_test_results.json'
NPERM = 8            # permutations averaged per bucket; the calibration above used 8
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'    # 5,419 types, the DEPLOYED coverage
H = m.transformer.h
NCOV = 16110       # the DEPLOYED coverage (§1834). §1834's 5419 is S1789_COV; §1788/§1789's accuracy figures
S1789_COV = 5419
S1883_TOP = {'skip7000': 0.536, 'skip11000': 0.541, 'skip1200': 0.539}   # §1883 DEPLOYED column
S1883_BOT = {'skip7000': 0.026, 'skip11000': 0.049, 'skip1200': 0.035}
S1887_UNC = {'skip7000': 3.14, 'skip11000': 3.56, 'skip1200': 3.11}   # §1887 PUBLISHED, 16,110 types
S1888_COV = {'skip7000': 7.19, 'skip11000': 7.29, 'skip1200': 7.64}   # §1888 PUBLISHED, covered arm
PROBE = [('attn', 17), ('attn', 3)]
MULTS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
S1905_NAT = {'attn17': {'ratio': 9.3, 'en': 7.53, 'cos': 0.9472},
             'attn3': {'ratio': 13.0, 'en': 7.36, 'cos': 0.7001}}   # §1905 ratio/en, §1907 cos
S1892_ATTN5_RATIO = 144.0   # §1892/§1905 PUBLISHED, 16,110 types
S1891_UNSCALED = {'attn5': 1.07, 'attn6': 1.37, 'attn10': 6.15, 'attn13': 8.10}   # §1891, skip7000
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
    print(f'THE CONJUNCTION TEST | buckets {BUCKETS} on the fit-row count of the TRUE '
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
    base = {}
    allh = [(st, row_hook(fr[st])) for st in sites]
    print(f'\n  === all-compiled baseline ===', flush=True)
    for ename in evs:
        c = compare_by_bucket(evs[ename], allh, seen)
        base[ename] = c['mech_cov']['enrichment']
        print(f'    {ename:10s} covered enrichment {base[ename]:5.2f}x  '
              f'(n {c["mech_cov"]["n"]})', flush=True)
    res['baseline'] = {e: base[e] for e in base}
    # measure each probe site's LIVE output norm and its TABLE row norm on the same positions, so the
    # rescale factor is data-derived rather than assumed.
    nrm = {}
    for s in PROBE:
        acc = {'live': 0.0, 'n': 0}

        def mk_n(st):
            def hook(mod, args, out_):
                y = out_[0] if isinstance(out_, tuple) else out_
                acc['live'] += float(y.float().norm(dim=-1).sum())
                acc['n'] += int(y.shape[0] * y.shape[1])
                return None
            return hook
        ev0 = evs[EVAL_SETS[0][0]]
        compare_by_bucket(ev0, [(st, row_hook(fr[st])) for st in sites if st != s]
                          + [(s, mk_n(s))], seen)
        livem = acc['live'] / max(acc['n'], 1)
        tabm = float(fr[s].norm(dim=-1).mean())
        nrm[f'{s[0]}{s[1]}'] = {'live_mean_norm': livem, 'table_mean_norm': tabm,
                                'ratio': tabm / max(livem, 1e-9)}
        print(f'    norm {s[0]}{s[1]:<2d}  live {livem:8.3f}  table {tabm:8.3f}  '
              f'rescale x{tabm / max(livem, 1e-9):.4f}', flush=True)
    res['norms'] = nrm

    def scale_hook(f):
        def hook(mod, args, out_):
            y = out_[0] if isinstance(out_, tuple) else out_
            sub = (y * f).to(y.dtype)
            return (sub,) + tuple(out_[1:]) if isinstance(out_, tuple) else sub
        return hook

    res['sites'] = {}
    for si, s in enumerate(PROBE):
        key = f'{s[0]}{s[1]}'
        f = nrm[key]['ratio']
        res['sites'][key] = {}
        for mu in MULTS:
            hk = [(st, row_hook(fr[st])) for st in sites if st != s] + [(s, scale_hook(mu))]
            row = {}
            for ename in evs:
                c = compare_by_bucket(evs[ename], hk, seen)
                row[ename] = {'enrichment': c['mech_cov']['enrichment'], 'n': c['mech_cov']['n']}
            res['sites'][key][f'm{mu:g}'] = row
            print(f'    {key:<7s} x{mu:<5g} eff-ratio {mu * (1.0 / max(f, 1e-12)):7.1f}x  '
                  + '  '.join(f'{e} {row[e]["enrichment"]:5.2f}x' for e in row), flush=True)
            hk = None
            torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    SITES = [f'{s[0]}{s[1]}' for s in PROBE]
    MK = [f'm{mu:g}' for mu in MULTS]

    def en(k2, mk, e):
        return res['sites'][k2][mk][e]['enrichment']
    nat = {k2: 1.0 / max(res['norms'][k2]['ratio'], 1e-12) for k2 in SITES}
    gap, onset, lastok = {}, {}, {}
    for k2 in SITES:
        gap[k2] = {e: max(en(k2, MK[i2], e) - en(k2, MK[i2 + 1], e) for i2 in range(len(MK) - 1))
                   for e in roles}
        onset[k2] = {e: next((MULTS[i2] for i2 in range(len(MK)) if en(k2, MK[i2], e) < 3.0), None)
                     for e in roles}
        lastok[k2] = {e: max([MULTS[i2] for i2 in range(len(MK)) if en(k2, MK[i2], e) > 5.0],
                             default=None) for e in roles}
    onset17 = onset['attn17']
    nat17 = nat['attn17']
    pa = all(onset17[e] is not None for e in roles)
    pb = all(en('attn3', mk, e) > 5.0 for mk in MK for e in roles)
    pc = all(onset17[e] is not None
             and 48.0 <= onset17[e] * nat17 <= 432.0 for e in roles)
    pd = (ncov == NCOV
          and abs(en('attn17', 'm1', 'skip7000') - S1905_NAT['attn17']['en']) <= 0.05
          and abs(en('attn3', 'm1', 'skip7000') - S1905_NAT['attn3']['en']) <= 0.05)

    print(f'\n  THE CONJUNCTION IS SUFFICIENT (attn17 falls below 3.0x) -> {pa}  ' + '  '.join(
        f'{e} onset x{onset17[e]}' if onset17[e] else f'{e} never' for e in roles), flush=True)
    print(f'  and the MID-COSINE contrast holds (attn3 stays > 5.0x) -> {pb}  ' + '  '.join(
        f'{e} min {min(en("attn3", mk, e) for mk in MK):.2f}x' for e in roles), flush=True)
    print(f'  and the onset is near attn5\'s 144x (48x-432x) -> {pc}  ' + '  '.join(
        f'{e} {(onset17[e] or 0) * nat17:.0f}x' for e in roles), flush=True)
    print(f'  coverage {ncov}, x1 reproduces §1905 -> control {pd}  attn17 '
          f'{en("attn17", "m1", "skip7000"):.2f} vs {S1905_NAT["attn17"]["en"]}  attn3 '
          f'{en("attn3", "m1", "skip7000"):.2f} vs {S1905_NAT["attn3"]["en"]}', flush=True)
    print(f'  covered-arm baseline ENRICHMENT, reported not anchored: ' + '  '.join(
        f'{e} {base[e]:.2f}x' for e in roles), flush=True)

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
          'predictions': {'pred_a_conjunction_sufficient': bool(pa),
                          'pred_b_midcosine_contrast_holds': bool(pb),
                          'pred_c_onset_near_attn5_ratio': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
