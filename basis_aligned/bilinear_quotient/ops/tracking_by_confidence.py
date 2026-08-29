# WHAT DOES MODULATE THE TRACKING? -- the model's own confidence, after position turned up flat.
#
# §1910 found the compiled program's agreement with the live model is FLAT across sequence position
# (-0.36 / -0.25 / +0.59x, inconsistent in sign) while the live model's own accuracy climbs on all three
# roles (37.5 -> 39.0, 40.6 -> 41.8, 35.1 -> 38.8%). So the amount of context available does not modulate
# the 6-8x tracking, which rules out the obvious reading of §1885-§1896: that the program follows the
# model precisely where the model is least contextual.
#
# §1895 had already shown the tracking survives on both-wrong positions at 6.10-6.50x, so it is not
# carried by easy agreement either. Two candidate modulators are now eliminated and none is established.
#
# This tries the model's own CONFIDENCE. Split covered scored positions into quartiles by the live
# model's top-2 logit margin -- its own measure of how decided the prediction is -- and measure the
# agreement enrichment in each. §1900 already computes that margin; this reuses it as an axis.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3: §1910's open question.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# Covered arm throughout, named per LESSON 71; baseline computed in-run and reported as an enrichment.
#   pred_a CONFIDENCE MODULATES IT: enrichment in the highest-margin quartile exceeds the lowest, on all
#          three roles. A decided prediction should be one a token-keyed table can also reach; an
#          undecided one is where context should matter most. If FALSE, confidence joins position and
#          correctness as a non-modulator, and the tracking is uniform along every axis tried -- which
#          would be a strong and strange fact about the 6-8x worth stating as such.
#   pred_b AND MATERIALLY: the top-minus-bottom quartile difference is at least 1.0x on all three roles,
#          about 14% of the pooled ~7.2x. Separates a real gradient from a detectable one, exactly as
#          §1910's pred_b did -- where the answer was that there was no gradient at all.
#   pred_c AND THE SPLIT IS SOUND: the live model's top-1 accuracy in the highest-margin quartile exceeds
#          the lowest by at least 20 percentage points. A known-answer check on the axis itself -- a
#          model's margin must predict its own correctness — so a null in pred_a cannot be blamed on a
#          broken split. This is the check that made §1910's null readable and it is registered again.
#   pred_d CONTROLS: coverage is exactly 16,110; the four quartiles partition every scored covered
#          position and are within 2% of equal size by construction; and the covered-arm baseline is
#          computed here and reported as an enrichment.
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
OUT = PT + 'ops/tracking_by_confidence_results.json'
NPERM = 8            # permutations averaged per bucket; the calibration above used 8
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n480_skip80.pt'    # 5,419 types, the DEPLOYED coverage
H = m.transformer.h
# scored-position buckets, module scope: three separate blocks read this and a local binding
# in two of them left the third unbound (LESSON 67's family, caught by the run).
PB = ((0, 0), (1, 1), (2, 2), (3, 3))   # live-margin QUARTILES, assigned per batch
NCOV = 16110       # the DEPLOYED coverage (§1834). §1834's 5419 is S1789_COV; §1788/§1789's accuracy figures
S1789_COV = 5419
S1883_TOP = {'skip7000': 0.536, 'skip11000': 0.541, 'skip1200': 0.539}   # §1883 DEPLOYED column
S1883_BOT = {'skip7000': 0.026, 'skip11000': 0.049, 'skip1200': 0.035}
S1887_UNC = {'skip7000': 3.14, 'skip11000': 3.56, 'skip1200': 3.11}   # §1887 PUBLISHED, 16,110 types
S1888_COV = {'skip7000': 7.19, 'skip11000': 7.29, 'skip1200': 7.64}   # §1888 PUBLISHED, covered arm
SCALES = (1.0,)              # the DEPLOYED build only; §1894 settled the scale question
S1893_TOP1_ORDER = ['g0.80', 'g1.00', 'g1.25', 'g0.50', 'g2.00', 'g4.00']   # §1893 PUBLISHED, skip1200
S1858_CE = {'skip7000': 6.01167, 'skip11000': 5.98477, 'skip1200': 6.00165}   # §1858 PUBLISHED, all-position


@torch.no_grad()
def allpos_ce(rows, hooks):
    """All-position CE at scored positions -- the unit §1866-§1883 is stated in."""
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        lg = forward_logits(idx, hooks)[:, 64:]
        e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                            reduction='none').double()
        tot += float(e.sum()); cnt += int(e.numel())
    return tot / cnt
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


def row_hook(full_rows, s=1.0):
    """Substitute the per-token row, optionally rescaled.

    The scale is applied AT HOOK TIME rather than by materialising a second [50257, D] bank per site:
    §1807 held a raw and a scaled bank and peaked at 26.4 GiB for no reason. This script's ancestor had
    no scale parameter and I assumed one from a different lineage -- PRE-FLIGHT C, verify before you
    assert -- which cost one launch."""
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = full_rows[STATE['idx'].reshape(-1)].reshape(y.shape).to(y.dtype)
        if s != 1.0:
            sub = sub * s
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
    cell = {f'p{a4}_{b4}': {'l': [], 'p': [], 'n': 0, 'al': 0} for a4, b4 in PB}
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
        lg2 = forward_logits(idx)[:, 64:].float()
        t2 = lg2.topk(2, dim=-1)
        marg = t2.values[..., 0] - t2.values[..., 1]
        mc = marg[iscov]
        if mc.numel() >= 4:
            qs = torch.quantile(mc.double(), torch.tensor([0.25, 0.5, 0.75],
                                                          device=DEV, dtype=torch.float64))
        else:
            qs = torch.zeros(3, device=DEV, dtype=torch.float64)
        qidx = torch.bucketize(marg.double(), qs)
        for a4, b4 in PB:
            k4 = f'p{a4}_{b4}'
            m5 = iscov & (qidx == a4)
            cell[k4]['l'].append(al[m5]); cell[k4]['p'].append(ap[m5])
            cell[k4]['n'] += int(m5.sum()); cell[k4]['al'] += int(cl[m5].sum())
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
    # the 2x2 cells. PRE-FLIGHT C: my first attempt anchored this on a `for b in UBANDS` block that
    # exists in a SIBLING branch of this script's lineage and not here, so it never landed and the run
    # died on KeyError 'cell_both_right' after the banks were built.
    for k4 in cell:
        out[f'cell_{k4}'] = {**_en(cell[k4]['l'], cell[k4]['p']), 'n': cell[k4]['n'],
                             'live_acc': cell[k4]['al'] / max(cell[k4]['n'], 1)}
    assert sum(cell[k4]['n'] for k4 in cell) == out['mech_cov']['n'], \
        'the position buckets do not partition the covered arm'
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
    print(f'TRACKING BY MODEL CONFIDENCE | buckets {BUCKETS} on the fit-row count of the TRUE '
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
    # per-site live/table ratios, measured on the same positions, for the PERSITE arm.
    nrm = {}
    ev0 = evs[EVAL_SETS[0][0]]
    for s in sites:
        acc = {'live': 0.0, 'n': 0}

        def mk_n(st):
            def hook(mod, args, out_):
                y = out_[0] if isinstance(out_, tuple) else out_
                acc['live'] += float(y.float().norm(dim=-1).sum())
                acc['n'] += int(y.shape[0] * y.shape[1])
                return None
            return hook
        compare_by_bucket(ev0, [(st, row_hook(fr[st])) for st in sites if st != s]
                          + [(s, mk_n(s))], seen)
        livem = acc['live'] / max(acc['n'], 1)
        tabm = float(fr[s].norm(dim=-1).mean())
        nrm[f'{s[0]}{s[1]}'] = tabm / max(livem, 1e-9)
    res['norms'] = nrm
    print(f'    per-site live/table ratios: min {1 / max(nrm.values()):.1f}x '
          f'max {1 / min(nrm.values()):.1f}x across 36 sites', flush=True)

    res['sites'] = {}
    arms = [('g%.2f' % g, g) for g in SCALES] + [('PERSITE', None)]
    for si, (lbl, g) in enumerate(arms):
        hk = [(st, row_hook(fr[st], s=(1.0 / nrm[f'{st[0]}{st[1]}'] if g is None else g)))
              for st in sites]
        row = {}
        for ename in evs:
            c = compare_by_bucket(evs[ename], hk, seen)
            row[ename] = {'enrichment': c['mech_cov']['enrichment'],
                          'cells': {k4: c[f'cell_{k4}'] for k4 in
                                    [f'p{a4}_{b4}' for a4, b4 in PB]},
                          'allpos_ce': allpos_ce(evs[ename], hk),
                          'top1_prog': c['overall']['top1_acc_prog'],
                          'n': c['mech_cov']['n']}
        res['sites'][lbl] = row
        print(f'    {lbl:8s} ' + '  '.join(
            f'{e} CE {row[e]["allpos_ce"]:.5f} ({row[e]["enrichment"]:5.2f}x)' for e in row)
            + f'   [{si + 1}/{len(arms)}]', flush=True)
        hk = None
        torch.cuda.empty_cache()

    fr, allh, evs = None, None, None
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    BK = [f'p{a4}_{b4}' for a4, b4 in PB]
    A1 = 'g1.00'

    def cl2(e, k4, f):
        return res['sites'][A1][e]['cells'][k4][f]
    lowq, hiq = BK[0], BK[-1]          # quartile 0 = LOWEST live margin, quartile 3 = HIGHEST
    rise = {e: (cl2(e, hiq, 'enrichment') or 0) - (cl2(e, lowq, 'enrichment') or 0) for e in roles}
    accgap = {e: cl2(e, hiq, 'live_acc') - cl2(e, lowq, 'live_acc') for e in roles}
    pa = all(rise[e] > 0 for e in roles)
    pb = all(rise[e] >= 1.0 for e in roles)
    pc = all(accgap[e] >= 0.20 for e in roles)
    sizes = [cl2(e, k4, 'n') for e in roles for k4 in BK]
    pd = (ncov == NCOV and all(cl2(e, k4, 'n') > 0 for e in roles for k4 in BK))

    print(f'\n  enrichment and live accuracy by LIVE-MARGIN quartile (0 = least confident):', flush=True)
    for e in roles:
        print(f'    {e:10s} ' + '  '.join(
            f'q{k4[1]} n{cl2(e, k4, "n"):6d} {(cl2(e, k4, "enrichment") or 0):5.2f}x '
            f'live {cl2(e, k4, "live_acc"):5.1%}' for k4 in BK), flush=True)
    print(f'\n  CONFIDENCE MODULATES THE TRACKING (q3 > q0) -> {pa}  ' + '  '.join(
        f'{e} {(cl2(e, lowq, "enrichment") or 0):.2f} -> {(cl2(e, hiq, "enrichment") or 0):.2f}'
        for e in roles), flush=True)
    print(f'  and MATERIALLY (>= 1.0x) -> {pb}  ' + '  '.join(
        f'{e} {rise[e]:+.2f}x' for e in roles), flush=True)
    print(f'  and the SPLIT is sound (live acc gap >= 20pp) -> {pc}  ' + '  '.join(
        f'{e} {cl2(e, lowq, "live_acc"):.1%} -> {cl2(e, hiq, "live_acc"):.1%} '
        f'({accgap[e] * 100:+.1f}pp)' for e in roles), flush=True)
    print(f'  coverage {ncov}, quartiles populated -> control {pd}; baseline ' + '  '.join(
        f'{e} {res["sites"][A1][e]["enrichment"]:.2f}x' for e in roles), flush=True)

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
          'predictions': {'pred_a_confidence_modulates': bool(pa),
                          'pred_b_materially': bool(pb),
                          'pred_c_split_is_sound': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
