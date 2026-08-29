# THE RELIABILITY SIGNAL AT DEPLOYED COVERAGE -- second-class confirmation of §1918.
#
# §1918 established that the compiled program's per-token margin is a precomputable reliability signal
# covering the whole vocabulary: within-token spread 0.00e+00 / 0.00e+00 / 5.72e-06 at UNCOVERED
# positions, ordering agreement at +3.90 / +4.10 / +2.06x there against +9.50 / +9.79 / +9.28x on the
# covered arm. §1919 then showed raising §1870's map rank from 64 to 512 lifts the uncovered gradient by
# +0.78 / +0.63 / +0.70, and §1920 confirmed those gains clear their own noise by 3.9x / 2.0x / 2.2x.
#
# Every one of those was measured at 16,110 covered types. The DEPLOYED build (§1789, §1834) runs at
# 5,419, where the uncovered arm is roughly 24% of scored positions rather than ~10% and its token
# population is quite different -- the extra 10,691 types that are covered at 16,110 are uncovered here.
# So "the signal is a property of the program" and "the signal is a property of the high-coverage build"
# are not yet distinguished, and the deployed build is the one anyone would ship.
#
# House pattern: a just-certified result gets a second-class confirmation before anything is built on it
# (§1595, §1598, §1603, §1886, §1888, §1896).
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 2.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# Populations named per LESSON 71; §1918/§1919's figures are at 16,110 and are what is being generalised,
# NOT controls -- anchoring a 5,419-type run to a 16,110-type constant is the trap that failed §1905's
# pred_d and cost §1882 a launch.
#   pred_a IT IS STILL A TOKEN PROPERTY: the within-token spread of the program's margin is below 1e-3 on
#          all three roles, at BOTH covered and uncovered positions. §1765 does not mention coverage, so
#          this should hold -- and if it does not, the signal is not precomputable for the build anyone
#          would actually deploy, which would matter more than any of §1918's numbers.
#   pred_b AND IT STILL ORDERS AGREEMENT ON BOTH ARMS: the covered and uncovered gradients are each at
#          least +1.0x on all three roles. A deliberately weak bar: the point is whether the signal
#          exists at the deployed coverage at all, not whether it matches §1918's size.
#   pred_c AND THE COVERED/UNCOVERED ORDERING SURVIVES: the covered gradient exceeds the uncovered one on
#          all three roles, as §1918 found (+9.50 vs +3.90 etc.). This is the qualitative shape of the
#          result. If FALSE the fallback carries as much reliability information as the tables do at the
#          deployed coverage, which would be a genuinely surprising fact about §1870's map and would mean
#          §1918's gap is a high-coverage artifact.
#   pred_d CONTROLS: coverage is exactly 5,419 -- §1834's published deployed figure; zero lookup misses
#          across the whole vocabulary; and the uncovered share of scored positions is REPORTED, since it
#          should be roughly 24% here against ~10% at 16,110 and a run that did not actually move the
#          split would confirm nothing.
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
OUT = PT + 'ops/reliability_at_deployed_coverage_results.json'
NPERM = 8            # permutations averaged per bucket; the calibration above used 8
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'    # 5,419 types, the DEPLOYED coverage
H = m.transformer.h
# scored-position buckets, module scope: three separate blocks read this and a local binding
# in two of them left the third unbound (LESSON 67's family, caught by the run).
PB = ((0, 0), (1, 1), (2, 2), (3, 3))   # PRECOMPUTED-TOKEN-margin quartiles
TOKMARG = None   # [V] the program's top-2 margin per token, from a length-1 pass
TOKSEEN = None   # [V] 1 where TOKMARG is populated
TOKQS = None     # quartile boundaries over COVERED tokens
UNCQS = None     # quartile boundaries over UNCOVERED tokens
S1911_EN = {'skip7000': [5.36, 5.74, 6.23, 10.11]}   # §1911 PUBLISHED, covered arm, q0..q3
NCOV = 5419       # the DEPLOYED coverage (§1834). §1834's 5419 is S1789_COV; §1788/§1789's accuracy figures
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
    both = {'dev': 0.0, 'miss': 0}
    allcov = {'l': [], 'p': []}                       # the whole covered arm, for the GLOBAL null
    tokq = {i8: {'l': [], 'p': []} for i8 in range(4)}   # COVERED, by precomputed token margin
    uncq = {i8: {'l': [], 'p': []} for i8 in range(4)}   # UNCOVERED, same signal
    alluncv = {'l': [], 'p': []}                          # the whole UNCOVERED arm
    livq = {i8: {'l': [], 'p': []} for i8 in range(4)}   # split by LIVE margin
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
        lgp = forward_logits(idx, hooks)[:, 64:].float()
        tp = lgp.topk(2, dim=-1)
        pmarg = tp.values[..., 0] - tp.values[..., 1]
        # the PRECOMPUTED per-token margin, looked up by the current token
        tokm = TOKMARG[cur]
        dev = (pmarg - tokm).abs()[iscov]
        if dev.numel():
            both['dev'] = max(both['dev'], float(dev.max()))
        both['miss'] += int((TOKSEEN[cur] == 0)[iscov].sum())
        qidx = torch.bucketize(tokm.double(), TOKQS)
        uidx = torch.bucketize(tokm.double(), UNCQS)
        alluncv['l'].append(al[~iscov]); alluncv['p'].append(ap[~iscov])
        allcov['l'].append(al[iscov]); allcov['p'].append(ap[iscov])
        mc = marg[iscov]
        lqs = (torch.quantile(mc.double(), torch.tensor([0.25, 0.5, 0.75], device=DEV,
                                                        dtype=torch.float64))
               if mc.numel() >= 4 else torch.zeros(3, device=DEV, dtype=torch.float64))
        lidx = torch.bucketize(marg.double(), lqs)
        for i8 in range(4):
            mt = iscov & (qidx == i8)
            tokq[i8]['l'].append(al[mt]); tokq[i8]['p'].append(ap[mt])
            mu2 = (~iscov) & (uidx == i8)
            uncq[i8]['l'].append(al[mu2]); uncq[i8]['p'].append(ap[mu2])
            ml = iscov & (lidx == i8)
            livq[i8]['l'].append(al[ml]); livq[i8]['p'].append(ap[ml])
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
        'the live-margin quartiles do not partition the covered arm'
    out['max_dev_from_length1'] = both['dev']
    out['lookup_misses'] = both['miss']
    GL, GP = torch.cat(allcov['l']), torch.cat(allcov['p'])
    # the GLOBAL null: shuffle across the WHOLE covered arm, the same denominator for every stratum.
    gnull = sum(float((GL == GP[torch.randperm(GP.numel(), generator=gen).to(GP.device)])
                      .double().mean()) for _ in range(NPERM)) / NPERM
    out['global_null'] = gnull
    UL, UP = torch.cat(alluncv['l']), torch.cat(alluncv['p'])
    unull = sum(float((UL == UP[torch.randperm(UP.numel(), generator=gen).to(UP.device)])
                      .double().mean()) for _ in range(NPERM)) / NPERM
    out['uncovered_null'] = unull
    out['uncovered_n'] = int(UL.numel())
    for i8 in range(4):
        L9, P9 = torch.cat(uncq[i8]['l']), torch.cat(uncq[i8]['p'])
        ob9 = float((L9 == P9).double().mean()) if L9.numel() else 0.0
        out[f'uncq{i8}'] = {'global_enrichment': ob9 / max(unull, 1e-12), 'n': int(L9.numel())}
    for nm, dd in (('tok', tokq), ('liv', livq)):
        for i8 in range(4):
            L8, P8 = torch.cat(dd[i8]['l']), torch.cat(dd[i8]['p'])
            ob = float((L8 == P8).double().mean()) if L8.numel() else 0.0
            loc = _en(dd[i8]['l'], dd[i8]['p'])
            out[f'{nm}q{i8}'] = {'agreement': ob, 'global_enrichment': ob / max(gnull, 1e-12),
                                 'local_enrichment': loc['enrichment'], 'n': int(L8.numel())}
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
    print(f'RELIABILITY AT DEPLOYED COVERAGE 5419 | buckets {BUCKETS} on the fit-row count of the TRUE '
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

    # PRECOMPUTE the per-token program margin: one length-1 forward per covered type, at build time.
    global TOKMARG, TOKSEEN, TOKQS, UNCQS
    TOKMARG = torch.zeros(V, device=DEV)
    TOKSEEN = torch.zeros(V, device=DEV)
    _ct = torch.arange(V, device=DEV)   # WHOLE vocabulary: uncovered tokens need one too
    for i7 in range(0, _ct.shape[0], 512):
        tt7 = _ct[i7:i7 + 512].unsqueeze(1)
        lg7 = forward_logits(tt7, allh)[:, 0].float()
        t7 = lg7.topk(2, dim=-1)
        TOKMARG[_ct[i7:i7 + 512]] = t7.values[..., 0] - t7.values[..., 1]
        TOKSEEN[_ct[i7:i7 + 512]] = 1.0
    _cov_ct = seen.nonzero(as_tuple=True)[0]
    TOKQS = torch.quantile(TOKMARG[_cov_ct].double(),
                           torch.tensor([0.25, 0.5, 0.75], device=DEV, dtype=torch.float64))
    print(f'  precomputed per-token program margin for {int(_ct.numel())} covered types '
          f'(covered-token quartile bounds {[round(float(x), 4) for x in TOKQS]})', flush=True)
    global UNCQS
    _unc_ct = (~seen).nonzero(as_tuple=True)[0]
    UNCQS = torch.quantile(TOKMARG[_unc_ct].double(),
                           torch.tensor([0.25, 0.5, 0.75], device=DEV, dtype=torch.float64))
    print(f'  uncovered-token quartile bounds {[round(float(x), 4) for x in UNCQS]} '
          f'over {int(_unc_ct.numel())} types', flush=True)
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
                          # carry the three quantities THIS run adds to compare_by_bucket. The row dict
                          # is a fixed whitelist inherited from an ancestor, so anything new in `out`
                          # is silently dropped here -- which KeyError'd at the report step after the
                          # whole arm had run.
                          'max_dev_from_length1': c['max_dev_from_length1'],
                          'lookup_misses': c['lookup_misses'],
                          'global_null': c['global_null'],
                          'uncovered_null': c['uncovered_null'], 'uncovered_n': c['uncovered_n'],
                          'u': {f'uncq{i8}': c[f'uncq{i8}'] for i8 in range(4)},
                          'q': {f'{nm}q{i8}': c[f'{nm}q{i8}']
                                for nm in ('tok', 'liv') for i8 in range(4)},
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
    def top(e, f):
        return res['sites'][A1][e][f]

    def q(e, nm, i8, f):
        return res['sites'][A1][e]['q'][f'{nm}q{i8}'][f]
    S1914_TOK = {'skip7000': 9.50, 'skip11000': 9.79, 'skip1200': 9.28}

    def u(e, i8, f):
        return res['sites'][A1][e]['u'][f'uncq{i8}'][f]
    gunc = {e: u(e, 3, 'global_enrichment') - u(e, 0, 'global_enrichment') for e in roles}
    gcov = {e: q(e, 'tok', 3, 'global_enrichment') - q(e, 'tok', 0, 'global_enrichment') for e in roles}
    dev = {e: top(e, 'max_dev_from_length1') for e in roles}
    miss = {e: top(e, 'lookup_misses') for e in roles}
    uncn = {e: sum(u(e, i8, 'n') for i8 in range(4)) for e in roles}
    totn = {e: uncn[e] + sum(q(e, 'tok', i8, 'n') for i8 in range(4)) for e in roles}
    pa = all(2.0 * dev[e] < 1e-3 for e in roles)
    pb = all(gcov[e] >= 1.0 and gunc[e] >= 1.0 for e in roles)
    pc = all(gcov[e] > gunc[e] for e in roles)
    pd = (ncov == NCOV and all(miss[e] == 0 for e in roles))

    print(f'\n  agreement by PRECOMPUTED token margin at DEPLOYED coverage {ncov}:', flush=True)
    for e in roles:
        print(f'    {e:10s} COVERED   ' + '  '.join(
            f'q{i8} {q(e, "tok", i8, "global_enrichment"):5.2f}x' for i8 in range(4))
            + f'   grad {gcov[e]:+.2f}   (§1918 at 16,110: +9.50/+9.79/+9.28)', flush=True)
        print(f'    {"":10s} UNCOVERED ' + '  '.join(
            f'q{i8} {u(e, i8, "global_enrichment"):5.2f}x (n{u(e, i8, "n"):5d})' for i8 in range(4))
            + f'   grad {gunc[e]:+.2f}   (§1918: +3.90/+4.10/+2.06)', flush=True)
    print(f'\n  STILL A TOKEN PROPERTY at 5,419 (< 1e-3) -> {pa}  ' + '  '.join(
        f'{e} <= {2.0 * dev[e]:.2e}' for e in roles), flush=True)
    print(f'  and it ORDERS agreement on BOTH arms (>= 1.0x) -> {pb}  ' + '  '.join(
        f'{e} cov {gcov[e]:+.2f} unc {gunc[e]:+.2f}' for e in roles), flush=True)
    print(f'  and COVERED still exceeds UNCOVERED -> {pc}  ' + '  '.join(
        f'{e} {gcov[e]:+.2f} > {gunc[e]:+.2f}' for e in roles), flush=True)
    print(f'  coverage {ncov}, zero misses -> control {pd}; uncovered share ' + '  '.join(
        f'{e} {uncn[e] / max(totn[e], 1):.1%}' for e in roles) + ' (~10% at 16,110)', flush=True)

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
          'predictions': {'pred_a_token_property_at_5419': bool(pa),
                          'pred_b_orders_both_arms': bool(pb),
                          'pred_c_covered_exceeds_uncovered': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
