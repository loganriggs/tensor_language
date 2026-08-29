# THE §1891 CHANGE-COUNTS, SPLIT BY COVERAGE -- applying LESSON 71 to my own attention numbers.
#
# §1903 resolved the §1898-§1903 loop: §1898's prediction-change counts were computed over ALL scored
# positions while §1899/§1901 measured COVERED ones only, and 99.24 / 99.41 / 100% of the MLP changes sit
# at uncovered current tokens, where the substituted row is §1870's fallback map output rather than the
# model's length-1 output and the no-op premise fails by construction. Codex found it; LESSON 71 is that
# when two numbers disagree, check the populations BEFORE proposing a mechanism.
#
# The same lesson has an immediate consequence I have not checked. §1891 and §1898 also reported
# ATTENTION change-counts from that same all-position comparison -- attn5 changing 96.56 / 96.19 / 96.89%
# of positions, attn13 15181 / 15462 / 8097 -- and I have quoted them on the board twice as evidence that
# attention is where the program's fidelity lives. **If the MLP counts were 99% uncovered, the attention
# counts may be skewed too**, and "attention changes 27x more positions than MLPs" would then be partly a
# statement about the fallback rather than about attention.
#
# The enrichment-based localisation (§1887, §1888, §1891's +0.91 for attn13) was computed on the COVERED
# arm throughout and is not at risk. This is specifically about the change-counts.
#
# ROLES. skip7000, skip11000, skip1200. DISCOVERY ONLY. Rung 3: LESSON 71 applied to my own numbers.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
# Every quantity below names its population, per LESSON 71's corollary.
#   pred_a THE ATTENTION COUNTS ARE NOT A FALLBACK ARTIFACT: for attn5, the COVERED-position change rate
#          is at least 50%, on all three roles. Covered positions are ~76% of the total at this coverage,
#          so a genuine attention effect must show there. If FALSE the 96.6% figure is carried by
#          uncovered positions, "attention changes 27x more than MLPs" is a statement about §1870's
#          fallback map, and I have quoted it wrongly to Codex twice.
#   pred_b AND THE MLP/ATTENTION CONTRAST SURVIVES ON COVERED POSITIONS ALONE: attn5's covered change
#          rate exceeds mlp16's covered change rate by at least 10x, on all three roles. §1903 puts
#          mlp16's covered count near zero, so this should be enormous -- and if it is not, the contrast
#          was a population effect.
#   pred_c AND THE MLP COVERED COUNT IS ~ZERO: mlp16 changes fewer than 30 covered positions on each
#          role, reproducing §1903's 99.24 / 99.41 / 100% localisation from the other direction. A
#          known-answer check on the split itself, per §1889's lesson.
#   pred_d CONTROLS: coverage is exactly 16,110; covered and uncovered counts sum to the all-position
#          count for every site; and those all-position totals reproduce §1898's PUBLISHED figures
#          (mlp16 1321 / 1350 / 650; attn5 35596 / 35458 / 17858).
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
OUT = PT + 'ops/change_counts_by_coverage_results.json'
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
S1898_ALL = {'mlp16': {'skip7000': 1321, 'skip11000': 1350, 'skip1200': 650},
             'attn5': {'skip7000': 35596, 'skip11000': 35458, 'skip1200': 17858}}   # §1898/§1891
PROBE2 = [('attn', 5), ('attn', 13), ('mlp', 16)]
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
    print(f'CHANGE-COUNTS SPLIT BY COVERAGE | buckets {BUCKETS} on the fit-row count of the TRUE '
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

    basep = {ename: preds(evs[ename], allh) for ename in evs}
    covmask = {}
    for ename in evs:
        cc = []
        for i in range(0, evs[ename].shape[0], 8):
            bb = evs[ename][i:i + 8]
            cc.append(seen[bb[:, :-1].to(DEV)[:, 64:]].reshape(-1))
        covmask[ename] = torch.cat(cc)
    res['baseline'] = {e: int(basep[e].numel()) for e in basep}
    print(f'\n  === all-compiled baseline: ' + '  '.join(
        f'{e} {basep[e].numel()} scored positions' for e in basep) + ' ===', flush=True)
    # pred_d's zero-check: the baseline against itself must differ nowhere.
    selfchk = max(int((basep[e] != preds(evs[e], allh)).sum()) for e in basep)
    print(f'    self-comparison differs at {selfchk} positions (must be 0)', flush=True)
    res['self_check'] = selfchk

    res['sites'] = {}
    for si, s in enumerate(PROBE2):
        hk = [(st, row_hook(fr[st])) for st in sites if st != s]
        row = {}
        for ename in evs:
            p2 = preds(evs[ename], hk)
            ch = (p2 != basep[ename])
            cv = covmask[ename]
            row[ename] = {'changed_all': int(ch.sum()),
                          'changed_covered': int((ch & cv).sum()),
                          'changed_uncovered': int((ch & ~cv).sum()),
                          'n_covered': int(cv.sum()), 'n_uncovered': int((~cv).sum()),
                          'rate_covered': float((ch & cv).sum()) / max(float(cv.sum()), 1.0),
                          'rate_uncovered': float((ch & ~cv).sum()) / max(float((~cv).sum()), 1.0)}
            p2 = None
        res['sites'][f'{s[0]}{s[1]}'] = row
        print(f'    restore {s[0]}{s[1]:<2d}  ' + '  '.join(
            f'{e} all {row[e]["changed_all"]:6d} = cov {row[e]["changed_covered"]:6d} '
            f'({row[e]["rate_covered"]:6.2%}) + unc {row[e]["changed_uncovered"]:6d} '
            f'({row[e]["rate_uncovered"]:6.2%})' for e in row), flush=True)
        hk = None
        torch.cuda.empty_cache()
    basep = None
    torch.cuda.empty_cache()

    fr, allh, evs = None, None, None
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]

    def w(k2, e, f):
        return res['sites'][k2][e][f]
    pa = all(w('attn5', e, 'rate_covered') >= 0.50 for e in roles)
    pb = all(w('attn5', e, 'rate_covered')
             >= 10.0 * max(w('mlp16', e, 'rate_covered'), 1e-9) for e in roles)
    pc = all(w('mlp16', e, 'changed_covered') < 30 for e in roles)
    pd = (ncov == NCOV
          and all(w(k2, e, 'changed_covered') + w(k2, e, 'changed_uncovered')
                  == w(k2, e, 'changed_all') for k2 in res['sites'] for e in roles)
          and all(w('mlp16', e, 'changed_all') == S1898_ALL['mlp16'][e] for e in roles)
          and all(w('attn5', e, 'changed_all') == S1898_ALL['attn5'][e] for e in roles))

    print(f'\n  ATTENTION IS NOT A FALLBACK ARTIFACT (attn5 covered rate >= 50%) -> {pa}  ' + '  '.join(
        f'{e} {w("attn5", e, "rate_covered"):.2%}' for e in roles), flush=True)
    print(f'  and the MLP/attention contrast survives on COVERED alone (>=10x) -> {pb}  ' + '  '.join(
        f'{e} attn5 {w("attn5", e, "rate_covered"):.2%} vs mlp16 {w("mlp16", e, "rate_covered"):.4%}'
        for e in roles), flush=True)
    print(f'  and mlp16 changes ~no COVERED positions (<30) -> {pc}  ' + '  '.join(
        f'{e} {w("mlp16", e, "changed_covered")}' for e in roles), flush=True)
    print(f'  splits sum, §1898 totals reproduce, coverage {ncov} -> control {pd}', flush=True)

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
          'predictions': {'pred_a_attention_not_fallback_artifact_program_concentrated_on_frequent_targets': bool(pa),
                          'pred_b_contrast_survives_covered_more_concentrated_than_live': bool(pb),
                          'pred_c_mlp16_covered_near_zero_keeps_60pc_on_the_top_bucket': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
