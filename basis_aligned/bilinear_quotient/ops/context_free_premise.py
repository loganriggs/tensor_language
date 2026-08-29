# IS THE COMPILED STREAM ACTUALLY CONTEXT-FREE? -- testing §1765's premise directly.
#
# §1898 retracted §1891's claim that MLP restorations are a provable no-op. The derivation was: with the
# other 35 sites compiled the residual stream is a function of the current token alone (§1765), an MLP is
# position-wise, therefore a live MLP returns its own table value. Measured directly, restoring an MLP
# changes up to 3.58 / 3.66 / 3.53% of predicted tokens -- rising from 0.003% at mlp0 with a dip at
# mlp7-mlp8 -- so the argument fails, and it fails by composition rather than at any single site.
#
# I offered a hypothesis in §1898 and labelled it untested: bilin18 threads a value residual `v1` and the
# initial embedding `x0` through `blk(x, v1, x0)`, and `v1` is computed from LIVE attention internals
# rather than from the substituted output, so the compiled stream need not equal the length-1 stream.
# §1888 and §1890 are why I label hypotheses now, and §1890 is why I run them.
#
# This tests the PREMISE rather than the hypothesis, which is the cleaner target. For each site, on a
# full-length eval sequence, compare the stream the COMPILED PROGRAM feeds that site at position t
# against the stream the MODEL feeds it on a LENGTH-1 sequence containing only the token at position t.
# §1765 asserts these are equal. If they diverge, and diverge in the shape §1898 measured, the premise is
# where the derivation broke -- and that localises it without needing to be right about v1.
#
# ROLES. skip7000 only for the stream comparison (it is a per-position tensor diff, not a CE), with
# skip11000 and skip1200 carried for the change-count correlation. DISCOVERY ONLY. Rung 3: §1898's
# named open question.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39.
#   pred_a THE PREMISE FAILS BY DEPTH: the median relative difference between the compiled stream and the
#          model's length-1 stream exceeds 1% at mlp16. If FALSE the streams agree and §1765's premise is
#          intact, which would mean the derivation broke somewhere I have not identified at all -- and
#          the honest §1898 statement would become "the argument fails for reasons unknown".
#   pred_b AND IT HOLDS AT DEPTH 0: the same quantity at mlp0 is below 1e-4. The stream entering mlp0 is
#          the normalised embedding, which IS a function of the current token, so this is a known-answer
#          check on the instrument -- a measurement that reports divergence everywhere is measuring a bug.
#          §1889 is why this is registered.
#   pred_c AND THE SHAPE MATCHES §1898: the per-site divergence and §1898's PUBLISHED per-MLP change
#          counts (0.003 / 0.13 / 0.13 / 0.20 / 1.21 / 1.61 / 1.60 / 0.84 / 0.81 / 1.27 / 1.07 / 1.53 /
#          1.33 / 1.40 / 1.34 / 2.32 / 3.58 / 3.15 %) have Spearman rank correlation at least 0.5. Two
#          different instruments on the same 18 sites; if they do not order the sites alike, stream
#          divergence is not what drives the prediction changes and the localisation is wrong.
#   pred_d CONTROLS: coverage is exactly 16,110; all 18 MLP sites are compared on identical positions;
#          and the comparison is run on COVERED current tokens only, since an uncovered token has no
#          table row and its stream is a fallback-map artifact rather than a test of §1765.
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
OUT = PT + 'ops/context_free_premise_results.json'
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
    print(f'IS THE COMPILED STREAM CONTEXT-FREE? | buckets {BUCKETS} on the fit-row count of the TRUE '
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

    ev1 = evs[EVAL_SETS[0][0]]
    MSITES = [('mlp', L2) for L2 in range(18)]
    capp, capl = {}, {}

    def mk_cap(st, into):
        def hook(mod, args, out_):
            into[st] = args[0].detach().float()
            return None
        return hook

    div = {}
    for s in MSITES:
        num, den, nn = 0.0, 0.0, 0
        for i in range(0, ev1.shape[0], 4):
            bb = ev1[i:i + 4]
            idx = bb[:, :-1].to(DEV).contiguous()
            cur = idx[:, 64:]
            # the COMPILED PROGRAM's stream entering site s, at every scored position
            forward_logits(idx, [(st, row_hook(fr[st])) for st in sites] + [(s, mk_cap(s, capp))])
            sp = capp[s][:, 64:]
            # the MODEL's LENGTH-1 stream entering site s, for the same tokens
            flat = cur.reshape(-1)
            m1 = torch.zeros(flat.shape[0], D, device=DEV)
            for j in range(0, flat.shape[0], 512):
                tt = flat[j:j + 512].unsqueeze(1)
                forward_logits(tt, [(s, mk_cap(s, capl))])
                m1[j:j + tt.shape[0]] = capl[s][:, 0]
            sp = sp.reshape(-1, D)
            msk = seen[flat]                       # COVERED current tokens only
            num += float((sp[msk] - m1[msk]).norm(dim=-1).sum())
            den += float(m1[msk].norm(dim=-1).sum())
            nn += int(msk.sum())
        div[f'{s[0]}{s[1]}'] = {'rel_diff': num / max(den, 1e-9), 'n': nn}
        print(f'    {s[0]}{s[1]:<2d}  relative stream difference {num / max(den, 1e-9):.6f}  '
              f'(n {nn})', flush=True)
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
    dv = [res['stream_divergence'][f'mlp{L2}']['rel_diff'] for L2 in range(18)]

    def rank(v):
        o = sorted(range(len(v)), key=lambda i2: v[i2])
        r = [0] * len(v)
        for pos, i2 in enumerate(o):
            r[i2] = pos
        return r

    ra, rb = rank(dv), rank(S1898_MLP)
    n2 = len(dv)
    sp = 1 - 6 * sum((ra[i2] - rb[i2]) ** 2 for i2 in range(n2)) / (n2 * (n2 * n2 - 1))
    pa = dv[16] > 0.01
    pb = dv[0] < 1e-4
    pc = sp >= 0.5
    pd = ncov == NCOV and res['self_check'] == 0 and len(dv) == 18

    print(f'\n  relative stream difference by depth (compiled program vs the model at length 1):',
          flush=True)
    print(f'    ' + '  '.join(f'mlp{L2}:{dv[L2]:.4f}' for L2 in range(18)), flush=True)
    print(f'\n  THE PREMISE FAILS BY DEPTH (mlp16 > 1%) -> {pa}  mlp16 {dv[16]:.4%}', flush=True)
    print(f'  and it HOLDS at depth 0 (mlp0 < 1e-4) -> {pb}  mlp0 {dv[0]:.3e}', flush=True)
    print(f'  and the SHAPE matches §1898 (Spearman >= 0.5) -> {pc}  rho {sp:.3f}', flush=True)
    print(f'  coverage {ncov}, 18 sites, self-check {res["self_check"]} -> control {pd}', flush=True)

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
          'predictions': {'pred_a_premise_fails_by_depth_program_concentrated_on_frequent_targets': bool(pa),
                          'pred_b_holds_at_depth0_more_concentrated_than_live': bool(pb),
                          'pred_c_shape_matches_S1898_keeps_60pc_on_the_top_bucket': bool(pc),
                          'pred_d_controls': bool(pd)},
          'runtime_s': round(time.time() - t0, 1)}
    json.dump(r2, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({r2["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
