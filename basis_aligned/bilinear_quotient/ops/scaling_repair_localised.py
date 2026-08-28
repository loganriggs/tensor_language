# WHERE DOES THE SCALING REPAIR COME FROM?  -- getting more data on §1807's one-point anomaly.
#
# §1807 found that norm-matching the compiled rows does essentially nothing to the bottom-up poisoning:
# at L3 the scaled arm is -45.0 / -43.0 / -47.0% against raw -44.8 / -43.2 / -46.9%. But at L5 it swung
# from -43.9% to +18.9% -- sixty-two points, on all three roles. I declined to theorise about that on
# one depth, which is the right response to one data point only if the next step is to get more.
#
# This samples the swing at ALL eighteen bottom-up depths, and decomposes it at the depths around it by
# scaling only the attention rows or only the MLP rows. The motivating asymmetry is in the calibration
# factors themselves: attention rows are uniformly 2.13x-7.07x too small, while MLP rows are 0.10x-2.33x
# and mostly too LARGE -- mlp4's row is ten times what mlp4 emits. Those are different corrections and
# they need not act the same way.
#
# It also fixes an efficiency defect of §1807: that run materialised a SECOND [50257, D] bank per site
# to hold the scaled rows and peaked at 26.4 GiB. The scale is now applied at hook time, which is
# identical arithmetic on the gathered rows at a thirty-sixth of the memory.
#
# ROLES. skip7000, skip11000, skip1200; settled program of §1786 at full coverage. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a THE REPAIR IS LOCAL, NOT GENERAL: at most three of the eighteen depths show a swing of at
#          least 10 points of gap, at every role. If FALSE the repair is broad and §1807's "norm-matching
#          does not repair the poisoning" was drawn from an unluckily chosen depth -- which would mean
#          my structural conclusion there rests on L3 alone and must be softened.
#   pred_b MLP SCALING CARRIES IT: scaling only the MLP rows reproduces at least 40% of the L5 swing, at
#          every role. If FALSE the MLP correction is not what moves it, and given pred_c the swing
#          would be a JOINT effect of both corrections that neither produces alone -- a third outcome
#          my two binary predicates would otherwise blur (LESSON 44), and one that would say the
#          repair is not attributable to a module kind at all.
#   pred_c ATTENTION SCALING DOES NOT: scaling only the attention rows reproduces under 20% of the L5
#          swing, at every role. Scored separately from pred_b for exactly the reason above.
#   pred_d CONTROLS, cross-run per LESSON 42: the endpoints reproduce §1789's PUBLISHED 0.1355 / 0.1425 /
#          0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001; the RAW bottom-up arms at L = 0, 3, 5
#          reproduce §1806's PUBLISHED gap fractions within 2 points; and the FULLY SCALED L5 arm
#          reproduces §1807's PUBLISHED +18.9 / +17.7 / +18.3% within 2 points, which is the anomaly
#          this run exists to explain and must therefore be rebuilt before it is explained; coverage
#          5419 of 50257.
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
OUT = PT + 'ops/scaling_repair_localised_results.json'
PROBE_LS = tuple(range(18))
KIND_LS = (3, 4, 5, 6)   # where the swing was, for the attn-only / mlp-only decomposition
S1807_SCALED_L5 = {'skip7000': 0.189, 'skip11000': 0.177, 'skip1200': 0.183}
S1806_BOTUP = {'skip7000': {0: 0.374, 3: -0.448, 5: -0.439},
               'skip11000': {0: 0.396, 3: -0.432, 5: -0.423},
               'skip1200': {0: 0.381, 3: -0.469, 5: -0.455}}
S1802_ONLY = {'skip7000': {4: -0.0003, 5: -0.1254, 6: -0.1115, 13: -0.0175},
              'skip11000': {4: -0.0006, 5: -0.1311, 6: -0.1198, 13: -0.0173},
              'skip1200': {4: -0.0003, 5: -0.1267, 6: -0.1160, 13: -0.0203}}
S1789_GAP = {'skip7000': 0.3932 - 0.1355, 'skip11000': 0.4235 - 0.1425,
             'skip1200': 0.3888 - 0.1364}
S1795_BG = {'skip7000': 0.12440, 'skip11000': 0.12880, 'skip1200': 0.12250}
TAUS = (10 ** 9, 20.0, 12.0, 8.0, 5.0, 3.0, 2.0, 1.0)  # DEFER when the bigram's LOO count >= tau
S1796_UNION = {'skip7000': 0.45966, 'skip11000': 0.48224, 'skip1200': 0.47517}
PICK_ROLE = 'skip7000'
ALPHA = 0.01
S1767_FITBIGRAM_CE = {'skip7000': 7.88804, 'skip11000': 7.90729}
BUCKETS = ((0, 0), (1, 4), (5, 24), (25, 124), (125, 10 ** 9))
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711),
             ('skip1200', PT + '.rowcache/fineweb_n96_skip1200.pt', 3.40277)]
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
NCOV = 5419
S1789_PROG = {'skip7000': 0.1355, 'skip11000': 0.1425, 'skip1200': 0.1364}
S1789_LIVE = {'skip7000': 0.3932, 'skip11000': 0.4235, 'skip1200': 0.3888}
S1790_LOOBG = {'skip7000': 0.1597, 'skip11000': 0.1663, 'skip1200': 0.1800}
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
    §1807 held a raw and a scaled bank and peaked at 26.4 GiB for no reason. One multiply on the
    gathered rows is identical arithmetic at a thirty-sixth of the memory."""
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
def bigram_ce(rows, lp_table):
    """Covered-position CE of the fit-row bigram, to control that it is §1767's object."""
    tot, cnt = 0.0, 0
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV)[:, 64:]
        tg = bb[:, 1:].to(DEV)[:, 64:]
        cov = COV['seen'][idx]
        r = COV['idmap'][idx].clamp(min=0)
        lp = lp_table[r].gather(-1, tg.unsqueeze(-1)).squeeze(-1)
        tot += float((-lp.double())[cov].sum()); cnt += int(cov.sum())
    return tot / cnt


@torch.no_grad()
def evaluate(rows, hooks, keep_mask):
    """Top-1 overall, on the head, and restricted to positions whose CURRENT token is still covered.

    That last slice is a KNOWN-ANSWER control (LESSON 34). A covered token's table is built from its
    own length-1 forward and the program is position-wise (§1765), so removing OTHER tokens from the
    covered set cannot change what happens at a position that kept its own table. Those numbers must
    be identical across every coverage fraction, not merely close."""
    a = {'n': 0, 'hit': 0, 'head_n': 0, 'head_hit': 0, 'kept_n': 0, 'kept_hit': 0}
    for i in range(0, rows.shape[0], 8):
        bb = rows[i:i + 8]
        idx = bb[:, :-1].to(DEV).contiguous()
        tg = bb[:, 1:].to(DEV)[:, 64:]
        h = forward_logits(idx, hooks)[:, 64:].argmax(-1) == tg
        hd = COV['freq'][tg] >= 125
        kp = keep_mask[idx[:, 64:]]
        a['n'] += int(tg.numel()); a['hit'] += int(h.sum())
        a['head_n'] += int(hd.sum()); a['head_hit'] += int(h[hd].sum())
        a['kept_n'] += int(kp.sum()); a['kept_hit'] += int(h[kp].sum())
    return {'n': a['n'], 'top1': a['hit'] / max(a['n'], 1),
            'head_n': a['head_n'], 'top1_head': a['head_hit'] / max(a['head_n'], 1),
            'kept_n': a['kept_n'], 'top1_kept': a['kept_hit'] / max(a['kept_n'], 1)}


def probe_hook(full_rows, store, key):
    """Substitute the row as usual, but record ||live module output|| and ||row|| first.

    This runs inside the FULLY COMPILED stream, so the recorded live output is what the real module
    would emit given compiled inputs -- exactly the quantity the L5/L6 cliff is about."""
    inner = row_hook(full_rows)

    def hook(mod, args, out):
        o = (out[0] if isinstance(out, tuple) else out).detach().float()
        r = inner(mod, args, out)
        rr = (r[0] if isinstance(r, tuple) else r).detach().float()
        s = store.setdefault(key, [0.0, 0.0, 0])
        s[0] += float(o[:, 64:].norm(dim=-1).sum())
        s[1] += float(rr[:, 64:].norm(dim=-1).sum())
        s[2] += int(o[:, 64:].shape[0] * o[:, 64:].shape[1])
        return r
    return hook


def main():
    t0 = time.time()
    fit = load(FIT_ROWS)
    T = fit.shape[1] - 1
    full_seen = torch.zeros(V, dtype=torch.bool)
    full_seen[fit[:, :T].reshape(-1).long()] = True
    NFULL = int(full_seen.sum())
    assert NFULL == NCOV, f'coverage {NFULL} != {NCOV}'
    COV['freq'] = torch.bincount(fit[:, 1:T + 1].reshape(-1).long(), minlength=V).to(DEV)
    all_toks = full_seen.nonzero(as_tuple=True)[0]
    # NESTED subsets: one fixed permutation, prefixes of it. 0.125 subset of 0.25 subset of 0.5 ...
    g = torch.Generator().manual_seed(0)
    perm = all_toks[torch.randperm(NFULL, generator=g)]
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    print(f'SCALING REPAIR, LOCALISED | all 18 bottom-up depths x raw/scaled, plus attn-only and '
          f'mlp-only at {KIND_LS} | '
          f'settled program (context-free tables + output-NN fallback + rank-{MAP_RANK} map) | '
          f'DISCOVERY ONLY', flush=True)

    def build(n):
        tk = perm[:n].sort().values.to(DEV)
        seen = torch.zeros(V, dtype=torch.bool, device=DEV)
        seen[tk] = True
        unc = (~seen).nonzero(as_tuple=True)[0]
        # the settled fallback: output-NN neighbour (§1780/§1781), rebuilt over THIS covered set
        lpc = torch.zeros(n, W, device=DEV)
        for i in range(0, n, 256):
            t = tk[i:i + 256].unsqueeze(1)
            lpc[i:i + t.shape[0]] = torch.log_softmax(forward_logits(t)[:, 0].float(), -1)
        pcn = torch.softmax(lpc, -1)
        pcn = (pcn / pcn.norm(dim=-1, keepdim=True).clamp_min(1e-9)).half()
        del lpc
        nnrow = torch.zeros(V, dtype=torch.long, device=DEV)
        nnrow[tk] = torch.arange(n, device=DEV)
        for s0 in range(0, unc.numel(), 512):
            u = unc[s0:s0 + 512]
            p = torch.softmax(forward_logits(u.unsqueeze(1))[:, 0].float(), -1)
            p = p / p.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            nnrow[u] = (p.half() @ pcn.T).float().argmax(-1)
        del pcn
        torch.cuda.empty_cache()
        tables = {st: torch.zeros(n, D, device=DEV) for st in sites}
        cap = {}

        def mk(st):
            def hook(mod, args, out):
                cap[st] = (out[0] if isinstance(out, tuple) else out)[:, 0].float()
                return None
            return hook
        for i in range(0, n, 256):
            t = tk[i:i + 256].unsqueeze(1)
            forward_logits(t, [(st, mk(st)) for st in sites])
            for st in sites:
                tables[st][i:i + t.shape[0]] = cap[st]
        # the learned embedding->row map, REFITTED inside this covered set (§1785)
        Ecov = m.transformer.wte.weight.detach()[tk].float().double()
        A = Ecov.T @ Ecov + RIDGE * torch.eye(D, device=DEV, dtype=torch.float64) * (n / D)
        Eunc = m.transformer.wte.weight.detach()[unc].float().double()
        out = {}
        for st in sites:
            Ws = torch.linalg.solve(A, Ecov.T @ tables[st].double())
            U, S, Vh = torch.linalg.svd(Ws, full_matrices=False)
            mp = (U[:, :MAP_RANK] * S[:MAP_RANK]) @ Vh[:MAP_RANK]
            fr = torch.zeros(V, D, device=DEV)
            fr[tk] = tables[st]
            fr[unc] = (Eunc @ mp).float()
            out[st] = fr
        del tables, Ecov, Eunc, A
        torch.cuda.empty_cache()
        return out, seen, n

    # the KEPT slice is defined by the SMALLEST coverage set, so it is the same positions at every
    # fraction -- that is what makes the known-answer control meaningful.
    # coverage is fixed at full here, so the 'kept' column is simply accuracy on COVERED
    # positions -- reported for context, not used as a control (there is no coverage variation
    # for a known-answer check to bite on).
    small = perm[:NFULL].to(DEV)
    keep_mask = torch.zeros(V, dtype=torch.bool, device=DEV)
    keep_mask[small] = True

    fr, seen, _ = build(NFULL)

    # ---- per-site NORM CALIBRATION, measured in the FULLY LIVE model.
    # §1804 found the substituted rows are systematically far smaller than what the live modules emit
    # (ratio 2.71 to 152.62 across the attention layers). §1806 showed a compiled layer beneath live
    # layers is catastrophic. If that poisoning is a SCALE mismatch, matching the mean output norm
    # should repair most of it; if it is a direction/content mismatch, scaling will not help.
    live_norm, row_norm = {}, {}

    def measure(st, store):
        def hook(mod, args, out):
            o = (out[0] if isinstance(out, tuple) else out).detach().float()
            s = store.setdefault(st, [0.0, 0])
            s[0] += float(o[:, 64:].norm(dim=-1).sum())
            s[1] += int(o[:, 64:].shape[0] * o[:, 64:].shape[1])
            return None
        return hook
    evs0 = load(EVAL_SETS[0][1])
    lstore = {}
    evaluate(evs0, [(st, measure(st, lstore)) for st in sites], keep_mask)
    for st in sites:
        live_norm[st] = lstore[st][0] / max(lstore[st][1], 1)
    # The row norms must be averaged over THE SAME POSITIONS, not over the 50,257 vocabulary rows:
    # eval positions are frequency-weighted and the vocabulary is not, so a flat row mean would
    # calibrate against a distribution the model never sees. Measured with the substituting hook in
    # place, exactly as §1804's probe did.
    rstore = {}
    evaluate(evs0, [(st, probe_hook(fr[st], rstore, st)) for st in sites], keep_mask)
    del evs0
    scale = {}
    for st in sites:
        rn = rstore[st][1] / max(rstore[st][2], 1)
        row_norm[st] = rn
        scale[st] = live_norm[st] / max(rn, 1e-9)
    print('\n  per-site norm calibration (live mean / row mean), measured on the LIVE model:',
          flush=True)
    for k in ('attn', 'mlp'):
        print(f'    {k:4s} ' + ' '.join(f'L{L}:{scale[(k, L)]:6.2f}' for L in range(18)), flush=True)
    allhooks = {st: row_hook(fr[st]) for st in sites}
    scaledhooks = {st: row_hook(fr[st], scale[st]) for st in sites}
    attnhooks = {st: row_hook(fr[st], scale[st] if st[0] == 'attn' else 1.0) for st in sites}
    mlphooks = {st: row_hook(fr[st], scale[st] if st[0] == 'mlp' else 1.0) for st in sites}
    evs = {e: load(p) for e, p, _ in EVAL_SETS}
    res = {}

    def run(label, hooked, hooks_src=None):
        src = hooks_src if hooks_src is not None else allhooks
        hs = [(st, src[st]) for st in hooked]
        for ename in evs:
            c = evaluate(evs[ename], hs, keep_mask)
            res.setdefault(ename, {})[label] = c
        print(f'  {label:20s} ' + '  '.join(
            f'{e} {res[e][label]["top1"]:6.2%}' for e in evs), flush=True)

    print(f'\n  arms  ({time.time() - t0:.0f}s)', flush=True)
    run('all_substituted', sites)
    run('all_sub_scaled', sites, scaledhooks)
    run('live_model', [])
    for L in PROBE_LS:
        run(f'L{L}_botup', [st for st in sites if st[1] <= L])
        run(f'L{L}_botup_scaled', [st for st in sites if st[1] <= L], scaledhooks)
    for L in KIND_LS:
        run(f'L{L}_botup_attnonly', [st for st in sites if st[1] <= L], attnhooks)
        run(f'L{L}_botup_mlponly', [st for st in sites if st[1] <= L], mlphooks)

    # ---- mechanism diagnostic: in the FULLY COMPILED stream, how big is what each live attention
    # module would have emitted, against the row that replaces it?  (LESSON 44: emit the quantity.)
    store = {}
    phooks = [(st, (probe_hook(fr[st], store, st[1]) if st[0] == 'attn' else allhooks[st]))
              for st in sites]
    evaluate(evs['skip7000'], phooks, keep_mask)
    norm = {L: (store[L][0] / max(store[L][2], 1), store[L][1] / max(store[L][2], 1))
            for L in sorted(store)}
    print('\n  mean ||live attn output|| vs ||substituted row||, in the compiled stream:', flush=True)
    for L in sorted(norm):
        print(f'    L{L:<2d} live {norm[L][0]:12.3f}  row {norm[L][1]:9.3f}  '
              f'ratio {norm[L][0] / max(norm[L][1], 1e-9):8.3f}', flush=True)
    del evs
    torch.cuda.empty_cache()

    roles = [e for e, _, _ in EVAL_SETS]
    base = {e: res[e]['all_substituted']['top1'] for e in roles}

    def d(e, label):
        return res[e][label]['top1'] - base[e]
    ratio = {L: norm[L][0] / max(norm[L][1], 1e-9) for L in norm}
    gap = {e: res[e]['live_model']['top1'] - base[e] for e in roles}
    bot = {e: {L: d(e, f'L{L}_botup') / gap[e] for L in PROBE_LS} for e in roles}
    bsc = {e: {L: d(e, f'L{L}_botup_scaled') / gap[e] for L in PROBE_LS} for e in roles}
    swing = {e: {L: bsc[e][L] - bot[e][L] for L in PROBE_LS} for e in roles}
    nbig = {e: sum(1 for L in PROBE_LS if swing[e][L] >= 0.10) for e in roles}
    att = {e: {L: d(e, f'L{L}_botup_attnonly') / gap[e] - bot[e][L] for L in KIND_LS}
           for e in roles}
    mlp = {e: {L: d(e, f'L{L}_botup_mlponly') / gap[e] - bot[e][L] for L in KIND_LS}
           for e in roles}
    pa = all(nbig[e] <= 3 for e in roles)
    pb = all(mlp[e][5] >= 0.40 * swing[e][5] for e in roles)
    pc = all(att[e][5] < 0.20 * swing[e][5] for e in roles)
    pd = (all(abs(res[e]['all_substituted']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['live_model']['top1'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(bot[e][L] - S1806_BOTUP[e][L]) <= 0.02
                  for e in roles for L in (0, 3, 5))
          and all(abs(bsc[e][5] - S1807_SCALED_L5[e]) <= 0.02 for e in roles)
          and NFULL == NCOV)

    print('\n  gap recovered, bottom-up: RAW vs NORM-MATCHED, and the swing', flush=True)
    for L in PROBE_LS:
        print(f'    L{L:<2d}  ' + '  '.join(
            f'{e} raw {bot[e][L]:7.1%} scaled {bsc[e][L]:7.1%} swing {100*swing[e][L]:+6.1f}'
            for e in roles), flush=True)
    print('\n  decomposition at the swing depths (change vs raw, points of gap):', flush=True)
    for L in KIND_LS:
        print(f'    L{L:<2d}  ' + '  '.join(
            f'{e} attn-only {100*att[e][L]:+6.1f} mlp-only {100*mlp[e][L]:+6.1f} '
            f'both {100*swing[e][L]:+6.1f}' for e in roles), flush=True)
    print(f'\n  the repair is confined to <=3 depths -> {pa}  ' + '  '.join(
        f'{e} {nbig[e]} depths' for e in roles), flush=True)
    print(f'  the MLP scaling carries it (>=40% of the L5 swing) -> {pb}  ' + '  '.join(
        f'{e} {mlp[e][5] / max(swing[e][5], 1e-9):.0%}' for e in roles), flush=True)
    print(f'  the attention scaling does not (<20%) -> {pc}  ' + '  '.join(
        f'{e} {att[e][5] / max(swing[e][5], 1e-9):.0%}' for e in roles), flush=True)
    print(f'  all-substituted and live reproduce §1789, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'l5_cliff_probe',
               'norm_live_vs_row': {str(L): list(norm[L]) for L in norm},
               'ratio': {str(L): ratio[L] for L in ratio},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'bottomup_raw': {e: {str(L): bot[e][L] for L in PROBE_LS} for e in roles},
               'bottomup_scaled': {e: {str(L): bsc[e][L] for L in PROBE_LS} for e in roles},
               'scale': {f'{k}{L}': scale[(k, L)] for k in ('attn', 'mlp') for L in range(18)},
               'swing': {e: {str(L): swing[e][L] for L in PROBE_LS} for e in roles},
               'attn_only': {e: {str(L): att[e][L] for L in KIND_LS} for e in roles},
               'mlp_only': {e: {str(L): mlp[e][L] for L in KIND_LS} for e in roles},
               'predictions': {'pred_a_repair_confined_to_3_depths': bool(pa),
                               'pred_b_mlp_scaling_carries_it': bool(pb),
                               'pred_c_attn_scaling_does_not': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
