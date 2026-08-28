# DOES A ONE-SITE MAGNITUDE CORRECTION HELP THE SETTLED PROGRAM?  -- §1809's closing question.
#
# §1807 found that rescaling EVERY site to its live output norm makes the fully compiled program WORSE:
# 0.0% -> -5.4 / -5.1 / -6.1% of the gap. §1809 then found that one site's rescale -- mlp4, whose row is
# ten times what mlp4 emits -- carries 78-82% of a 62-point swing in a bottom-up arm, while all
# seventeen other MLPs together carry 0%.
#
# Those two facts are about different objects. The second is about a partially compiled stack; the first
# is about the fully compiled settled program of §1786, which is the baseline for every accuracy figure
# from §1788 onward. If one site's rescale improves THAT, it is a free correction to a certified object.
# If none does, §1807's all-sites regression is not masking a per-site win and the settled row
# magnitudes are locally optimal, which closes the magnitude line.
#
# ROLES. skip7000, skip11000, skip1200; settled program at full coverage; 36 single-site arms, the
# all-sites arm, and the two endpoints. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a A SINGLE SITE HELPS: some one-site rescale improves overall top-1 by at least 0.5 percentage
#          points, at every role. If FALSE, no per-site magnitude correction helps the settled program
#          and the row magnitudes should be treated as locally optimal at full compile -- which closes
#          the magnitude line that began at §1804.
#   pred_b AND IT IS mlp4: the best site is ('mlp', 4) at every role. If FALSE the bottom-up and
#          fully-compiled settings disagree about which site's magnitude is wrong, which would mean
#          "this row is mis-scaled" is a property of where the row SITS rather than of the row -- and no
#          site-level correction generalises across compile configurations.
#   pred_c THE CORRECTIONS DO NOT COMPOSE: the sum of the 36 individual changes differs from the
#          all-sites change by at least 3 percentage points, at every role. §1808 found a sharp
#          non-additivity (at L3, mlp-only +9.0 and attn-only -1.7 combining to -0.1) and §1809 found a
#          super-additive one (+49.0 and +0.3 combining to +63.2). If FALSE they do compose here and a
#          greedy per-site correction is worth building; if TRUE this scan is diagnostic only and must
#          not be read as a recipe.
#   pred_d CONTROLS, cross-run per LESSON 42: the unscaled arm reproduces §1789's PUBLISHED 0.1355 /
#          0.1425 / 0.1364 and the live arm 0.3932 / 0.4235 / 0.3888 within 0.001; the all-sites scaled
#          arm reproduces §1807's PUBLISHED -5.4 / -5.1 / -6.1% of gap within 2 points; coverage 5419.
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
OUT = PT + 'ops/single_site_rescale_scan_results.json'
PROBE_LS = ()
KIND_LS = ()
S1807_ALLSCALED = {'skip7000': -0.054, 'skip11000': -0.051, 'skip1200': -0.061}
S1807_SCALED_L5 = {'skip7000': 0.189, 'skip11000': 0.177, 'skip1200': 0.183}
S1808_SWING = {'skip7000': {3: -0.001, 5: 0.628}, 'skip11000': {3: 0.001, 5: 0.609},
               'skip1200': {3: -0.001, 5: 0.647}}
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
    print(f'SINGLE-SITE RESCALE SCAN | fully compiled program, one of 36 sites rescaled at a '
          f'time | '
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
    # the single-site test: mlp4 ALONE rescaled, every other row untouched
    only4 = {st: row_hook(fr[st], scale[st] if st == ('mlp', 4) else 1.0) for st in sites}
    # and its complement: every MLP EXCEPT mlp4
    not4 = {st: row_hook(fr[st], scale[st] if (st[0] == 'mlp' and st[1] != 4) else 1.0)
            for st in sites}
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
    for st in sites:
        one = {s2: row_hook(fr[s2], scale[s2] if s2 == st else 1.0) for s2 in sites}
        run(f'only_{st[0]}{st[1]}', sites, one)
        del one

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
    single = {e: {st: d(e, f'only_{st[0]}{st[1]}') for st in sites} for e in roles}
    best = {e: max(sites, key=lambda st: single[e][st]) for e in roles}
    tot = {e: sum(single[e][st] for st in sites) for e in roles}
    alls = {e: d(e, 'all_sub_scaled') for e in roles}
    pa = all(single[e][best[e]] >= 0.005 for e in roles)
    pb = all(best[e] == ('mlp', 4) for e in roles)
    pc = all(abs(tot[e] - alls[e]) >= 0.03 for e in roles)
    pd = (all(abs(res[e]['all_substituted']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['live_model']['top1'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(alls[e] / gap[e] - S1807_ALLSCALED[e]) <= 0.02 for e in roles)
          and NFULL == NCOV)

    print('\n  change in overall top-1 from rescaling ONE site, fully compiled program (pp):',
          flush=True)
    for k in ('attn', 'mlp'):
        for e in roles:
            print(f'    {k:4s} {e:10s} ' + ' '.join(
                f'L{L}:{100*single[e][(k, L)]:+5.2f}' for L in range(18)), flush=True)
    print(f'\n  some single-site rescale gains >=0.5pp -> {pa}  ' + '  '.join(
        f'{e} best {best[e][0]}{best[e][1]} {100*single[e][best[e]]:+.2f}pp' for e in roles),
        flush=True)
    print(f'  and the best site is mlp4 -> {pb}  ' + '  '.join(
        f'{e} {best[e][0]}{best[e][1]}' for e in roles), flush=True)
    print(f'  the 36 singles do NOT compose to the all-sites change -> {pc}  ' + '  '.join(
        f'{e} sum {100*tot[e]:+.2f}pp vs all {100*alls[e]:+.2f}pp' for e in roles), flush=True)
    print(f'  all-substituted and live reproduce §1789, coverage {NFULL} -> control {pd}',
          flush=True)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)

    json.dump({'run': 'l5_cliff_probe',
               'norm_live_vs_row': {str(L): list(norm[L]) for L in norm},
               'ratio': {str(L): ratio[L] for L in ratio},
               'results': {e: {f: {k: (round(v, 6) if isinstance(v, float) else v)
                                   for k, v in c.items()} for f, c in d.items()}
                           for e, d in res.items()},
               'scale': {f'{k}{L}': scale[(k, L)] for k in ('attn', 'mlp') for L in range(18)},
               'single_site_pp': {e: {f'{st[0]}{st[1]}': single[e][st] for st in sites}
                                  for e in roles},
               'best_site': {e: f'{best[e][0]}{best[e][1]}' for e in roles},
               'sum_singles': tot, 'all_scaled_pp': alls,
               'predictions': {'pred_a_some_single_site_gains': bool(pa),
                               'pred_b_best_is_mlp4': bool(pb),
                               'pred_c_singles_do_not_compose': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
