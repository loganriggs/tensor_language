# IS IT mlp4?  -- the single-site test §1808 named rather than assumed.
#
# §1808 localised the 62-point bottom-up swing to exactly ONE depth of eighteen (L5) and to ONE module
# kind (MLP-only reproduces 101 / 102 / 102% of it; attention-only reproduces 0 / 0 / 0%). It then
# offered a hypothesis and refused to claim it: the compiled prefix at L5 is depths 0..5, the first to
# contain mlp4, whose calibration factor is 0.10 -- its row is TEN TIMES what mlp4 actually emits, the
# most extreme factor in the stack. The prefix at L3 does not contain it and shows no swing.
#
# That is one pattern across two depths, which LESSON 37 says is not a law. This tests it directly:
# rescale mlp4 and NOTHING ELSE, and separately every MLP EXCEPT mlp4.
#
# It also carries a placement control. If mlp4 is the cause, rescaling it must do nothing at L3, where
# it is not inside the compiled prefix at all -- a site that is not being substituted cannot be
# rescaled into mattering. That distinguishes "mlp4's row is the problem" from "any change to the MLP
# rows at this depth helps".
#
# ROLES. skip7000, skip11000, skip1200; settled program of §1786 at full coverage. DISCOVERY ONLY.
#
# Registered predictions, TWO-SIDED per LESSONS 31, margins per LESSON 40, read back per LESSON 39,
# failure branches enumerated per LESSON 44:
#   pred_a mlp4 ALONE CARRIES IT: rescaling only mlp4 reproduces at least 80% of the L5 swing (+62.8 /
#          +60.9 / +64.7 points of gap), at every role. If FALSE the swing is distributed across the
#          MLP rows in the prefix and §1808's hypothesis is wrong -- the extreme factor at mlp4 would
#          then be a coincidence of index rather than a cause, and the honest statement reverts to
#          "MLP-kind, depth L5" with no site attribution.
#   pred_b AND THE OTHER MLPs DO NOT: rescaling every MLP except mlp4 reproduces under 20% of the L5
#          swing, at every role. Scored separately because both could be large -- if mlp4 alone gives
#          80% and the others also give 60%, the corrections are super-additive and no single-site
#          story holds, which is a third outcome the two predicates would otherwise blur (LESSON 44).
#   pred_c PLACEMENT CONTROL: rescaling only mlp4 changes the L3 arm by less than 2 points of gap,
#          at every role. mlp4 is not inside the compiled prefix at L3, so if this moves, the hooks
#          are being applied where they should not be and nothing else in the run is readable.
#   pred_d CONTROLS, cross-run per LESSON 42: endpoints reproduce §1789's PUBLISHED 0.1355 / 0.1425 /
#          0.1364 and 0.3932 / 0.4235 / 0.3888 within 0.001; the raw bottom-up arms at L3 and L5
#          reproduce §1806's PUBLISHED figures within 2 points; the fully scaled L5 arm reproduces
#          §1807's PUBLISHED +18.9 / +17.7 / +18.3% within 2 points; and the swings at L3 and L5
#          reproduce §1808's PUBLISHED -0.1 / +62.8 (and the matching figures for the other roles)
#          within 2 points. Coverage 5419 of 50257.
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
OUT = PT + 'ops/mlp4_single_site_results.json'
PROBE_LS = (3, 5)
KIND_LS = (3, 5)
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
    print(f'MLP4 SINGLE SITE | L in {KIND_LS} x (raw / all / mlp / attn / only-mlp4 / '
          f'mlp-except-4) | '
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
    for L in PROBE_LS:
        run(f'L{L}_botup', [st for st in sites if st[1] <= L])
        run(f'L{L}_botup_scaled', [st for st in sites if st[1] <= L], scaledhooks)
    for L in KIND_LS:
        run(f'L{L}_botup_attnonly', [st for st in sites if st[1] <= L], attnhooks)
        run(f'L{L}_botup_mlponly', [st for st in sites if st[1] <= L], mlphooks)
        run(f'L{L}_botup_only4', [st for st in sites if st[1] <= L], only4)
        run(f'L{L}_botup_not4', [st for st in sites if st[1] <= L], not4)

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
    att = {e: {L: d(e, f'L{L}_botup_attnonly') / gap[e] - bot[e][L] for L in KIND_LS}
           for e in roles}
    mlp = {e: {L: d(e, f'L{L}_botup_mlponly') / gap[e] - bot[e][L] for L in KIND_LS}
           for e in roles}
    o4 = {e: {L: d(e, f'L{L}_botup_only4') / gap[e] - bot[e][L] for L in KIND_LS} for e in roles}
    n4 = {e: {L: d(e, f'L{L}_botup_not4') / gap[e] - bot[e][L] for L in KIND_LS} for e in roles}
    pa = all(o4[e][5] >= 0.80 * swing[e][5] for e in roles)
    pb = all(n4[e][5] < 0.20 * swing[e][5] for e in roles)
    pc = all(abs(o4[e][3]) < 0.02 for e in roles)
    pd = (all(abs(res[e]['all_substituted']['top1'] - S1789_PROG[e]) <= 0.001
              and abs(res[e]['live_model']['top1'] - S1789_LIVE[e]) <= 0.001 for e in roles)
          and all(abs(bot[e][L] - S1806_BOTUP[e][L]) <= 0.02 for e in roles for L in (3, 5))
          and all(abs(bsc[e][5] - S1807_SCALED_L5[e]) <= 0.02 for e in roles)
          and all(abs(swing[e][L] - S1808_SWING[e][L]) <= 0.02 for e in roles for L in (3, 5))
          and NFULL == NCOV)

    print('\n  change vs raw (points of gap), by which rows are rescaled:', flush=True)
    for L in KIND_LS:
        print(f'    L{L}  ' + '  '.join(
            f'{e}  all {100*swing[e][L]:+6.1f}  mlp {100*mlp[e][L]:+6.1f}  attn {100*att[e][L]:+6.1f}'
            f'  ONLY-mlp4 {100*o4[e][L]:+6.1f}  mlp-except-4 {100*n4[e][L]:+6.1f}'
            for e in roles), flush=True)
    print(f'\n  mlp4 ALONE reproduces >=80% of the L5 swing -> {pa}  ' + '  '.join(
        f'{e} {o4[e][5] / max(swing[e][5], 1e-9):.0%} ({100*o4[e][5]:+.1f} of '
        f'{100*swing[e][5]:+.1f})' for e in roles), flush=True)
    print(f'  every OTHER mlp together reproduces <20% -> {pb}  ' + '  '.join(
        f'{e} {n4[e][5] / max(swing[e][5], 1e-9):.0%}' for e in roles), flush=True)
    print(f'  and mlp4 alone does nothing at L3, where it is not in the prefix -> {pc}  '
          + '  '.join(f'{e} {100*o4[e][3]:+.2f}pp' for e in roles), flush=True)
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
               'only_mlp4': {e: {str(L): o4[e][L] for L in KIND_LS} for e in roles},
               'mlp_except_4': {e: {str(L): n4[e][L] for L in KIND_LS} for e in roles},
               'predictions': {'pred_a_mlp4_alone_carries_it': bool(pa),
                               'pred_b_other_mlps_do_not': bool(pb),
                               'pred_c_placement_control_at_L3': bool(pc),
                               'pred_d_controls': bool(pd)}},
              open(OUT, 'w'), indent=1)
    print(f'wrote {OUT} ({time.time() - t0:.1f}s)', flush=True)


if __name__ == '__main__':
    main()
