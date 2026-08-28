# CLASS RATIO SITE SWEEP -- is "MLP damage lands on tokens you cannot copy, attention damage lands
# everywhere" a real structural law, per site, on held-out rows?
#
# §1727 (circuit_audit v4) decomposed each circuit's constant-ablation damage over disjoint
# target-side token classes and found a clean contrast in the PER-TOKEN ratio:
#
#     induction nats/tok over novel nats/tok --  18-MLP stack 0.838   18-attention stack 1.002
#
# with the extremes at mlp16+17 (0.222) and mlp0 (0.386). Attention's damage is flat across classes;
# MLP damage concentrates on targets that are ABSENT from the context and so cannot be retrieved by
# routing -- exactly where a weight-resident answer is the only source.
#
# THAT CONTRAST WAS NOT REGISTERED IN ADVANCE. v4's pred_a tested the damage SHARE, which is
# dominated by class base rates (induction is 8.4% of scored tokens, so every circuit's induction
# share sits near 8.4% and the spread was 6.4pp against a 10pp bar -- pred_a FAILED as written).
# The per-token ratio is the base-rate-free quantity and it is the one that separated, but it
# separated in a statistic I looked at AFTER the run. This sweep is the prospective test it needs:
# every site individually, both eval sets, with bootstrap intervals, and the held-out set carrying
# the claim.
#
# WHAT IS NEW HERE vs v4. v4 scored SETS of sites named by registry entries. This scores all 36
# sites INDIVIDUALLY, so the contrast can be attributed to component kind rather than to which
# groups happened to be named, and depth is available as a competing explanation to test against.
#
# Registered predictions, TWO-SIDED per LESSONS 31:
#   pred_a THE CONTRAST REPLICATES HELD OUT: on skip11000, the median induction/novel per-token
#          removal ratio over the 18 MLP sites is BELOW the median over the 18 attention sites.
#          If FALSE, §1727's contrast was a skip7000 artifact and does not survive fresh documents.
#   pred_b IT IS A KIND EFFECT, NOT A DEPTH EFFECT: at matched depth, mlp_L has a lower ratio than
#          attn_L for at least 12 of the 18 layers on skip11000. If FALSE, the stack medians differ
#          because the two kinds' ratios vary with depth in different places, not because the kinds
#          differ, and the law should be stated per depth instead.
#   pred_c THE SEPARATION IS RESOLVED, NOT NOISE: the 95% bootstrap intervals (2000 paired
#          row-level draws) on the two stack medians do not overlap on skip11000. If FALSE the
#          contrast is real in point estimate and unresolved at this row count, and gets reported
#          that way rather than as a law.
#   pred_d CONTROLS: the skip7000 baseline CE reproduces 3.29205 (§1695) within 1e-3, the skip11000
#          baseline reproduces 3.09711 (§1701) within 1e-2, per-class token counts sum exactly to
#          the scored count on both sets, and the 18-MLP and 18-attention JOINT ratios reproduce
#          §1727's 0.838 and 1.002 within 0.02 on skip7000.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256; NB = 2000
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ops/class_ratio_site_sweep_results.json'
EVAL_SETS = [('skip7000', PT + '.rowcache/fineweb_n192_skip7000.pt', 3.29205, 1e-3),
             ('skip11000', PT + '.rowcache/fineweb_n192_skip11000.pt', 3.09711, 1e-2)]
CONSTS = PT + 'opt_ablation_consts_all.pt'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip80.pt'
H = m.transformer.h
S1727_JOINT = {'mlp': 0.838, 'attn': 1.002}
CLASSES = ('induction', 'repeat', 'novel')
STATE = {}
COV = {}


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def const_hook(c):
    def hook(mod, args, out):
        y = out[0] if isinstance(out, tuple) else out
        sub = c.to(y.dtype).expand_as(y)
        return (sub,) + tuple(out[1:]) if isinstance(out, tuple) else sub
    return hook


def mod_of(kind, L):
    return H[L].mlp if kind == 'mlp' else H[L].attn


@torch.no_grad()
def token_classes(idx, tg):
    """Disjoint target-side classes, identical to circuit_audit v4 (§1727)."""
    B, L = idx.shape
    ar = torch.arange(L, device=idx.device)
    causal = ar.unsqueeze(1) < ar.unsqueeze(0)
    causal_incl = ar.unsqueeze(1) >= ar.unsqueeze(0)
    nxt = torch.cat([idx[:, 1:], torch.full((B, 1), -1, device=idx.device, dtype=idx.dtype)], 1)
    prev_match = idx.unsqueeze(1) == idx.unsqueeze(2)
    copy_match = nxt.unsqueeze(1) == tg.unsqueeze(2)
    induction = (prev_match & copy_match & causal.unsqueeze(0)).any(2)
    seen_tg = ((idx.unsqueeze(1) == tg.unsqueeze(2)) & causal_incl.unsqueeze(0)).any(2)
    return {'induction': induction, 'repeat': seen_tg & ~induction,
            'novel': ~seen_tg & ~induction}


@torch.no_grad()
def per_row(rows, hooks=()):
    """Per-ROW class loss sums and counts, so the bootstrap can resample rows (§1701 convention)."""
    n = rows.shape[0]
    s = {c: torch.zeros(n) for c in CLASSES}
    k = {c: torch.zeros(n) for c in CLASSES}
    hs = list(hooks)
    try:
        for i in range(0, n, 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous()
            STATE['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            tg = bb[:, 1:].to(DEV)
            e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                reduction='none').reshape(tg.shape)[:, 64:]
            cls = token_classes(idx, tg)
            # SAME SCORED POPULATION as circuit_audit v1-v4: only positions whose input token was
            # seen in the fit rows. Dropping this scored every position and moved the baseline CE
            # from 3.29205 to 3.13704 -- caught by the pred_d baseline assert on the first run,
            # which is the whole reason it is a hard assert (LESSONS 29).
            cov = COV['seen'][idx[:, 64:]]
            for c in CLASSES:
                msk = cls[c][:, 64:] & cov
                s[c][i:i + bb.shape[0]] = (e * msk).sum(1).cpu()
                k[c][i:i + bb.shape[0]] = msk.sum(1).float().cpu()
    finally:
        for h in hs:
            h.remove()
    return s, k


def ratio(cs, ck, ls, lk, sel=None):
    """(induction nats/tok of damage) / (novel nats/tok of damage), over the selected rows."""
    def d(c):
        a = cs[c] if sel is None else cs[c][sel]
        b = ls[c] if sel is None else ls[c][sel]
        nn = ck[c] if sel is None else ck[c][sel]
        n = float(nn.sum())
        return (float(a.sum()) - float(b.sum())) / n if n else float('nan')
    den = d('novel')
    return d('induction') / den if abs(den) > 1e-9 else float('nan')


@torch.no_grad()
def main():
    t0 = time.time()
    K = torch.load(CONSTS, map_location='cpu')
    fit = load(FIT_ROWS)
    seen = torch.zeros(50257, dtype=torch.bool)
    seen[fit[:, :T].reshape(-1).long()] = True
    COV['seen'] = seen.to(DEV)
    print(f'  coverage: {int(seen.sum())} of 50257 token ids appear in the fit rows; only positions '
          f'with a covered input token are scored, matching circuit_audit v1-v4', flush=True)
    sites = [(k, L) for k in ('mlp', 'attn') for L in range(18)]
    out = {}
    print(f'CLASS RATIO SITE SWEEP | all 36 sites individually | induction/novel per-token removal '
          f'| {NB} row-level bootstrap draws', flush=True)

    for ename, epath, ce_ref, tol in EVAL_SETS:
        ev = load(epath)
        ls, lk = per_row(ev)
        base = sum(float(ls[c].sum()) for c in CLASSES) / sum(float(lk[c].sum()) for c in CLASSES)
        ntok = int(sum(float(lk[c].sum()) for c in CLASSES))
        assert abs(base - ce_ref) <= tol, f'{ename} baseline CE {base:.5f} != {ce_ref} (+/-{tol})'
        print(f'\n  {ename}: baseline CE {base:.5f} (ref {ce_ref}) | scored {ntok} tokens | '
              f'induction {int(lk["induction"].sum())} repeat {int(lk["repeat"].sum())} '
              f'novel {int(lk["novel"].sum())}', flush=True)

        rows = {}
        store = {}
        for st in sites:
            cs, ck = per_row(ev, hooks=[mod_of(*st).register_forward_hook(
                const_hook(K[f'{st[0]}{st[1]}'].to(DEV).float()))])
            store[st] = (cs, ck)
            rows[f'{st[0]}{st[1]}'] = {'ratio': round(ratio(cs, ck, ls, lk), 5),
                                       'removal_nats': round(
                                           sum(float(cs[c].sum()) - float(ls[c].sum()) for c in CLASSES)
                                           / ntok, 5)}
        # the joint stacks, for the §1727 control
        joint = {}
        for kind in ('mlp', 'attn'):
            ks = [s for s in sites if s[0] == kind]
            cs, ck = per_row(ev, hooks=[mod_of(*s).register_forward_hook(
                const_hook(K[f'{s[0]}{s[1]}'].to(DEV).float())) for s in ks])
            joint[kind] = round(ratio(cs, ck, ls, lk), 5)

        g = torch.Generator().manual_seed(1727)
        nrow = ev.shape[0]
        med = {'mlp': [], 'attn': []}
        for _ in range(NB):
            sel = torch.randint(0, nrow, (nrow,), generator=g)
            for kind in ('mlp', 'attn'):
                rr = sorted(ratio(*store[s], ls, lk, sel) for s in sites if s[0] == kind)
                med[kind].append(0.5 * (rr[8] + rr[9]))
        ci = {kind: (round(sorted(v)[int(0.025 * NB)], 4), round(sorted(v)[int(0.975 * NB)], 4))
              for kind, v in med.items()}
        pm = sorted(rows[f'mlp{L}']['ratio'] for L in range(18))
        pat = sorted(rows[f'attn{L}']['ratio'] for L in range(18))
        mmed, amed = 0.5 * (pm[8] + pm[9]), 0.5 * (pat[8] + pat[9])
        matched = sum(1 for L in range(18)
                      if rows[f'mlp{L}']['ratio'] < rows[f'attn{L}']['ratio'])

        print(f'    {"layer":5s} {"mlp ratio":>10s} {"attn ratio":>11s}   mlp<attn', flush=True)
        for L in range(18):
            a, b = rows[f'mlp{L}']['ratio'], rows[f'attn{L}']['ratio']
            print(f'    {L:5d} {a:10.3f} {b:11.3f}   {"yes" if a < b else "no"}', flush=True)
        print(f'    stack medians: mlp {mmed:.3f} 95% CI {ci["mlp"]} | attn {amed:.3f} 95% CI '
              f'{ci["attn"]}', flush=True)
        print(f'    joint-stack ratios (§1727 control): mlp {joint["mlp"]:.3f} attn '
              f'{joint["attn"]:.3f}', flush=True)
        out[ename] = {'baseline_ce': round(base, 5), 'scored_tokens': ntok,
                      'class_counts': {c: int(lk[c].sum()) for c in CLASSES},
                      'sites': rows, 'joint_stack_ratio': joint,
                      'median_ratio': {'mlp': round(mmed, 4), 'attn': round(amed, 4)},
                      'median_ratio_ci95': ci, 'matched_depth_mlp_below_attn': matched}
        del ev, store
        torch.cuda.empty_cache()

    ho = out['skip11000']
    pa = ho['median_ratio']['mlp'] < ho['median_ratio']['attn']
    pb = ho['matched_depth_mlp_below_attn'] >= 12
    pc = ho['median_ratio_ci95']['mlp'][1] < ho['median_ratio_ci95']['attn'][0]
    ref = out['skip7000']
    pd = all(abs(ref['joint_stack_ratio'][k] - v) <= 0.02 for k, v in S1727_JOINT.items())

    print(f'\n  HELD OUT (skip11000): mlp median {ho["median_ratio"]["mlp"]:.3f} vs attn '
          f'{ho["median_ratio"]["attn"]:.3f} -> contrast replicates {pa}', flush=True)
    print(f'  matched depth: mlp below attn at {ho["matched_depth_mlp_below_attn"]}/18 layers '
          f'-> kind not depth {pb}', flush=True)
    print(f'  CIs {ho["median_ratio_ci95"]["mlp"]} vs {ho["median_ratio_ci95"]["attn"]} '
          f'-> resolved {pc}', flush=True)
    print(f'  skip7000 joint ratios {ref["joint_stack_ratio"]} vs §1727 {S1727_JOINT} '
          f'-> control {pd}', flush=True)

    res = {'config': {'eval_sets': [e[0] for e in EVAL_SETS], 'bootstrap_draws': NB,
                      'bootstrap': 'ROW-level clusters over the 192 eval rows; the rowcache carries '
                                   'no document ids, so this is NOT document-clustered (§1701)',
                      'ratio': 'per-token constant-ablation damage on `induction` targets divided by '
                               'the same on `novel` targets; base-rate free, unlike the damage SHARE '
                               'that v4 pred_a tested and that failed',
                      'classes': 'identical definition to circuit_audit v4 (§1727)'},
           'results': out,
           'predictions': {'pred_a_contrast_replicates_heldout': bool(pa),
                           'pred_b_kind_not_depth': bool(pb),
                           'pred_c_cis_disjoint': bool(pc),
                           'pred_d_controls': bool(pd)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1)
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc} | pred_d {pd}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
