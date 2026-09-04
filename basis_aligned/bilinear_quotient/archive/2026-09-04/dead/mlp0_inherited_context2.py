# mlp0_inherited_context2: THE COVERAGE-CORRECTED VERSION — v1's headline was an
# artifact and this run carries an instrument check that would have caught it.
#
# v1 (§1659) measured mlp0's token-table ceiling normally (74.42%) and with attn0
# frozen at its optimal constant (49.37%), and reported a 25-point FALL. That number is
# not usable. With attn0 frozen, mlp0's input is `lambda0*x + lambda1*x0 + const` where
# x = x0 = rms_norm(wte(idx)) -- a PURE FUNCTION OF THE CURRENT TOKEN. A fully-covered
# per-token table would therefore be EXACT, ceiling ~100%. Observing 49.37% is
# impossible on structure alone.
#
# The cause: the table is fitted on 96 rows covering 6009 distinct tokens, and 8627 of
# 36864 eval positions -- 23.4% -- hold a token with no fitted entry and fall back to
# the position-weighted mean. That penalty bites HARDER in the frozen condition,
# because there the table could otherwise be exact, so it manufactures a spurious drop.
#
# FIX: score the ceiling on COVERED positions only, and report both. The covered-only
# frozen ceiling is now also an INSTRUMENT CHECK with a known answer -- it must come
# out near 1.0, because the quantity being tabulated is exactly token-determined there.
# If it does not, the measurement is broken in some further way and no reading of the
# normal condition is trustworthy either.
#
# Registered predictions:
#   pred_a INSTRUMENT CHECK, known answer: the covered-only ceiling under attn0-frozen
#          is >= 0.97. This is not a finding about the model -- it is the condition
#          under which the rest of the run means anything.
#   pred_b COVERAGE WAS THE ARTIFACT: the covered-only ceiling under attn0-frozen
#          exceeds the all-positions frozen ceiling by >= 30 percentage points.
#   pred_c THE REAL QUESTION, now measurable: on covered positions the NORMAL ceiling
#          is BELOW the frozen ceiling by >= 5 points -- i.e. live attn0 genuinely
#          injects context that a current-token table cannot reproduce, which is the
#          inherited-context hypothesis stated so it can fail.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_inherited_context2_results.json'
FIT_ROWS = PT + '.rowcache/fineweb_n96_skip1200.pt'
EVAL_ROWS = PT + '.rowcache/fineweb_n192_skip7000.pt'
CONSTS = PT + 'opt_ablation_consts_all.pt'
H = m.transformer.h
S1326_CEILING = 0.863


def load(p):
    r = torch.load(p, map_location='cpu')
    r = r['rows'] if isinstance(r, dict) else r
    return r[:, :T + 1].contiguous()


def attn0_freeze_hook(const):
    def hook(mod, args, out):
        if isinstance(out, tuple):
            y = out[0]
            return (const.to(y.dtype).expand_as(y),) + tuple(out[1:])
        return const.to(out.dtype).expand_as(out)
    return hook


@torch.no_grad()
def forward_collect(rows, freeze_attn0, const_a0, collect=True):
    """Run the model; optionally freeze attn0's write. Returns per-token sums for the
    mlp0 table, the position-weighted mean, and mlp0's output norm."""
    s = torch.zeros(50257, D, device=DEV)
    c = torch.zeros(50257, device=DEV)
    sq = {'n': 0.0, 'sum': 0.0}
    cap = {}
    hs = [H[0].mlp.register_forward_hook(lambda mo, a, o: cap.__setitem__('o', o.float()))]
    if freeze_attn0:
        hs.append(H[0].attn.register_forward_hook(attn0_freeze_hook(const_a0)))
    try:
        for i in range(0, rows.shape[0], 8):
            idx = rows[i:i + 8, :-1].to(DEV).contiguous()
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            o = cap['o'].reshape(-1, D)
            if collect:
                t = idx.reshape(-1)
                s.index_add_(0, t, o)
                c.index_add_(0, t, torch.ones_like(t, dtype=torch.float32))
            sq['sum'] += float((o ** 2).sum()); sq['n'] += o.shape[0]
    finally:
        for h in hs:
            h.remove()
    seen = c > 0
    gmean = s.sum(0) / c.sum()
    tbl = gmean.unsqueeze(0).repeat(50257, 1)          # unseen -> weighted mean (§1655 fix)
    tbl[seen] = s[seen] / c[seen].unsqueeze(1)
    return tbl, (sq['sum'] / max(sq['n'], 1)) ** 0.5, seen


@torch.no_grad()
def ce(rows, mode, freeze_attn0, const_a0, const_m0, tbl=None, seen_mask=None):
    """mode: 'live' | 'const' (optimal-constant ablate mlp0) | 'table'."""
    hs = []
    state = {}
    if freeze_attn0:
        hs.append(H[0].attn.register_forward_hook(attn0_freeze_hook(const_a0)))
    if mode == 'const':
        hs.append(H[0].mlp.register_forward_hook(
            lambda mo, a, o: const_m0.to(o.dtype).expand_as(o)))
    elif mode == 'table':
        hs.append(H[0].mlp.register_forward_hook(
            lambda mo, a, o: tbl[state['idx'].reshape(-1)].reshape(o.shape).to(o.dtype)))
    tot, n = 0.0, 0
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            state['idx'] = idx
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            e = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                reduction='none').reshape(tg.shape)[:, 64:]
            if seen_mask is None:
                tot += float(e.sum()); n += e.numel()
            else:
                cov = seen_mask.to(DEV)[idx[:, 64:]]
                tot += float(e[cov].sum()); n += int(cov.sum())
    finally:
        for h in hs:
            h.remove()
    return tot / max(n, 1)


@torch.no_grad()
def main():
    t0 = time.time()
    fit = load(FIT_ROWS); ev = load(EVAL_ROWS)
    K = torch.load(CONSTS, map_location='cpu')
    const_a0 = K['attn0'].to(DEV).float(); const_m0 = K['mlp0'].to(DEV).float()
    print(f'mlp0 INHERITED CONTEXT v2 (coverage-corrected) | optimal constants from opt_ablation_consts_all.pt '
          f'(attn0, mlp0) | fit skip1200, eval skip7000', flush=True)

    out = {}
    norms = {}
    for label, freeze in (('normal', False), ('attn0_frozen', True)):
        tbl, nrm, seen = forward_collect(fit, freeze, const_a0)
        norms[label] = nrm
        rec = {'mlp0_output_rms': round(nrm, 4)}
        for scope, mask in (('all_positions', None), ('covered_only', seen)):
            cl = ce(ev, 'live', freeze, const_a0, const_m0, None, mask)
            cc = ce(ev, 'const', freeze, const_a0, const_m0, None, mask)
            ct = ce(ev, 'table', freeze, const_a0, const_m0, tbl, mask)
            st = cc - cl
            ceil = (cc - ct) / st if st > 1e-6 else float('nan')
            rec[scope] = {'ce_live': round(cl, 5), 'ce_const': round(cc, 5),
                          'ce_table': round(ct, 5), 'stake': round(st, 5),
                          'ceiling': round(ceil, 5)}
            print(f'  {label:13s} {scope:14s} stake {st:7.4f} | CEILING {ceil:7.2%}', flush=True)
        out[label] = rec

    frz_cov = out['attn0_frozen']['covered_only']['ceiling']
    frz_all = out['attn0_frozen']['all_positions']['ceiling']
    nrm_cov = out['normal']['covered_only']['ceiling']
    pa = frz_cov >= 0.97
    pb = (frz_cov - frz_all) >= 0.30
    pc = (frz_cov - nrm_cov) >= 0.05

    print(f'\n  INSTRUMENT CHECK -- frozen/covered ceiling (known answer ~1.0): {frz_cov:.2%}',
          flush=True)
    print(f'  coverage artifact size: frozen covered {frz_cov:.2%} vs all-positions '
          f'{frz_all:.2%}  = {frz_cov-frz_all:+.2%}', flush=True)
    print(f'  THE REAL COMPARISON (covered only): normal {nrm_cov:.2%} vs frozen '
          f'{frz_cov:.2%}   delta {frz_cov-nrm_cov:+.2%}', flush=True)
    print(f'  (v1 §1659 reported 74.42% -> 49.37% on all positions; that delta was the artifact)',
          flush=True)

    res = {'config': {'site': 0, 'fit_rows': 'fineweb_n96_skip1200.pt',
                      'eval_rows': 'fineweb_n192_skip7000.pt',
                      'ablation_constants': 'opt_ablation_consts_all.pt (§1655 fix)',
                      'unseen_token_fallback': 'position-weighted mean (§1655 fix)',
                      'table_refitted_per_condition': True,
                      's1326_normal_ceiling_other_rows': S1326_CEILING},
           'conditions': out,
           'instrument_check_frozen_covered': round(frz_cov, 5),
           'coverage_artifact_points': round(frz_cov - frz_all, 5),
           'real_delta_covered': round(frz_cov - nrm_cov, 5),
           'predictions': {'pred_a_instrument_check_ge_097': bool(pa),
                           'pred_b_coverage_was_artifact_ge_30pts': bool(pb),
                           'pred_c_live_attn0_injects_context_ge_5pts': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
