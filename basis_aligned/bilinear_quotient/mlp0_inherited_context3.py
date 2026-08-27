# mlp0_inherited_context3: THE ACTUAL FIX — v2's instrument check failed and the cause
# is that POSITION-WISE MASKING AT SCORING TIME CANNOT ISOLATE A POSITION-WISE
# INTERVENTION IN A TRANSFORMER.
#
# v1 measured a 25-point ceiling fall under attn0-freeze and I called it coverage.
# v2 scored covered-only positions and carried an INSTRUMENT CHECK with a known answer:
# with attn0 frozen, mlp0's input is a pure function of the current token, so a covered
# table must reproduce it and the ceiling must be ~1.0. v2 returned 55.83%. The check
# refused the result, correctly.
#
# Two diagnostics located the cause and both are recorded because each ruled something
# out:
#   1. Does the freeze work? Within-token variance of mlp0's output falls to 8.1e-06
#      (ratio 0.0000 of total) under freeze. YES -- mlp0 is exactly token-determined.
#   2. Is the table wrong? Relative L2 error of table vs actual mlp0 output on covered
#      eval positions: 3e-06. NO -- the table is exact.
#
# So the table is exact, the freeze is exact, and the ceiling still read 55.83%. The
# cause is that v2 SUBSTITUTED AT EVERY POSITION -- including the 23.4% whose token has
# no fitted entry and received the position-weighted mean -- and then merely EXCLUDED
# those positions from the CE average. But a wrong mlp0 output at an uncovered position
# propagates through layers 1-17 and ATTENTION MIXES IT INTO THE PREDICTIONS AT COVERED
# POSITIONS. The damage is non-local; masking the score does not undo it.
#
# FIX: substitute only where the table has a fitted entry and leave mlp0 LIVE at
# uncovered positions. The forward is then exact everywhere the table does not claim to
# apply, and the covered-position CE measures the table's own fidelity.
#
# Registered predictions:
#   pred_a THE INSTRUMENT CHECK NOW PASSES: with attn0 frozen and the hybrid hook, the
#          covered-position ceiling is >= 0.97. Until this holds nothing else in the run
#          is interpretable.
#   pred_b THE DIAGNOSIS WAS RIGHT: that ceiling exceeds v2's 55.83% by >= 35 points.
#   pred_c THE REAL QUESTION, finally measurable: with live attn0 the ceiling is BELOW
#          the frozen ceiling by >= 5 points, i.e. attn0 injects context a
#          current-token table cannot reproduce.
import json, time, sys, os, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV

D = 1152; T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp0_inherited_context3_results.json'
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
        def tbl_hook(mo, a, o):
            idx_f = state['idx'].reshape(-1)
            sub = tbl[idx_f].reshape(o.shape).to(o.dtype)
            if seen_mask is None:
                return sub
            cov = seen_mask.to(DEV)[state['idx']].unsqueeze(-1)   # LIVE where uncovered
            return torch.where(cov, sub, o)
        hs.append(H[0].mlp.register_forward_hook(tbl_hook))
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
    print(f'mlp0 INHERITED CONTEXT v3 (hybrid hook: table where covered, LIVE where not) | optimal constants from opt_ablation_consts_all.pt '
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
    V2_FROZEN_COVERED = 0.5583
    pa = frz_cov >= 0.97
    pb = (frz_cov - V2_FROZEN_COVERED) >= 0.35
    pc = (frz_cov - nrm_cov) >= 0.05

    print(f'\n  INSTRUMENT CHECK -- frozen/covered ceiling (known answer ~1.0): {frz_cov:.2%}',
          flush=True)
    print(f'  vs v2 (masked-score, contaminated forward) 55.83%: {frz_cov-0.5583:+.2%}', flush=True)
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
                           'pred_b_beats_v2_by_ge_35pts': bool(pb),
                           'pred_c_live_attn0_injects_context_ge_5pts': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(res, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({res["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
