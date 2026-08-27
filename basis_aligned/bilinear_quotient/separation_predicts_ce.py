# separation_predicts_ce: DOES THE SEPARATION STATISTIC PREDICT ANYTHING CAUSAL?
#
# Twenty-one registered runs (§1623-§1642) calibrated the eigen-slice separation
# statistic: use >= 20 independent bases, match controls on cell, class type and
# configuration, read the GAP not the saturated count, and use rank <= 4. That arc
# established what the statistic MEASURES and how to measure it honestly. It never
# established that it measures anything that MATTERS.
#
# This is the bridge. For each class the slice is ablated at its own site and the CE
# cost is measured on that class's own positions, then correlated against the
# separation gap already measured for the same cells at mlp11 rank-2 TOP-4 with a
# 20-seed null (§1630, §1636, §1642):
#
#     question +.1633 | to +.1435 | period +.0713 | and +.0331 | comma +.0221 | the +.0034
#
# ABLATION: a forward_pre_hook on H[11].mlp replaces the slice coordinates of the
# MLP's input with their GLOBAL mean (mean-ablation, the house optimal-constant
# pattern), leaving everything else untouched. The rest of the model runs unmodified,
# so no block forward is reimplemented and there is no opportunity for the manual
# forward to drift from the real one.
#
# MANDATORY SANITY GATE, NOT A SCORED PREDICTION (LESSONS 22 -- a no-op succeeds
# silently): if the ablation moves CE by less than 1e-4 nats for EVERY class, the hook
# is not wired and the run ABORTS rather than reporting a clean null correlation. A
# dead ablation would otherwise produce "separation predicts nothing" with no error.
#
# Registered predictions:
#   pred_a THE ABLATION COSTS SOMETHING: at least 5 of the 6 classes show a POSITIVE
#          CE rise on their own positions when their slice is mean-ablated.
#   pred_b SEPARATION PREDICTS CAUSAL COST: Spearman rho between the separation gap
#          and the CE rise across the six classes is >= +0.60.
#   pred_c THE TOP OF THE RANGE HOLDS: `question`, which has the largest separation
#          gap, has the largest CE rise of the six.
import json, time, sys, re, torch
import torch.nn.functional as F
sys.path.insert(0, '/workspace/rspd')
from bilin18_joint_removal import m, DEV
import census_lib as cl
import tiktoken

D = 1152; T = 256
SITE = 11                       # single source of truth
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'separation_predicts_ce_results.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
NROWS = 480

# separation gaps from §1630/§1636/§1642, all mlp11 rank-2 TOP-4, 20-seed null
GAP = {'question': 0.1633, 'to': 0.1435, 'period': 0.0713,
       'and': 0.0331, 'comma': 0.0221, 'the': 0.0034}
PATS = {'question': r'^\?$| \?$', 'to': r'^ to$', 'period': r'^\.$|^ \.$',
        'and': r'^ and$', 'comma': r'^,$|^ ,$', 'the': r'^ the$'}
RANK = 2


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def slice_basis(mask_v):
    """The |lambda|-ordered rank-RANK eigen slice of the class-projected quadratic."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:RANK]
    return V[:, o].contiguous()


def mk_pre_hook(V2, mu):
    """Replace the slice coordinates of the MLP input with their global mean."""
    def pre(mod, args):
        xin = args[0]
        f = xin.float()
        p = f @ V2
        return ((f - (p - mu) @ V2.T).to(xin.dtype),) + tuple(args[1:])
    return pre


@torch.no_grad()
def run_pass(rows, mask_v, V2=None, mu=None):
    """Per-position CE, plus the mean slice projection when V2 is given and mu is not."""
    handle = None
    if V2 is not None and mu is not None:
        handle = H[SITE].mlp.register_forward_pre_hook(mk_pre_hook(V2, mu))
    tot_ce = 0.0
    n_pos = 0
    proj_sum = torch.zeros(RANK, device=DEV)
    proj_n = 0
    cap = {}
    if V2 is not None and mu is None:
        def cap_pre(mod, args):
            f = args[0].float().reshape(-1, D)
            cap['s'] = cap.get('s', torch.zeros(RANK, device=DEV)) + (f @ V2).sum(0)
            cap['n'] = cap.get('n', 0) + f.shape[0]
            return None
        handle = H[SITE].mlp.register_forward_pre_hook(cap_pre)
    try:
        for i in range(0, rows.shape[0], 8):
            bb = rows[i:i + 8]
            idx = bb[:, :-1].to(DEV).contiguous(); tg = bb[:, 1:].to(DEV)
            x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
            for blk in H:
                x, v1 = blk(x, v1, x0)
            lg = 30.0 * torch.tanh(m.lm_head(F.rms_norm(x, (D,))) / 30.0)
            ce = F.cross_entropy(lg.reshape(-1, lg.shape[-1]).float(), tg.reshape(-1),
                                 reduction='none').reshape(tg.shape)
            pm = mask_v.to(DEV)[tg]
            pm[:, :64] = False
            tot_ce += float(ce[pm].sum())
            n_pos += int(pm.sum())
    finally:
        if handle is not None:
            handle.remove()
    if cap:
        proj_sum = cap['s']; proj_n = cap['n']
    return (tot_ce / max(n_pos, 1)), n_pos, (proj_sum / max(proj_n, 1))


@torch.no_grad()
def main():
    import os, hashlib
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    rows = raw[:NROWS, :T + 1].contiguous()
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'CANONICAL .rowcache/fineweb_n480_skip80.pt: {NROWS} rows, mean-ablation at '
          f'mlp{SITE} rank-{RANK} (receipt {rh})', flush=True)

    out_cls = {}
    for c, pat in PATS.items():
        mask_v = rx(pat)
        V2 = slice_basis(mask_v)
        base_ce, npos, mu = run_pass(rows, mask_v, V2=V2, mu=None)
        abl_ce, _, _ = run_pass(rows, mask_v, V2=V2, mu=mu)
        rise = abl_ce - base_ce
        out_cls[c] = {'gap': GAP[c], 'base_ce': round(base_ce, 5),
                      'abl_ce': round(abl_ce, 5), 'ce_rise': round(rise, 5),
                      'n_positions': npos}
        print(f'  {c:9s} gap {GAP[c]:+.4f} | n={npos:5d} | CE {base_ce:.5f} -> '
              f'{abl_ce:.5f} | rise {rise:+.5f}', flush=True)

    rises = {c: out_cls[c]['ce_rise'] for c in out_cls}

    # MANDATORY SANITY GATE -- a dead hook must abort, not report a clean null
    if max(abs(v) for v in rises.values()) < 1e-4:
        print('\n  SANITY GATE FAILED: no class moved CE by >= 1e-4. The ablation hook is '
              'not wired. ABORTING rather than reporting a null correlation.', flush=True)
        sys.stdout.flush(); os._exit(3)
    print(f'\n  sanity gate PASSED: max |CE move| = {max(abs(v) for v in rises.values()):.5f}',
          flush=True)

    ks = list(out_cls)
    def rank_of(d):
        o = sorted(d, key=lambda k: -d[k]); return {k: o.index(k) + 1 for k in d}
    rg = rank_of({k: GAP[k] for k in ks}); rc = rank_of(rises)
    n = len(ks); dsq = sum((rg[k] - rc[k]) ** 2 for k in ks)
    rho = 1 - 6 * dsq / (n * (n * n - 1))

    pa = sum(1 for v in rises.values() if v > 0) >= 5
    pb = rho >= 0.60
    pc = max(rises, key=lambda k: rises[k]) == 'question'

    print(f'\n  separation gap vs CE rise, {n} classes at mlp{SITE} rank-{RANK}:', flush=True)
    for k in sorted(ks, key=lambda x: -GAP[x]):
        print(f'    {k:9s} gap {GAP[k]:+.4f} (rank {rg[k]})   CE rise {rises[k]:+.5f} '
              f'(rank {rc[k]})', flush=True)
    print(f'  Spearman rho = {rho:+.3f}', flush=True)
    print(f'  classes with positive rise: {sum(1 for v in rises.values() if v > 0)}/6', flush=True)
    print(f'  largest CE rise: {max(rises, key=lambda k: rises[k])}', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'ablation': 'mean-ablation of the slice coordinates at the mlp input '
                                  'via forward_pre_hook; rest of the model unmodified',
                      'row_source': 'fineweb_n480_skip80.pt (.rowcache, authority '
                                    'pinned_local_ordered_manifest)',
                      'gaps_from': 'S1630/S1636/S1642, mlp11 rank-2 TOP-4, 20-seed null'},
           'classes': out_cls, 'spearman_rho': round(rho, 4),
           'predictions': {'pred_a_5of6_positive_rise': bool(pa),
                           'pred_b_rho_ge_060': bool(pb),
                           'pred_c_question_largest': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
