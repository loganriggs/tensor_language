# ratio_rank8_ablation: A REAL THIRD AXIS — change WHAT is removed, not what it is
# replaced by.
#
# S1650 was designed as a third independent axis and was not one: zero- and
# mean-ablation agree to within .002 on 11 of 12 classes (mean |difference| .00067),
# because they coincide when the global mean of the slice coordinates is near zero.
# The manipulation check passed by .0001 and that invalidated the premise. The design
# error was asserting the intervention differed without measuring whether it differed
# IN EFFECT.
#
# S1650 named what a real axis would be: change WHAT is removed. This ablates a
# RANK-8 slice instead of rank-2 -- four times the subspace -- while the predictor
# stays |lam1/lam2|, the dominance of the top TWO eigendirections. That makes the
# question genuinely new: does top-2 dominance predict the cost of removing the top
# EIGHT? If the slice's causal content is concentrated in the leading pair, it should;
# if directions 3-8 carry independent content, it should not.
#
# APPLYING S1650'S LESSON DIRECTLY: the manipulation check is a REGISTERED prediction
# this time, not an afterthought. pred_b requires the rank-8 intervention to cost
# materially more than rank-2 did on these exact classes. If it does not, the axis did
# not vary and rho is uninterpretable -- and the run says so rather than being read as
# robustness. (A separate pilot was weighed; the full run costs 80 s, so registering
# the check is proportionate rather than piloting first.)
#
# S1648's twelve TYPE-SPANNING classes at mlp11, whose rank-2 mean-ablation numbers are
# already on disk for an exact paired comparison.
#
# Registered predictions:
#   pred_a TOP-2 DOMINANCE STILL PREDICTS: rho(|lam1/lam2|, relative CE rise under
#          RANK-8 ablation) >= +.30.
#   pred_b MANIPULATION CHECK -- THE AXIS ACTUALLY VARIED: mean relative CE rise under
#          rank-8 is at least 2x the rank-2 value of +.00676 (S1648).
#   pred_c CAUSAL CONTENT IS CONCENTRATED IN THE LEADING PAIR: the rank-8 mean rise is
#          LESS than 4x the rank-2 value, i.e. removing four times the dimensions costs
#          less than four times as much (sub-linear in rank).
import json, time, sys, os, re, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152; T = 256
SITE = 11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ratio_rank8_ablation_results.json'
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
NROWS = 480
RANK2_RHO = 0.5105              # §1648, these exact twelve classes, rank-2 mean-ablation
RANK2_MEANRISE = 0.00676        # §1648, mean relative rise over the same twelve
ABL_RANK = 8                    # the axis being varied: 4x the subspace removed

# twelve TYPE-SPANNING classes; none has ever had its CE rise measured
PATS = {'exclaim': r'^!$|^ !$', 'semicolon': r'^;$|^ ;$', 'colon': r'^:$|^ :$',
        'quote': r'^"$|^ "$', 'dash': r'^-$|^ -$', 'digit': r'^ ?[0-9]+$',
        'cap': r'^ [A-Z][a-z]+$', 'we': r'^ (we|We)$', 'you': r'^ (you|You)$',
        'it': r'^ (it|It)$', 'an': r'^ an$', 'my': r'^ my$'}


def rx(pat):
    v = torch.zeros(50257, dtype=torch.bool)
    for t in range(50257):
        if re.match(pat, ENC.decode([t])):
            v[t] = True
    return v


def slice_and_eigs(mask_v):
    """Returns the ABL_RANK subspace to ablate, and the top-2 eigenvalues that form
    the predictor. The ablated subspace is LARGER than the predictor's support -- that
    is the axis being varied."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    order = lam.abs().argsort(descending=True)
    o_abl = order[:ABL_RANK]          # what gets removed
    o_pred = order[:2]                # what the predictor is built from
    return V[:, o_abl].contiguous(), [float(lam[i]) for i in o_pred]


def mk_pre_hook(V2, mu):
    """MEAN-ablation, held fixed at the house default: S1650 showed zero- and
    mean-ablation are near-identical, so the intervention TYPE is not the variable
    here -- the ablated RANK is."""
    def pre(mod, args):
        f = args[0].float()
        p = f @ V2
        return ((f - (p - mu) @ V2.T).to(args[0].dtype),) + tuple(args[1:])
    return pre


@torch.no_grad()
def ce_pass(rows, mask_v, V2, mu):
    """Mean CE on the class's own positions. mu=None captures the global slice mean."""
    cap = {}
    if mu is None:
        def h(mod, args):
            f = args[0].float().reshape(-1, D)
            cap['s'] = cap.get('s', torch.zeros(V2.shape[1], device=DEV)) + (f @ V2).sum(0)
            cap['n'] = cap.get('n', 0) + f.shape[0]
            return None
        handle = H[SITE].mlp.register_forward_pre_hook(h)
    else:
        handle = H[SITE].mlp.register_forward_pre_hook(mk_pre_hook(V2, mu))
    tot, npos = 0.0, 0
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
            pm = mask_v.to(DEV)[tg]; pm[:, :64] = False
            tot += float(ce[pm].sum()); npos += int(pm.sum())
    finally:
        handle.remove()
    return tot / max(npos, 1), npos, ((cap['s'] / max(cap['n'], 1)) if cap else None)


@torch.no_grad()
def main():
    import hashlib
    t0 = time.time()
    raw = torch.load(ROWCACHE, map_location='cpu')
    raw = raw['rows'] if isinstance(raw, dict) else raw
    rows = raw[:NROWS, :T + 1].contiguous()
    rh = hashlib.sha256(open(RECEIPT, 'rb').read()).hexdigest()[:16]
    print(f'RANK-{ABL_RANK} ABLATION (predictor stays top-2): {len(PATS)} classes at mlp{SITE}. {NROWS} canonical '
          f'rows (receipt {rh}). Single registered hypothesis: |lam1/lam2|.', flush=True)

    per = {}
    for c, pat in PATS.items():
        mask_v = rx(pat)
        V2, ev = slice_and_eigs(mask_v)
        base, npos, mu = ce_pass(rows, mask_v, V2, None)
        abl, _, _ = ce_pass(rows, mask_v, V2, mu)
        rise = abl - base
        rel = rise / base if base > 0 else 0.0
        ratio = abs(ev[0]) / max(abs(ev[1]), 1e-12)
        per[c] = {'lam1': round(ev[0], 3), 'lam2': round(ev[1], 3),
                  'ratio': round(ratio, 4), 'base_ce': round(base, 5),
                  'ce_rise': round(rise, 5), 'rel_ce_rise': round(rel, 5),
                  'n_positions': npos}
        print(f'  {c:6s} ratio {ratio:6.3f} | n={npos:5d} | CE {base:.4f} -> {abl:.4f} '
              f'| rel rise {rel:+.5f}', flush=True)

    ks = list(per); n = len(ks)
    ratio = {k: per[k]['ratio'] for k in ks}
    rel = {k: per[k]['rel_ce_rise'] for k in ks}

    def rk(x):
        o = sorted(x, key=lambda z: -x[z]); return [o.index(z) + 1 for z in ks]

    def rho(a, b):
        return 1 - 6 * sum((a[i] - b[i]) ** 2 for i in range(n)) / (n * (n * n - 1))

    r = rho(rk(ratio), rk(rel))
    random.seed(20260827)
    NPERM = 200000
    basep = list(range(1, n + 1))
    hits = sum(1 for _ in range(NPERM)
               if abs(rho(rk(ratio), random.sample(basep, n))) >= abs(r) - 1e-12)
    pval = hits / NPERM
    npos_pos = sum(1 for k in ks if per[k]['ce_rise'] > 0)

    mean_rise = sum(rel.values()) / len(rel)
    pa = r >= 0.30
    pb = mean_rise >= 2.0 * RANK2_MEANRISE
    pc = mean_rise < 4.0 * RANK2_MEANRISE

    print(f'\n  HELD-OUT rho(|lam1/lam2|, relative CE rise) = {r:+.4f}', flush=True)
    print(f'  two-sided sampled permutation p ({NPERM}) = {pval:.5f}', flush=True)
    print(f'  same twelve at RANK-2 (§1648): rho {RANK2_RHO:+.4f}, mean rise '
          f'{RANK2_MEANRISE:+.5f}', flush=True)
    print(f'  MANIPULATION CHECK: rank-{ABL_RANK} mean rise {mean_rise:+.5f} = '
          f'{mean_rise/RANK2_MEANRISE:.2f}x rank-2  (need >=2x for the axis to have varied)',
          flush=True)
    print(f'  concentration: {mean_rise/RANK2_MEANRISE:.2f}x cost for 4x the dimensions', flush=True)
    print(f'  classes with positive CE rise: {npos_pos}/{n}  (§1644 had 11/12)', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'classes': 'TYPE-SPANNING held out -- punctuation, digits, capitalised, function words; no CE rise ever measured for any',
                      'single_registered_hypothesis': 'abs(lam1/lam2) predicts relative CE rise',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)',
                      'rank2_rho_same_classes': RANK2_RHO, 'ablated_rank': ABL_RANK, 'predictor_rank': 2},
           'per_class': per, 'heldout_rho': round(r, 4),
           'permutation_p_two_sided': pval, 'n_permutations': NPERM,
           'n_positive_rise': npos_pos,
           'predictions': {'pred_a_rho_ge_030_rank8': bool(pa),
                           'pred_b_axis_varied_ge_2x': bool(pb),
                           'pred_c_sublinear_lt_4x': bool(pc)},
           'cost_multiple_vs_rank2': round(mean_rise / RANK2_MEANRISE, 3),
           'mean_relative_rise': round(mean_rise, 5),
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
