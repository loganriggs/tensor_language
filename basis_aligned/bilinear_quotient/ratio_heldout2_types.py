# ratio_heldout2_types: DOES THE RATIO HOLD ACROSS CLASS TYPES, OR ONLY WITHIN
# FUNCTION WORDS? — the second held-out set §1647 said was required before any claim.
#
# §1647 was the arc's first significant result: |lam1/lam2| predicts relative CE rise
# at rho +.678, permutation p .019, on twelve held-out classes with the hypothesis
# registered in advance. I did NOT promote it, for three recorded reasons. The second
# is the one this run tests:
#
#   Out-of-sample rho (.678) EXCEEDED in-sample (.301), which is backwards. The likely
#   cause is composition -- §1646's discovery set MIXED punctuation with function
#   words, while every one of §1647's held-out classes is a FUNCTION WORD. So the
#   relationship may be clean within function words and confounded across types, which
#   is exactly what §1637's class-type lesson would predict.
#
# This set spans types deliberately: five punctuation classes (exclaim, semicolon,
# colon, quote, dash), digits, capitalised tokens, and four function words (we, you,
# it, an, my). None has ever had its CE rise measured -- §1637 and §1642 measured
# SEPARATION for some of them, never the ablation cost, so the ratio->CE relationship
# has never been fitted on any of these.
#
# The first caution in §1647 remains live and this run does not address it: rho .678 at
# p .019 on n=12 sits almost exactly on §1614's .6727 at p .0192, which §1616 refuted.
# Only accumulated independent replication touches that, not a single further set.
#
# Registered predictions:
#   pred_a THE RELATIONSHIP SURVIVES TYPE-SPANNING: rho(|lam1/lam2|, relative CE rise)
#          >= +.30 on this mixed-type set.
#   pred_b AND STAYS SIGNIFICANT: two-sided sampled permutation p < .05.
#   pred_c BUT IT DEGRADES, as caution 2 predicts: rho is BELOW §1647's +.6783. A
#          type-spanning set should be harder than a homogeneous one; if rho instead
#          holds or rises, caution 2 is wrong and the relationship is type-general.
import json, time, sys, os, re, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152; T = 256
SITE = 11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ratio_heldout2_types_results.json'
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
NROWS = 480
HELDOUT1_RHO = 0.6783           # §1647, function words only, p .019

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
    """Top-RANK |lambda| eigenpair of the class-projected quadratic. Weights only."""
    WU = m.lm_head.weight.float().to(DEV)[:50257]
    u = WU[mask_v.to(DEV)].mean(0); u = u / u.norm()
    Lw = H[SITE].mlp.Left.weight.float(); Rw = H[SITE].mlp.Right.weight.float()
    Dw = H[SITE].mlp.Down.weight.float()
    S = 0.5 * ((Lw.T @ ((u @ Dw)[:, None] * Rw)) + (Lw.T @ ((u @ Dw)[:, None] * Rw)).T)
    lam, V = torch.linalg.eigh(S)
    o = lam.abs().argsort(descending=True)[:RANK]
    return V[:, o].contiguous(), [float(lam[i]) for i in o]


def mk_pre_hook(V2, mu):
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
            cap['s'] = cap.get('s', torch.zeros(RANK, device=DEV)) + (f @ V2).sum(0)
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
    print(f'HELD-OUT 2 (TYPE-SPANNING): {len(PATS)} classes, no CE rise ever measured. {NROWS} canonical '
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

    pa = r >= 0.30
    pb = pval < 0.05
    pc = r < HELDOUT1_RHO

    print(f'\n  HELD-OUT rho(|lam1/lam2|, relative CE rise) = {r:+.4f}', flush=True)
    print(f'  two-sided sampled permutation p ({NPERM}) = {pval:.5f}', flush=True)
    print(f'  §1647 function-words-only held-out was {HELDOUT1_RHO:+.4f} (p .019)', flush=True)
    print(f'  caution-2 test -- degrades across types: {r < HELDOUT1_RHO}', flush=True)
    print(f'  classes with positive CE rise: {npos_pos}/{n}  (§1644 had 11/12)', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'classes': 'TYPE-SPANNING held out -- punctuation, digits, capitalised, function words; no CE rise ever measured for any',
                      'single_registered_hypothesis': 'abs(lam1/lam2) predicts relative CE rise',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)',
                      'heldout1_rho_function_words': HELDOUT1_RHO},
           'per_class': per, 'heldout_rho': round(r, 4),
           'permutation_p_two_sided': pval, 'n_permutations': NPERM,
           'n_positive_rise': npos_pos,
           'predictions': {'pred_a_rho_ge_030_typespanning': bool(pa),
                           'pred_b_perm_p_lt_05': bool(pb),
                           'pred_c_degrades_vs_heldout1': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
