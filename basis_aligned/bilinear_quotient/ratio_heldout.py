# ratio_heldout: THE HONEST COMPLETION — the eigenvalue RATIO, pre-registered as the
# SINGLE hypothesis, on TWELVE CLASSES IT WAS NEVER FITTED ON.
#
# §1646 tested three candidate predictors of causal cost on the same twelve classes at
# the same corrected currency (relative CE rise, §1645): separation gap +.231 (p .235),
# eigenvalue ratio |lam1/lam2| +.301 (p .342), eigenvalue magnitude |lam1| +.154 (p
# .635). None significant. I wrote there that chasing the ratio to significance on that
# data would be wrong, because it is the BEST OF THREE tried on a fixed set and its
# .301 is optimistically biased by selection.
#
# The stated correct move was a held-out set with the ratio registered IN ADVANCE as
# the single hypothesis. This is that run. Twelve classes that appear in NO prior
# section of this arc: for, on, that, as, was, but, not, this, which, or, had, they.
#
# It is cheap because the gap is not needed. The ratio comes from weights alone (one
# eigendecomposition per class, no rows) and the CE rise needs two forward passes per
# class, so this costs ~3 minutes rather than the ~25 a gap sweep would.
#
# ONE HYPOTHESIS, REGISTERED BEFORE THE RUN: the eigenvalue ratio predicts relative CE
# rise. No second predictor is computed, so there is nothing to select over.
#
# Outcome either way closes the line honestly. If the ratio holds on classes it was
# never fitted on, that is a real finding about what makes a slice causally load-
# bearing. If it does not, the negative from §1644/§1646 is complete: nothing tested
# predicts the causal cost, and the separation apparatus has no demonstrated
# consequence.
#
# NOTE: sys.path is set to this file's own directory explicitly. /workspace/rspd does
# NOT exist (measured 14:44) and every earlier script in this lineage imported the
# model by accident of the runner's cwd.
#
# Registered predictions:
#   pred_a THE RATIO REPLICATES OUT OF SAMPLE: rho(|lam1/lam2|, relative CE rise) >=
#          +.30 on the twelve held-out classes, matching its discovery value.
#   pred_b AND IT CLEARS SIGNIFICANCE: two-sided sampled permutation p < .05.
#   pred_c THE ONE SOLID FINDING REPLICATES: at least 10 of 12 held-out classes show a
#          POSITIVE CE rise when their own slice is mean-ablated (§1644 had 11/12).
import json, time, sys, os, re, random, torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bilin18_joint_removal import m, DEV
import tiktoken

D = 1152; T = 256
SITE = 11
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'ratio_heldout_results.json'
ROWCACHE = PT + '.rowcache/fineweb_n480_skip80.pt'
RECEIPT = PT + '.rowcache/fineweb_oracle_v2_receipt.json'
H = m.transformer.h
ENC = tiktoken.get_encoding('gpt2')
RANK = 2
NROWS = 480
DISCOVERY_RHO = 0.3007          # §1646, in-sample, best of three

# twelve classes appearing in NO prior section of this arc
PATS = {'for': r'^ for$', 'on': r'^ on$', 'that': r'^ that$', 'as': r'^ as$',
        'was': r'^ was$', 'but': r'^ but$', 'not': r'^ not$', 'this': r'^ this$',
        'which': r'^ which$', 'or': r'^ or$', 'had': r'^ had$', 'they': r'^ they$'}


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
    print(f'HELD-OUT: {len(PATS)} classes never used in §1623-§1646. {NROWS} canonical '
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
    pc = npos_pos >= 10

    print(f'\n  HELD-OUT rho(|lam1/lam2|, relative CE rise) = {r:+.4f}', flush=True)
    print(f'  two-sided sampled permutation p ({NPERM}) = {pval:.5f}', flush=True)
    print(f'  in-sample discovery value was {DISCOVERY_RHO:+.4f} (best of three, p .342)',
          flush=True)
    print(f'  classes with positive CE rise: {npos_pos}/{n}  (§1644 had 11/12)', flush=True)

    out = {'config': {'site': SITE, 'rank': RANK, 'n_rows': NROWS,
                      'classes': 'held out -- appear in no prior section of the arc',
                      'single_registered_hypothesis': 'abs(lam1/lam2) predicts relative CE rise',
                      'currency': 'relative CE rise = ce_rise / base_ce (§1645)',
                      'discovery_rho_in_sample': DISCOVERY_RHO},
           'per_class': per, 'heldout_rho': round(r, 4),
           'permutation_p_two_sided': pval, 'n_permutations': NPERM,
           'n_positive_rise': npos_pos,
           'predictions': {'pred_a_rho_ge_030_heldout': bool(pa),
                           'pred_b_perm_p_lt_05': bool(pb),
                           'pred_c_10of12_positive_rise': bool(pc)},
           'runtime_s': round(time.time() - t0, 1)}
    json.dump(out, open(OUT, 'w'), indent=1,
              default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    print(f'\npred_a {pa} | pred_b {pb} | pred_c {pc}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]}s)', flush=True)
    sys.stdout.flush(); sys.stderr.flush(); os._exit(0)


if __name__ == '__main__':
    main()
