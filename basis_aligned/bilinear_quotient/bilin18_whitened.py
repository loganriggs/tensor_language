"""Does whitening actually change which directions you would pick out of a bilin18 form?

§6.5 of BILIN18_CONNECTION.md ends with a recommendation -- whiten by the input second
moment before reading any mass statistic off bilin18 weights -- justified by the finding
that on layers 1-13 most of a form's Frobenius norm sits in directions the data never
visits. A recommendation is not a result. This tests it.

THE TEST. For an output direction d with interaction form M_d, take the rank-k
approximation two ways and compare how much of the FUNCTION each keeps:

  raw       top-k eigenpairs of M_d, by |eigenvalue|          -- what mass statistics do
  whitened  top-k eigenpairs of S^{1/2} M_d S^{1/2}, mapped back  -- the Lambda-weighted metric

where S is the second moment of the layer's own rms-normed MLP input. Score both by the
fraction of variance unexplained on held-out inputs,

  FVU(k) = Var_x[ x^T M_d x - x^T M_k x ] / Var_x[ x^T M_d x ]

which is a purely functional quantity: it does not care about Frobenius norm at all.

Whitened truncation is the right thing to do under the functional metric, so it should
win; the question is by how much and where. REGISTERED PREDICTION, written before the
run: the gap is large through the middle of the network (layers 5-13, where curvature
along the dominant direction is 0.00-0.19x random and the norm is therefore mostly dead
weight) and small at layer 0 and layer 17 (top-PC share 0.086 and 0.543, forms enriched
rather than blind). If instead the gap is uniform across depth, the mechanism story in
§6.4 is not what drives it and the recommendation is right for the wrong reason.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language')
from tier2_model import load_elriggs
from bilin18_identifiable import mlp_inputs, form_for_direction

DEV = 'cuda'
N_FIT = 6000            # inputs used to build S
N_TEST = 6000           # held-out inputs used to score
LAYERS = (0, 1, 5, 9, 13, 17)
KS = (1, 2, 4, 8, 16, 32, 64, 128)
N_DIRS = 8


def sqrtm_psd(S, eps=1e-10):
    ev, U = torch.linalg.eigh(S)
    ev = ev.clamp_min(0)
    cut = eps * ev.max()
    r = ev.sqrt()
    ri = torch.where(ev > cut, ev.clamp_min(cut).rsqrt(), torch.zeros_like(ev))
    return (U * r) @ U.T, (U * ri) @ U.T


def fvu(M, Mk, X):
    """Fraction of variance unexplained of the quadratic feature on held-out X."""
    f = torch.einsum('ni,ij,nj->n', X, M, X)
    fk = torch.einsum('ni,ij,nj->n', X, Mk, X)
    return float((f - fk).var() / f.var().clamp_min(1e-300))


def truncate(M, k):
    ev, U = torch.linalg.eigh(M)
    idx = ev.abs().argsort(descending=True)[:k]
    Uk, ek = U[:, idx], ev[idx]
    return (Uk * ek) @ Uk.T


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    X = mlp_inputs(model, tokens, LAYERS, N_FIT + N_TEST)
    g = torch.Generator().manual_seed(0)
    out = {'ks': list(KS), 'n_fit': N_FIT, 'n_test': N_TEST, 'layers': {},
           'prediction': 'gap large at layers 5-13, small at 0 and 17'}

    print('== how much of the FUNCTION survives a rank-k truncation of the interaction '
          'form? ==')
    print('   FVU on held-out inputs; lower is better. "raw" ranks eigenpairs by '
          '|eigenvalue|,')
    print('   "whitened" ranks them in the Lambda-weighted functional metric.\n')
    hdr = f"  {'layer':>5}  {'rank':>9}  " + ''.join(f"{'k=' + str(k):>9}" for k in KS)
    for li in LAYERS:
        Xa = X[li].to(DEV)
        Xf, Xt = Xa[:N_FIT], Xa[N_FIT:N_FIT + N_TEST]
        S = Xf.T @ Xf / Xf.shape[0]
        Sh, Sih = sqrtm_psd(S)
        mlp = model.transformer.h[li].mlp
        raw = {k: [] for k in KS}
        wht = {k: [] for k in KS}
        for _ in range(N_DIRS):
            d = torch.randn(cfg['n_embd'], generator=g).to(DEV)
            M = form_for_direction(mlp, d / d.norm())
            Mw = Sh @ M @ Sh
            for k in KS:
                raw[k].append(fvu(M, truncate(M, k), Xt))
                wht[k].append(fvu(M, Sih @ truncate(Mw, k) @ Sih, Xt))
        r = [sum(raw[k]) / N_DIRS for k in KS]
        w = [sum(wht[k]) / N_DIRS for k in KS]
        # k needed to reach 10% FVU, by each ranking
        def k90(c):
            for k, v in zip(KS, c):
                if v <= 0.10:
                    return k
            return None
        out['layers'][li] = {'raw_fvu': r, 'whitened_fvu': w,
                             'k_for_90pct_raw': k90(r), 'k_for_90pct_whitened': k90(w),
                             'gap_at_k16': r[KS.index(16)] / max(w[KS.index(16)], 1e-30)}
        print(hdr if li == LAYERS[0] else '', end='')
        print(f"  {li:>5}  {'raw':>9}  " + ''.join(f"{v:>9.3f}" for v in r))
        print(f"  {'':>5}  {'whitened':>9}  " + ''.join(f"{v:>9.3f}" for v in w))
        print(f"  {'':>5}  {'->':>9}  rank needed for 90% of the function: raw "
              f"{out['layers'][li]['k_for_90pct_raw']}, whitened "
              f"{out['layers'][li]['k_for_90pct_whitened']}"
              f"   (FVU ratio at k=16: {out['layers'][li]['gap_at_k16']:.1f}x)\n",
              flush=True)

    mid = [out['layers'][li]['gap_at_k16'] for li in (1, 5, 9, 13)]
    end = [out['layers'][li]['gap_at_k16'] for li in (0, 17)]
    out['summary'] = {'mean_gap_mid_layers': sum(mid) / len(mid),
                      'mean_gap_end_layers': sum(end) / len(end)}
    print(f"SUMMARY at k=16: whitening helps by {out['summary']['mean_gap_mid_layers']:.1f}x "
          f"through the middle (layers 1,5,9,13) and "
          f"{out['summary']['mean_gap_end_layers']:.1f}x at the ends (layers 0,17)")
    out['prediction_held'] = bool(out['summary']['mean_gap_mid_layers'] >
                                  2 * out['summary']['mean_gap_end_layers'])
    print(f"registered prediction (large in the middle, small at the ends): "
          f"{'HELD' if out['prediction_held'] else 'FAILED'}")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_whitened_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
