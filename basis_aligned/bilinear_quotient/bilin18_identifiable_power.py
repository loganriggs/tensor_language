"""Does the identifiable-fraction measurement on bilin18 have any power at all?

`bilin18_identifiable.py` measured, at N = 4000 samples, an identifiable fraction of
0.0075 against a random-form null of 0.0060 -- i.e. ~1x chance on 5 of 7 layers. Read
naively that says "an MLP's interaction form is no more aligned with the data than a
random symmetric matrix is". But two very different worlds produce that same number:

  (a) the learned part of the form really is a tiny share of its Frobenius mass, the
      rest being initialisation residue that no gradient ever touched; or
  (b) the learned part is large but spread over FAR more than 4000 directions, so
      4000 samples capture only 4000/k of it -- which for k ~ 10^5 is chance.

The measurement as run cannot tell these apart, so the honest thing is to test which.
Two diagnostics, both cheap:

1. SAMPLE SCALING. A form with no relationship to the data has identifiable fraction
   exactly N/dim -- dead linear in N, doubling when N doubles. A form concentrated on
   a k-dimensional data-aligned subspace saturates once N approaches k. So sweep N and
   look at the SHAPE, not the value. Linear to the last point = no concentration
   detectable below k ~ N. Bending = concentration, and where it bends estimates k.

2. THE DATA'S OWN DIMENSION. The Gram G_ij = (x_i.x_j)^2 IS the metric on
   span{vec(x x^T)}; its eigenvalue spectrum says how many directions of Sym^2 the
   residual stream actually visits. If the effective rank of G is far below N, the
   samples are redundant and a data-aligned form would have shown up loudly -- the
   test HAD power and world (a) is the answer. If the effective rank tracks N, the
   data itself is high-dimensional in Sym^2, the test is sample-starved, and no
   conclusion about the weights is licensed.

Diagnostic 2 is what decides whether diagnostic 1's linearity means anything.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language')
from tier2_model import load_elriggs
from bilin18_identifiable import mlp_inputs, form_for_direction, identifiable_fraction

DEV = 'cuda'
N_MAX = 8000
NS = (250, 500, 1000, 2000, 4000, 8000)
LAYERS = (0, 5, 13, 17)
N_DIRS = 4
DIM_SYM2 = 1152 * 1153 // 2


def eff_rank(evals):
    """Participation ratio of the spectrum: (sum e)^2 / sum e^2, the scale-free count
    of directions carrying the mass."""
    e = evals.clamp_min(0)
    return float(e.sum() ** 2 / (e ** 2).sum().clamp_min(1e-300))


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    X = mlp_inputs(model, tokens, LAYERS, N_MAX)
    g = torch.Generator().manual_seed(0)
    out = {'dim_sym2': DIM_SYM2, 'sample_counts': list(NS), 'layers': {}}

    print('== diagnostic 2: how many directions of Sym^2 does the residual stream '
          'actually visit? ==')
    print(f"  (Gram G_ij = (x_i.x_j)^2 over {N_MAX} samples; Sym^2 has "
          f"{DIM_SYM2:,} dimensions)\n")
    print(f"  {'layer':>5}  {'eff. rank of G':>15}  {'/N':>7}  "
          f"{'evals > 1e-6 * max':>19}")
    grams = {}
    for li in LAYERS:
        Xi = X[li].to(DEV)
        G = (Xi @ Xi.T) ** 2
        ev = torch.linalg.eigvalsh(G)
        grams[li] = ev
        er = eff_rank(ev)
        nz = int((ev > 1e-6 * ev.max()).sum())
        out['layers'][li] = {'gram_eff_rank': er, 'gram_eff_rank_over_N': er / N_MAX,
                             'gram_numerical_rank': nz}
        print(f"  {li:>5}  {er:>15.1f}  {er/N_MAX:>7.3f}  {nz:>19d}")

    print(f"\n== diagnostic 1: does the identifiable fraction bend, or stay linear "
          f"in N? ==")
    print(f"  a form unrelated to the data sits at exactly N/dim (last column);")
    print(f"  a form concentrated on k data directions saturates as N -> k\n")
    hdr = '  ' + f"{'layer':>5}  " + ''.join(f"{'N=' + str(n):>10}" for n in NS) + \
          f"{'linear?':>10}"
    print(hdr)
    for li in LAYERS:
        Xfull = X[li].to(DEV)
        mlp = model.transformer.h[li].mlp
        dirs = []
        for _ in range(N_DIRS):
            d = torch.randn(cfg['n_embd'], generator=g).to(DEV)
            dirs.append(form_for_direction(mlp, d / d.norm()))
        curve = []
        for n in NS:
            Xi = Xfull[:n]
            curve.append(sum(identifiable_fraction(M, Xi) for M in dirs) / N_DIRS)
        # a perfectly linear curve has fraction/N constant; measure the drift
        ratios = [c / (n / DIM_SYM2) for c, n in zip(curve, NS)]
        bend = ratios[-1] / ratios[0]
        out['layers'][li]['curve'] = curve
        out['layers'][li]['ratio_to_chance_curve'] = ratios
        out['layers'][li]['bend'] = bend
        print('  ' + f"{li:>5}  " + ''.join(f"{c:>10.4f}" for c in curve) +
              f"{bend:>9.2f}x", flush=True)

    # the same sweep for a form with no relationship to the data, as the shape null
    Xi = X[LAYERS[0]].to(DEV)
    An = torch.randn(cfg['n_embd'], cfg['n_embd'], generator=g).double().to(DEV)
    An = 0.5 * (An + An.T)
    ncurve = [identifiable_fraction(An, Xi[:n]) for n in NS]
    nratio = [c / (n / DIM_SYM2) for c, n in zip(ncurve, NS)]
    out['null_curve'] = ncurve
    out['null_bend'] = nratio[-1] / nratio[0]
    print('  ' + f"{'null':>5}  " + ''.join(f"{c:>10.4f}" for c in ncurve) +
          f"{out['null_bend']:>9.2f}x")

    ers = [out['layers'][li]['gram_eff_rank_over_N'] for li in LAYERS]
    bends = [out['layers'][li]['bend'] for li in LAYERS]
    out['summary'] = {'mean_gram_eff_rank_over_N': sum(ers) / len(ers),
                      'mean_bend': sum(bends) / len(bends)}
    print(f"\nSUMMARY")
    print(f"  data's own Sym^2 effective rank is {out['summary']['mean_gram_eff_rank_over_N']:.2f} "
          f"x N -- i.e. {'the samples are nearly independent, so the test is sample-starved' if out['summary']['mean_gram_eff_rank_over_N'] > 0.3 else 'the samples are highly redundant, so the test HAD power'}")
    print(f"  trained forms bend {out['summary']['mean_bend']:.2f}x over a 32x range in N; "
          f"the unrelated-form null bends {out['null_bend']:.2f}x")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_identifiable_power_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
