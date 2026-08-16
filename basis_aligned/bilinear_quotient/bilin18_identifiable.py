"""bilin18, test #2 from BILIN18_CONNECTION.md: how much of an MLP's interaction
form is identifiable from the data at all?

A2-3 on the toy: one-hot inputs probe only 529 of the 1081 dimensions of Sym^2, so
29-35% of a trained model's interaction mass cannot affect the function and must be
projected away before any weight-space mass statistic means anything. The prediction
recorded for bilin18 was that the identifiable fraction there is MUCH lower, because
the rms-normed residual stream at width 1152 occupies a small manifold inside a
664,128-dimensional Sym^2.

The measurement is cheap despite the dimension, because it never builds anything of
size dim Sym^2. For N sampled inputs the identifiable subspace is
span{vec(x x^T)}, of dimension at most N, and the two ingredients are

    Gram   G_ij = <x_i x_i^T, x_j x_j^T> = (x_i . x_j)^2
    coeffs b_i  = <M_d, x_i x_i^T>       = x_i^T M_d x_i

so the projection of M_d onto that span has squared norm c.b where G c = b. The
identifiable fraction is then (c.b) / ||M_d||_F^2, and the chance level for a form
with no relationship to the data is N / dim Sym^2.

Per the repo's own convention (METHODS.md), the input to each MLP is the rms-normed
block input, and the form for an output direction d is
    M_d = sym( sum_j (W_D^T d)_j  W_L[j] (x) W_R[j] ).
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
sys.path.insert(0, '/workspace/tensor_language')
from tier2_model import load_elriggs, eval_ce

DEV = 'cuda'
N_SAMPLES = 4000
LAYERS = (0, 1, 5, 9, 13, 16, 17)
N_DIRS = 8


def mlp_inputs(model, tokens, layers, n_samples):
    """rms-normed input to each block's MLP, sampled over positions."""
    store = {}
    hooks = []

    def mk(li):
        def hook(mod, inp, outp):
            store.setdefault(li, []).append(inp[0].detach().reshape(-1, inp[0].shape[-1]))
        return hook

    for li in layers:
        hooks.append(model.transformer.h[li].mlp.register_forward_hook(mk(li)))
    with torch.no_grad():
        for i in range(0, len(tokens), 4):
            model(tokens[i:i + 4].to(DEV))
    for h in hooks:
        h.remove()
    out = {}
    g = torch.Generator().manual_seed(0)
    for li in layers:
        X = torch.cat(store[li], 0)
        idx = torch.randperm(X.shape[0], generator=g)[:n_samples]
        out[li] = X[idx].double()
    return out


def form_for_direction(mlp, d):
    """M_d = sym( sum_j c_j L_j (x) R_j ), c = W_D^T d."""
    c = (mlp.Down.weight.T @ d).double()            # (4608,)
    L = mlp.Left.weight.double()                    # (4608, 1152)
    R = mlp.Right.weight.double()
    A = (L * c[:, None]).T @ R                      # (1152, 1152)
    return 0.5 * (A + A.T)


def identifiable_fraction(M, X, ridge=1e-8):
    """Share of ||M||_F^2 that survives projection onto span{x x^T : x in X}."""
    G = (X @ X.T) ** 2
    b = torch.einsum('ni,ij,nj->n', X, M, X)
    G = G + ridge * torch.diag(G).mean() * torch.eye(G.shape[0], device=G.device,
                                                     dtype=G.dtype)
    c = torch.linalg.solve(G, b)
    return float((c @ b) / (M ** 2).sum())


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    out = {'config': {k: cfg[k] for k in ('n_layer', 'n_embd', 'n_head')},
           'n_samples': N_SAMPLES, 'dim_sym2': 1152 * 1153 // 2,
           'chance_fraction': N_SAMPLES / (1152 * 1153 // 2)}
    print(f"dim Sym^2 = {out['dim_sym2']:,} | {N_SAMPLES} samples -> chance identifiable "
          f"fraction {out['chance_fraction']:.5f}")

    ce = eval_ce(model, tokens, batch=4)
    out['ce_gate'] = ce
    print(f'CE gate: {ce:.4f}\n')

    X = mlp_inputs(model, tokens, LAYERS, N_SAMPLES)
    g = torch.Generator().manual_seed(0)
    out['layers'] = {}
    print(f"{'layer':>5}  {'identifiable fraction of the interaction form':<46} "
          f"{'x chance':>9}")
    for li in LAYERS:
        mlp = model.transformer.h[li].mlp
        Xi = X[li].to(DEV)
        fr = []
        for k in range(N_DIRS):
            d = torch.randn(cfg['n_embd'], generator=g).to(DEV)
            d = d / d.norm()
            M = form_for_direction(mlp, d)
            fr.append(identifiable_fraction(M, Xi))
        # a form with no relationship to the data, as the null
        Anull = torch.randn(cfg['n_embd'], cfg['n_embd'], generator=g).double().to(DEV)
        Anull = 0.5 * (Anull + Anull.T)
        null = identifiable_fraction(Anull, Xi)
        mean = sum(fr) / len(fr)
        out['layers'][li] = {'identifiable_mean': mean, 'identifiable_all': fr,
                             'random_form_null': null, 'ratio_to_null': mean / max(null, 1e-12)}
        print(f"{li:>5}  mean {mean:.4f}  (range {min(fr):.4f}-{max(fr):.4f})   "
              f"random-form null {null:.5f}   {mean/max(null,1e-12):>8.1f}x", flush=True)

    print(f"\ntoy comparison (A2-3): trained 0.65-0.71 against a chance level of 0.49, "
          f"i.e. 1.4x")
    means = [v['identifiable_mean'] for v in out['layers'].values()]
    ratios = [v['ratio_to_null'] for v in out['layers'].values()]
    out['summary'] = {'mean_identifiable': sum(means) / len(means),
                      'mean_ratio_to_null': sum(ratios) / len(ratios)}
    print(f"bilin18: mean identifiable fraction {out['summary']['mean_identifiable']:.4f}, "
          f"mean {out['summary']['mean_ratio_to_null']:.0f}x its own null")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_identifiable_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
