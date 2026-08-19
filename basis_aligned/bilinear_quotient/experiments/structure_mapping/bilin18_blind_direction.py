"""Why are bilin18's mid-layer interaction forms BELOW chance at small sample counts?

`bilin18_identifiable_power.py` found the identifiable fraction of layers 5 and 13
sitting at 0.10x and 0.19x chance at N=250, climbing to ~1x only by N=8000, while the
unrelated-form null sits flat at 1.00x throughout. Below chance is not noise and is not
"no structure" -- an unrelated form gets exactly chance. It means the form has LESS
mass on the data's leading directions than a random matrix does, i.e. it is actively
orthogonal to them.

The same run measured why that is possible: the Gram of x x^T at layers 5 and 13 has
participation ratio 1.3 and 2.1. One direction carries nearly all the second-moment
mass of the rms-normed MLP input. So the prediction is concrete and falsifiable:

    the MLP's quadratic form very nearly ANNIHILATES that dominant direction.

Test: take v = the top eigenvector of the input second moment. For an output direction
d with form M_d, compare the curvature along v,

    |v^T M_d v|   against   the same quantity for a random unit direction,

and against a random symmetric matrix of the same Frobenius norm evaluated along the
same v (which is the correct null -- it asks "is the form small along v specifically",
not "is v special"). A blind direction shows up as a ratio far below 1.

If it holds, it is a mechanism, not a curiosity: the residual stream carries a large
always-on component, and the bilinear MLP is built to not see it. That also explains
the sample-starvation -- the directions the MLP does use are the ones the data visits
rarely, which is exactly the regime where 4000 samples buy nothing.
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
N_SAMPLES = 8000
LAYERS = (0, 1, 3, 5, 7, 9, 11, 13, 15, 16, 17)
N_DIRS = 16


def main():
    t0 = time.time()
    model, cfg = load_elriggs('bilin18', device=DEV)
    tokens = torch.load('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
                        'bilin18_eval_tokens.pt')
    X = mlp_inputs(model, tokens, LAYERS, N_SAMPLES)
    g = torch.Generator().manual_seed(0)
    out = {'layers': {}}

    print('== is the residual stream dominated by one direction, and is the MLP blind '
          'to it? ==\n')
    print(f"  {'layer':>5}  {'top-PC share':>12}  {'|v.xhat| mean':>13}  "
          f"{'curvature along v':>18}  {'vs random dir':>13}  {'vs random form':>14}")
    for li in LAYERS:
        Xi = X[li].to(DEV)
        Xn = Xi / Xi.norm(dim=1, keepdim=True)
        S = Xn.T @ Xn / Xn.shape[0]
        ev, evec = torch.linalg.eigh(S)
        v = evec[:, -1]
        share = float(ev[-1] / ev.sum())
        align = float((Xn @ v).abs().mean())

        mlp = model.transformer.h[li].mlp
        cv, cr, cn = [], [], []
        for _ in range(N_DIRS):
            d = torch.randn(cfg['n_embd'], generator=g).to(DEV)
            M = form_for_direction(mlp, d / d.norm())
            fro = M.norm()
            u = torch.randn(cfg['n_embd'], generator=g).to(DEV).double()
            u = u / u.norm()
            # curvature along v, and along a random direction, both scaled by ||M||_F
            cv.append(float((v.double() @ M @ v.double()).abs() / fro))
            cr.append(float((u @ M @ u).abs() / fro))
            # a random symmetric form of the same norm, along the SAME v
            A = torch.randn(cfg['n_embd'], cfg['n_embd'], generator=g).double().to(DEV)
            A = 0.5 * (A + A.T)
            cn.append(float((v.double() @ A @ v.double()).abs() / A.norm()))
        mv, mr, mn = [sum(c) / len(c) for c in (cv, cr, cn)]
        out['layers'][li] = {'top_pc_share': share, 'mean_abs_align': align,
                             'curv_along_v': mv, 'curv_random_dir': mr,
                             'curv_random_form_along_v': mn,
                             'ratio_vs_random_dir': mv / max(mr, 1e-30),
                             'ratio_vs_random_form': mv / max(mn, 1e-30)}
        print(f"  {li:>5}  {share:>12.3f}  {align:>13.3f}  {mv:>18.2e}  "
              f"{mv/max(mr,1e-30):>12.2f}x  {mv/max(mn,1e-30):>13.2f}x", flush=True)

    rs = [v['ratio_vs_random_form'] for v in out['layers'].values()]
    sh = [v['top_pc_share'] for v in out['layers'].values()]
    out['summary'] = {'mean_ratio_vs_random_form': sum(rs) / len(rs),
                      'n_blind': sum(r < 0.5 for r in rs), 'n_layers': len(rs),
                      'max_top_pc_share': max(sh)}
    print(f"\n  'top-PC share' is how much of the normalised input's second moment the "
          f"single leading\n  direction v carries; 'curvature along v' is |v^T M v| / "
          f"||M||_F for the MLP's own forms.")
    print(f"\nSUMMARY: {out['summary']['n_blind']}/{out['summary']['n_layers']} layers "
          f"have curvature along v below half what an equally-sized random form gets "
          f"there;\n  mean ratio {out['summary']['mean_ratio_vs_random_form']:.2f}x, "
          f"largest top-PC share {out['summary']['max_top_pc_share']:.3f}")

    out['runtime_s'] = time.time() - t0
    p = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
         'bilin18_blind_direction_results.json')
    with open(p, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {p} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
