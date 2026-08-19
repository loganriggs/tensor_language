"""Phase D: the weights-first theory pass, scored against the measurements.

The bilinear MLP is a third-order tensor

    T = sum_j  Down_{:,j} (x) Left_{j,:} (x) Right_{j,:}      (output, in, in)

and linear algebra gives its output-mode structure in closed form. The mode-1 unfolding
T_(1) has Gram

    G_plain = Down [ (L L^T) o (R R^T) ] Down^T               (o = Hadamard)

whose top eigenvectors are the HOSVD output factor: the weight-only prediction of
"which output directions the layer is built around". The program's validated metric
(§7: measure weights in the geometry of the realised input) upgrades this with ONE data
statistic, the input second moment S:

    G_lam  = Down [ (L S L^T) o (R S R^T) ] Down^T

which is the output Gram of the layer AS A FUNCTION ON THE DATA -- derivable by
Isserlis if x were Gaussian with second moment S, and exactly the Λ-weighted object
bq_common has used since the toys.

The test: for layers 0, 1, 16, 17, how well do the top eigenvectors of G_plain (pure
weights) and G_lam (weights + one matrix of data) predict
    (a) the empirical PCA basis of the layer's output,
    (b) the Shapley-leading causal direction?
Scored as subspace energy and leader cosine, with a random-basis baseline. This
quantifies "how much of the pipeline could have been known without running the model",
which is the honest version of making the most of the weight-based part.

Also D2: at layer 1, the per-head folded operators B_h = Wp_h^T M_d0 Wp_h (128x128,
pure weights) -- does ||B_h|| rank head 4 first, as the interchange measurement (79%)
says it should?

And D4 (reported, not computed): the verified surrogate as a tensor network --
parameter and contraction-cost accounting for the compressed layer descriptions.
"""

import json
import sys
import time

import torch

sys.path.insert(0, '/workspace/tensor_language/basis_aligned/bilinear_quotient')
from bilin18_joint_removal import fwd, orth, m, FW, DEV
from bilin18_identifiable import form_for_direction

D = 1152
LAYERS = (0, 1, 16, 17)
OUT = ('/workspace/tensor_language/basis_aligned/bilinear_quotient/'
       'bilin18_theory_pass_results.json')


@torch.no_grad()
def collect_io(li, n_seq=60):
    """MLP input (xhat) and output (mo) for layer li over n_seq sequences."""
    ins, outs = [], []
    hs = []

    def hook(mod, inp, outp):
        ins.append(inp[0].detach().reshape(-1, D).float())
        outs.append(outp.detach().reshape(-1, D).float())

    h = m.transformer.h[li].mlp.register_forward_hook(hook)
    for i in range(0, n_seq, 6):
        b = FW[i:i + 6, :513].to(DEV)
        m(b[:, :-1].contiguous(), b[:, 1:].contiguous())
    h.remove()
    return torch.cat(ins), torch.cat(outs)


def out_gram(mlp, S=None):
    L = mlp.Left.weight.detach().float()      # (4608, 1152)
    R = mlp.Right.weight.detach().float()
    Dw = mlp.Down.weight.detach().float()     # (1152, 4608)
    if S is None:
        K = (L @ L.T) * (R @ R.T)
    else:
        LS = L @ S
        RS = R @ S
        K = (LS @ L.T) * (RS @ R.T)
    return Dw @ K @ Dw.T


def energy(Qsub, W8):
    """mean energy of columns of W8 inside span(Qsub)."""
    return float((Qsub.T @ W8).pow(2).sum() / W8.shape[1])


def main():
    t0 = time.time()
    out = {'layers': {}}
    shap_leaders = {}
    # Shapley leader directions, from the recorded batteries (refit bases)
    lead_src = {0: ('bilin18_layer0_battery_results.json', 'b1'),
                16: ('bilin18_layer16_battery_results.json', 'b1')}

    print('== D1: can weights (+ one data matrix) predict the empirical structure? ==\n')
    print(f"  {'layer':>5} {'basis':>10} {'energy in top-8 of G':>21} "
          f"{'leader |cos| best':>18} {'leader energy in top-8':>23}")
    for li in LAYERS:
        mlp = m.transformer.h[li].mlp
        X, Y = collect_io(li)
        S = X.T @ X / X.shape[0]
        Yc = Y - Y.mean(0)
        _, _, Vh = torch.linalg.svd(Yc, full_matrices=False)
        Qd = orth(Vh[:8].T)                    # empirical output basis (top 8)
        # Shapley leader
        if li in lead_src:
            rec = json.load(open(lead_src[li][0]))
            phi = torch.tensor(rec['b1']['phi'])
            Q32 = orth(torch.linalg.svd(Yc, full_matrices=False)[2][:32].T)
            d0 = Q32[:, int(phi.argmax())]
        else:
            d0 = Qd[:, 0]
        shap_leaders[li] = d0

        Gp = out_gram(mlp)
        Gl = out_gram(mlp, S.double().float())
        rows = {}
        g = torch.Generator(device=DEV).manual_seed(li)
        Qr = orth(torch.randn(D, 8, device=DEV, generator=g))
        for tag, G in (('plain', Gp), ('lam', Gl), ('random', None)):
            if G is None:
                W8 = Qr
            else:
                ev, U = torch.linalg.eigh(G)
                W8 = U[:, ev.argsort(descending=True)[:8]]
            e = energy(W8, Qd)                 # empirical basis inside weight span
            cos = float((W8.T @ d0).abs().max())
            elead = float((W8.T @ d0).pow(2).sum())
            rows[tag] = {'energy_top8': e, 'leader_best_cos': cos,
                         'leader_energy': elead}
            print(f"  {li:>5} {tag:>10} {e:>21.3f} {cos:>18.3f} {elead:>23.3f}",
                  flush=True)
        out['layers'][li] = rows
        print()

    # ---- D2: per-head folded operators at layer 1, pure weights ----
    print('== D2: does weight algebra alone rank head 4 first at layer 1? ==')
    mlp1 = m.transformer.h[1].mlp
    d0 = shap_leaders.get(1)
    if d0 is None:
        X, Y = collect_io(1)
        Yc = Y - Y.mean(0)
        _, _, Vh = torch.linalg.svd(Yc, full_matrices=False)
        d0 = orth(Vh[:32].T)[:, 0]
    M = form_for_direction(mlp1, d0.float()).float()
    Wp = m.transformer.h[1].attn.c_proj.weight.detach().float()   # (D, D)
    norms = []
    for h in range(9):
        Ph = Wp[:, h * 128:(h + 1) * 128]      # (D, 128) writes of head h
        Bh = Ph.T @ M @ Ph
        norms.append(float(Bh.norm()))
    tot = sum(n ** 2 for n in norms)
    shares = [n ** 2 / tot for n in norms]
    rank_pred = int(torch.tensor(norms).argmax())
    out['d2'] = {'head_fro_norms': norms, 'head_shares_sq': shares,
                 'predicted_head': rank_pred,
                 'measured_interchange_shares': 'head4=0.791 (bilin18_interchange)'}
    for h in range(9):
        bar = '#' * int(40 * shares[h] / max(shares))
        print(f'  head {h}: weight-only share {100*shares[h]:5.1f}%  {bar}')
    print(f'  -> weights alone rank head {rank_pred} first; the interchange '
          f'measurement said head 4 at 79%')

    # ---- D4: tensor-network accounting (closed form, no runs) ----
    out['d4'] = {
        'full_layer': {'params': 3 * D * 4608, 'flops_per_token': 3 * D * 4608},
        'leader_surrogate_l1': {'network': 'z=u.x ; c0=a z^2+b ; write=c0 d0',
                                'params': 2 * D + 2,
                                'flops_per_token': 2 * D + 3},
        'layer16_replacement': {'network': '4 dirs x rank-2 forms (S8/S10)',
                                'params': 13832, 'flops_per_token': 4 * (2 * D + 3) + 4 * D},
    }
    fl = out['d4']['full_layer']['flops_per_token']
    print(f"\n== D4: tensor-network accounting ==")
    print(f"  full bilinear layer: {out['d4']['full_layer']['params']:,} params, "
          f"~{fl/1e6:.1f}M flops/token")
    print(f"  layer-1 leader surrogate: {2*D+2:,} params, {2*D+3:,} flops/token "
          f"({fl/(2*D+3):,.0f}x cheaper)")

    out['runtime_s'] = time.time() - t0
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
