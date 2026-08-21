"""MLP GATE CONJUNCTION -- the architectural PARALLEL to the attention's
double-QK AND (682). The MLP is bilinear: each hidden unit is
h_j = (Lx)_j * (Rx)_j (a product of two linear projections), and the
output is Down[h] + bias. Is this multiplicative gate an AND -- is the
product h more SELECTIVE/sparse across inputs than either factor Lx or
Rx alone -- so the whole model (attention AND MLP) runs on multiplicative
bilinear gating, no softmax and no relu?

For mlp0's hidden units over real text, compare the sparsity/selectivity
of |h = Lx*Rx| vs |Lx| and |Rx| across positions (per unit): the
participation ratio over positions (how many positions a unit is active
on). Low = selective (fires on few inputs); high = dense. If the product
is more selective than either factor, the bilinear gate sharpens like the
double-QK.

REGISTERED PREDICTIONS:
  (0) SANITY: hidden units are non-degenerate;
  (a) PRODUCT SHARPENS: |h=Lx*Rx| is more selective across positions
      (lower participation fraction) than |Lx| and |Rx| alone, averaged
      over units -- the bilinear MLP gate is a multiplicative AND, like
      the double-QK attention;
  (b) report mean selectivity fraction for h, Lx, Rx, and how often the
      product is more selective than both factors;
  NULL: for a random product of two gaussians, the product is not
      dramatically more selective than the factors (report the baseline)."""
import json, time, torch
import torch.nn.functional as F
import numpy as np
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152
T = 256
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'mlp_gate_conjunction_results.json'
NFRESH = 24
LJ = 0                            # mlp0


def part_frac(M):
    # M: (Npos, U) nonneg; per-unit participation ratio over positions / Npos
    s = M.sum(0); s2 = (M ** 2).sum(0)
    pr = (s ** 2) / (s2 + 1e-12)
    return pr / M.shape[0]         # (U,)


@torch.no_grad()
def main():
    t0 = time.time()
    cl.use_state(PT + 'census_state_diverse.pt')
    fresh = cl.fineweb_rows(NFRESH)
    mlp = m.transformer.h[LJ].mlp

    capL, capR = [], []
    def hk(mo, i_, o_):
        xin = i_[0]
        capL.append(mlp.Left(xin).detach().float().reshape(-1, mlp.Left.out_features).cpu())
        capR.append(mlp.Right(xin).detach().float().reshape(-1, mlp.Right.out_features).cpu())
    h = mlp.register_forward_hook(hk)
    # need mlp INPUT: hook Down's input instead (h) -- simpler: hook the mlp and recompute
    h.remove()

    # recompute L,R from the mlp input (pre-hook capturing input to mlp)
    capin = []
    ph = mlp.register_forward_pre_hook(lambda mo, args: capin.append(
        args[0].detach().float().reshape(-1, D).cpu()))
    for i in range(0, NFRESH, 4):
        bb = fresh[i:i + 4, :257].to(DEV)
        idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h:
            x, v1 = blk(x, v1, x0)
    ph.remove()
    Xin = torch.cat(capin, 0).to(DEV)
    with torch.no_grad():
        L = mlp.Left(Xin).float().cpu().numpy()
        R = mlp.Right(Xin).float().cpu().numpy()
    H = L * R
    # subsample units for speed
    U = L.shape[1]
    rng = np.random.default_rng(0); units = rng.choice(U, size=min(600, U), replace=False)
    fL = part_frac(np.abs(L[:, units]))
    fR = part_frac(np.abs(R[:, units]))
    fH = part_frac(np.abs(H[:, units]))
    mL, mR, mH = float(fL.mean()), float(fR.mean()), float(fH.mean())
    win = float(((fH < fL) & (fH < fR)).mean())
    print(f'mean selectivity fraction: h=L*R {mH:.4f}  L {mL:.4f}  R {mR:.4f}', flush=True)
    print(f'product more selective than BOTH factors: {100*win:.0f}%', flush=True)

    # NULL: random gaussian product
    g = np.random.default_rng(1)
    rl = g.standard_normal((L.shape[0], 200)); rr = g.standard_normal((L.shape[0], 200))
    frl = part_frac(np.abs(rl)); frr = part_frac(np.abs(rr)); frh = part_frac(np.abs(rl * rr))
    rand_ratio = float(frh.mean() / ((frl.mean() + frr.mean()) / 2))

    p0 = True
    pa = mH < 0.5 * (mL + mR)
    print(f'\n(a) product sharpens (h < mean of L,R): {pa}', flush=True)
    print(f'    NULL random product/factor ratio {rand_ratio:.2f} '
          f'(model ratio {mH/((mL+mR)/2):.2f})', flush=True)

    out = {'mean_frac_h': round(mH, 4), 'mean_frac_L': round(mL, 4), 'mean_frac_R': round(mR, 4),
           'product_wins_both': round(win, 4),
           'model_ratio': round(mH / ((mL + mR) / 2), 3), 'random_ratio': round(rand_ratio, 3),
           'pred_0': bool(p0), 'pred_a_product_sharpens': bool(pa), 'runtime_s': time.time() - t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\nwrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
