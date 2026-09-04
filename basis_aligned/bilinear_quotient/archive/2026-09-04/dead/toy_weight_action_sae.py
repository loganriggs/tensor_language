"""TOY WEIGHT-ACTION SAE (user: sparse-code the WEIGHT'S ACTION, faithful --
novel vs activation SAEs). Factor a weight W = D @ E overcomplete (D: Dout x
P, E: P x Din, P > rank) so that (i) D@E reconstructs W EXACTLY (FAITHFUL,
like A-SVD at full rank) and (ii) the per-datapoint codes E@x are SPARSE.
Objective: min ||W - D@E||_F^2 + lambda * mean_i ||E x_i||_1. This decomposes
the WEIGHT into an overcomplete sparse form -- faithful to W (not a lossy fit
to activations) AND sparse on data.

Ground-truth toy: plant W = D_true @ E_true (P_true rank-1 atoms) and data X
whose codes Z=E_true@X are sparse (k_true atoms per datapoint). Check the
method (a) is FAITHFUL (||W-DE||/||W|| ~ 0), (b) RECOVERS the planted atoms
(D_true up to permutation), (c) gives SPARSE codes ~ k_true. Compare to
A-SVD (dense response-SVD) which is faithful but NOT sparse/overcomplete.

REGISTERED PREDICTIONS:
  (0) SANITY: planted codes Z=E_true@X have ~k_true nonzeros per datapoint;
  (a) FAITHFUL + SPARSE + RECOVERS: the weight-action SAE reaches low weight-
      reconstruction error (||W-DE||/||W|| < 0.1), recovers the atoms (D-atom
      recovery >= 0.85), and its codes are SPARSE (effective L0 near k_true,
      << P); A-SVD recovers atoms poorly (dense);
  (b) report faithfulness, atom-recovery (SAE vs A-SVD), code sparsity;
  NULL: with lambda=0 (no sparsity pressure) the codes are DENSE (effective
      L0 >> k_true) -- the sparsity comes from the L1 objective."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'toy_weight_action_sae_results.json'
Dout = 96; Din = 128; P_true = 48; K_true = 3; N = 8000; P = 72; STEPS = 3000; LAM = 3e-3


def atom_recovery(Dtrue, Dlearn):
    Dt = Dtrue/Dtrue.norm(dim=0, keepdim=True); Dl = Dlearn/Dlearn.norm(dim=0, keepdim=True)
    cos = (Dt.T @ Dl).abs()                 # (P_true, P)
    return float(cos.max(1).values.mean())


def eff_l0(codes, thresh=0.05):
    # mean nonzeros per datapoint (relative to per-datapoint max)
    a = codes.abs(); m = a.max(1, keepdim=True).values.clamp_min(1e-9)
    return float((a > thresh*m).float().sum(1).mean())


def train_wa_sae(W, X, P, lam, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    D = (torch.randn(Dout, P, device=DEV)/np.sqrt(Dout)).requires_grad_(True)
    E = (torch.randn(P, Din, device=DEV)/np.sqrt(Din)).requires_grad_(True)
    opt = torch.optim.Adam([D, E], lr=5e-3)
    for s in range(steps):
        recon_w = D @ E
        wloss = F.mse_loss(recon_w, W)
        codes = X @ E.T                      # (N, P)
        sparse = codes.abs().mean()
        loss = wloss + lam*sparse
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        faith = float((W - D@E).norm()/W.norm())
        codes = X @ E.T
    return D.detach(), E.detach(), faith, codes


def main():
    t0 = time.time(); torch.manual_seed(0)
    Dtrue = torch.randn(Dout, P_true, device=DEV); Dtrue = Dtrue/Dtrue.norm(dim=0, keepdim=True)
    Etrue = torch.randn(P_true, Din, device=DEV); Etrue = Etrue/Etrue.norm(dim=1, keepdim=True)
    Wtrue = Dtrue @ Etrue
    # data: sparse codes Z, X = Z @ pinv(Etrue).T so Etrue@x = z
    Z = torch.zeros(N, P_true, device=DEV)
    for i in range(N):
        idx = torch.randperm(P_true, device=DEV)[:K_true]; Z[i, idx] = torch.randn(K_true, device=DEV).abs()+0.3
    Epinv = torch.linalg.pinv(Etrue)          # (Din, P_true)
    X = Z @ Epinv.T                            # (N, Din); Etrue @ x_i = z_i
    print(f'planted: P_true={P_true}, k_true={K_true}; check codes L0 {eff_l0(Z):.2f}', flush=True)

    D, E, faith, codes = train_wa_sae(Wtrue, X, P, LAM)
    rec_sae = atom_recovery(Dtrue, D); l0 = eff_l0(codes)
    # A-SVD of the response W@X.T (dense)
    resp = (Wtrue @ X.T).T                      # (N, Dout) output
    Usvd = torch.linalg.svd(resp - resp.mean(0), full_matrices=False)[2]  # right dirs (Dout)... use left
    U = torch.linalg.svd(Wtrue, full_matrices=False)[0][:, :P]            # weight-SVD output dirs
    rec_svd = atom_recovery(Dtrue, U)
    # null: lambda=0
    _, _, faith0, codes0 = train_wa_sae(Wtrue, X, P, 0.0)
    l0_dense = eff_l0(codes0)

    print(f'weight-action SAE: faithfulness ||W-DE||/||W|| {faith:.3f}  atom-recovery {rec_sae:.3f}  '
          f'code-L0 {l0:.2f} (true k={K_true})', flush=True)
    print(f'A-SVD (weight-SVD) atom-recovery {rec_svd:.3f} (dense)', flush=True)
    print(f'NULL lambda=0: code-L0 {l0_dense:.2f} (dense, >> k_true)', flush=True)
    pa = faith < 0.1 and rec_sae >= 0.85 and l0 < 2*K_true and rec_sae > rec_svd + 0.2
    null_ok = l0_dense > 2*l0
    print(f'\n(a) faithful+sparse+recovers, beats A-SVD: {pa}; NULL lambda=0 dense: {null_ok}', flush=True)
    out = {'Dout': Dout, 'Din': Din, 'P_true': P_true, 'K_true': K_true, 'P': P, 'lambda': LAM,
           'faithfulness': round(faith,4), 'sae_atom_recovery': round(rec_sae,3),
           'code_l0': round(l0,2), 'svd_atom_recovery': round(rec_svd,3), 'null_l0_dense': round(l0_dense,2),
           'pred_a': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
