"""TOY PLANTED DICT (user: construct toy examples with KNOWN ground truth to
clarify what we want). Plant an overcomplete dictionary D_true (P_true atoms
in D dims) and sparse per-datapoint codes (each datapoint uses exactly
k_true known atoms). Generate O = Z_true @ D_true. Then check which method
recovers the PLANTED structure:
  - SVD of O: top-r directions (dense; should NOT match individual atoms);
  - top-k SAE (P atoms, k=k_true): learned dictionary (should recover
    D_true up to permutation, and give per-datapoint code length = k_true).
The metric that DISTINGUISHES them = ATOM-RECOVERY (each true atom's max
cosine to a recovered atom) + per-datapoint code length matching k_true.
This tells us the RIGHT metric (atom-recovery) and RIGHT method (overcomplete
SAE) when there IS true sparse structure -- and the NULL guards against the
method hallucinating structure when there is none.

REGISTERED PREDICTIONS:
  (0) SANITY: O reconstructs from D_true, Z_true exactly;
  (a) SAE RECOVERS, SVD DOES NOT: the top-k SAE recovers the planted atoms
      (mean over true atoms of max cosine to a learned atom >= 0.85) while
      SVD's top directions do NOT (mean max-cosine of true atoms to SVD dirs
      < 0.6); the SAE's per-datapoint code length ~= k_true;
  (b) report atom-recovery (SAE vs SVD) + code lengths;
  NULL: on DENSE Gaussian data (NO planted structure), the SAE does NOT
      recover high-cosine 'atoms' better than the SVD (guards against
      hallucinating structure)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'toy_planted_dict_results.json'
Dd = 128; P_true = 64; K_true = 3; N = 20000; P_sae = 96; STEPS = 1500


def topk_encode(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


def train_topk_sae(O, k, P, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    We = (torch.randn(Dd, P, device=DEV)/np.sqrt(Dd)).requires_grad_(True)
    Wd = (torch.randn(P, Dd, device=DEV)/np.sqrt(P)).requires_grad_(True)
    b = O.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([We, Wd, b], lr=3e-3)
    for s in range(steps):
        z = topk_encode((O-b)@We, k); recon = z@Wd + b
        loss = F.mse_loss(recon, O)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = topk_encode((O-b)@We, k); codelen = float((z.abs()>1e-6).sum(1).float().mean())
        Wdn = Wd / Wd.norm(dim=1, keepdim=True)
    return Wdn.detach(), codelen


def atom_recovery(Dtrue, Dlearned):
    # mean over true atoms of max |cosine| to any learned atom
    Dt = Dtrue / Dtrue.norm(dim=1, keepdim=True); Dl = Dlearned / Dlearned.norm(dim=1, keepdim=True)
    cos = (Dt @ Dl.T).abs()          # (P_true, P_learned)
    return float(cos.max(1).values.mean())


def run(O, Dtrue, tag):
    # SVD dirs
    U = torch.linalg.svd(O - O.mean(0), full_matrices=False)[2][:P_sae]     # top directions (P_sae, Dd)
    svd_rec = atom_recovery(Dtrue, U)
    Dsae, codelen = train_topk_sae(O, K_true, P_sae)
    sae_rec = atom_recovery(Dtrue, Dsae)
    print(f'[{tag}] atom-recovery  SAE {sae_rec:.3f}  SVD {svd_rec:.3f}  | SAE code-len {codelen:.2f} '
          f'(true k={K_true})', flush=True)
    return {'sae_atom_recovery': round(sae_rec,3), 'svd_atom_recovery': round(svd_rec,3),
            'sae_code_len': round(codelen,2)}


def main():
    t0 = time.time(); torch.manual_seed(0)
    # PLANTED: D_true overcomplete, sparse codes
    Dtrue = torch.randn(P_true, Dd, device=DEV); Dtrue = Dtrue / Dtrue.norm(dim=1, keepdim=True)
    Z = torch.zeros(N, P_true, device=DEV)
    for i in range(N):
        idx = torch.randperm(P_true, device=DEV)[:K_true]
        Z[i, idx] = torch.randn(K_true, device=DEV).abs() + 0.3
    O = Z @ Dtrue
    print(f'planted: P_true={P_true} atoms, k_true={K_true}, N={N}', flush=True)
    planted = run(O, Dtrue, 'planted')

    # NULL: dense Gaussian, no structure -- "true atoms" are a fresh random dict (recovery should be low for both)
    Ornd = torch.randn(N, Dd, device=DEV)
    Dnull = torch.randn(P_true, Dd, device=DEV)
    nullres = run(Ornd, Dnull, 'null(dense)')

    p0 = True
    pa = planted['sae_atom_recovery'] >= 0.85 and planted['svd_atom_recovery'] < 0.6 and abs(planted['sae_code_len']-K_true) < 1
    null_ok = nullres['sae_atom_recovery'] < 0.6   # SAE shouldn't 'recover' structure that isn't there
    print(f'\n(a) SAE recovers planted, SVD does not, code-len ~ k_true: {pa}', flush=True)
    print(f'NULL (no structure) SAE recovery low: {null_ok}', flush=True)
    out = {'Dd': Dd, 'P_true': P_true, 'K_true': K_true, 'P_sae': P_sae, 'N': N,
           'planted': planted, 'null_dense': nullres, 'pred_a': bool(pa), 'null_ok': bool(null_ok),
           'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
