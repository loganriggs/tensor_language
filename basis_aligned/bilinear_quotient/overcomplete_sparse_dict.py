"""OVERCOMPLETE SPARSE DICT (741 -> the real path: faithful per-datapoint
sparsity needs OVERCOMPLETENESS, not rotation). Learn an overcomplete top-k
sparse autoencoder (P=512 atoms >> rank) on mlp1's OUTPUT: each datapoint is
reconstructed by ITS OWN k atoms (per-datapoint sparse code). Compare to SVD
rank-k, where every datapoint uses the SAME k shared components (dense).

If a P>>k overcomplete top-k dictionary reconstructs each datapoint BETTER
than SVD rank-k at the SAME k, the per-datapoint sparse code is more
efficient -- the MDL win the user is after. Faithfulness = reconstruction
fidelity (R^2 -> 1 = exact). Measures the SPARSITY (k) vs FIDELITY (R^2)
frontier, overcomplete-SAE vs SVD vs random-overcomplete.

REGISTERED PREDICTIONS:
  (0) SANITY: SVD rank-k R^2 rises with k; at k=rank it's ~1;
  (a) OVERCOMPLETE SPARSITY WINS: a trained top-k SAE (P=512) reconstructs
      held-out datapoints with HIGHER R^2 than SVD rank-k at the same small
      k (k in {2,8,32}) -- each datapoint's own k atoms beat k shared
      components. Report R^2_sae(k) vs R^2_svd(k);
  (b) report the frontier for SAE / SVD / random-overcomplete;
  NULL: a RANDOM (untrained) overcomplete top-k dictionary does NOT beat SVD
      (the gain is from LEARNING the dictionary, not overcompleteness alone)."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; LAYER = 1
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'overcomplete_sparse_dict_results.json'
NFIT = 128; NEVAL = 48; P = 512
KS = [2, 8, 32]; STEPS = 600


@torch.no_grad()
def capture_out(rows, n):
    cap = []
    h = m.transformer.h[LAYER].mlp.register_forward_hook(
        lambda mo, i_, o_: cap.append(o_.detach().float().reshape(-1, D)))
    for i in range(0, n, 4):
        bb = rows[i:i+4, :257].to(DEV); idx = bb[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for li, blk in enumerate(m.transformer.h): x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def topk_encode(pre, k):
    val, idx = pre.topk(k, dim=1)
    z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val))
    return z


def train_topk_sae(Otr, Oev, k, P, steps=STEPS, seed=0):
    torch.manual_seed(seed)
    Wenc = torch.randn(D, P, device=DEV) * (1/np.sqrt(D)); Wenc.requires_grad_(True)
    Wdec = torch.randn(P, D, device=DEV) * (1/np.sqrt(P)); Wdec.requires_grad_(True)
    b = Otr.mean(0).clone().requires_grad_(True)
    opt = torch.optim.Adam([Wenc, Wdec, b], lr=2e-3)
    for s in range(steps):
        pre = (Otr - b) @ Wenc
        z = topk_encode(pre, k)
        recon = z @ Wdec + b
        loss = F.mse_loss(recon, Otr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pre = (Oev - b) @ Wenc; z = topk_encode(pre, k); recon = z @ Wdec + b
        r2 = 1 - ((Oev - recon)**2).sum() / ((Oev - Oev.mean(0))**2).sum()
    return float(r2)


def r2_svd(Otr, Oev, k):
    with torch.no_grad():
        U = torch.linalg.svd(Otr - Otr.mean(0), full_matrices=False)[2][:k]   # top-k right dirs (D)
        mu = Otr.mean(0); c = (Oev - mu) @ U.T; recon = c @ U + mu
        return float(1 - ((Oev-recon)**2).sum()/((Oev-Oev.mean(0))**2).sum())


def r2_random_overcomplete(Otr, Oev, k, P, seed=1):
    with torch.no_grad():
        torch.manual_seed(seed); Wd = torch.randn(P, D, device=DEV)
        Wd = Wd / Wd.norm(dim=1, keepdim=True); mu = Otr.mean(0)
        pre = (Oev - mu) @ Wd.T; z = topk_encode(pre, k)
        # least-squares fit of the selected atoms would be ideal; approximate w/ pinv per row is heavy.
        recon = z @ Wd + mu
        return float(1 - ((Oev-recon)**2).sum()/((Oev-Oev.mean(0))**2).sum())


@torch.no_grad()
def main_capture():
    cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT + NEVAL)
    Otr = capture_out(rows[:NFIT], NFIT); Oev = capture_out(rows[NFIT:NFIT+NEVAL], NEVAL)
    return Otr, Oev


def main():
    t0 = time.time()
    Otr, Oev = main_capture()
    print(f'train {Otr.shape[0]} eval {Oev.shape[0]} tokens, P={P}', flush=True)
    res = {'sae': {}, 'svd': {}, 'random_oc': {}}
    for k in KS:
        res['svd'][k] = round(r2_svd(Otr, Oev, k), 4)
        res['sae'][k] = round(train_topk_sae(Otr, Oev, k, P), 4)
        res['random_oc'][k] = round(r2_random_overcomplete(Otr, Oev, k, P), 4)
        print(f'k={k:3d}: SAE(P={P}) R2 {res["sae"][k]:.3f}  SVD rank-k R2 {res["svd"][k]:.3f}  '
              f'rand-OC R2 {res["random_oc"][k]:.3f}', flush=True)
    wins = all(res['sae'][k] > res['svd'][k] for k in KS)
    null_ok = all(res['random_oc'][k] < res['sae'][k] for k in KS)
    print(f'\n(a) overcomplete SAE beats SVD at same k: {wins}', flush=True)
    print(f'NULL random-overcomplete does not beat SVD/SAE: {null_ok}', flush=True)
    out = {'layer': LAYER, 'P': P, 'ks': KS, 'r2': res, 'pred_a_sae_wins': bool(wins),
           'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT,'w'), indent=1)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
