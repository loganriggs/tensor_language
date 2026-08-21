"""SUBSPACE STABILITY -- is 763's atom-instability actually ROTATIONAL ambiguity?
(atoms unstable across seeds, but the SPAN they cover stable). Train Down_0 weight-
action SAE with S seeds; compare, across seeds: (i) ATOM stability (best decoder-
cosine match, 763's 0.40) vs (ii) SUBSPACE stability (principal-angle overlap of the
top-r decoder directions). If subspace >> atom, the instability is rotational: the
subspace is a real model feature, the basis within it is arbitrary and seed-picked
-- which means the stability fix is to PIN a canonical basis in the subspace (rotate
for interpretability), not a better fit.

REGISTERED PREDICTIONS:
  (0) SANITY: atom stability reproduces 763 (~0.4);
  (a) ROTATIONAL: top-r decoder subspaces overlap MUCH more than atoms match --
      mean principal-angle cosine >= 0.8 at r=64 and >> atom-match 0.40 and >>
      random r-subspace overlap; the span is stable while the basis is not;
  (b) report subspace overlap vs r (64,128,256) + random-subspace baseline;
  NULL: two independent random r-subspaces overlap near the r/D floor."""
import json, time, sys, torch
import torch.nn.functional as F
import numpy as np
sys.path.insert(0, '/workspace/rspd')
import census_lib as cl
from bilin18_joint_removal import m, DEV

D = 1152; HID = 4608
PT = '/workspace/tensor_language/basis_aligned/bilinear_quotient/'
OUT = PT + 'subspace_stability_results.json'
NFIT = 48; P = 512; K = 32; NSEED = 4; RS = [64, 128, 256]


def topk(pre, k):
    val, idx = pre.topk(k, dim=1); z = torch.zeros_like(pre); z.scatter_(1, idx, F.relu(val)); return z


@torch.no_grad()
def capture(rows, n):
    cap = []
    h = m.transformer.h[0].mlp.Down.register_forward_hook(lambda mo, i_, o_: cap.append(i_[0].detach().float().reshape(-1, HID)))
    for i in range(0, n, 4):
        idx = rows[i:i+4, :257].to(DEV)[:, :-1].contiguous()
        x = F.rms_norm(m.transformer.wte(idx), (D,)); x0 = x; v1 = None
        for blk in m.transformer.h: x, v1 = blk(x, v1, x0)
    h.remove(); return torch.cat(cap, 0)


def train_sae(Xin, Ytrue, seed):
    torch.manual_seed(seed)
    Dm = (torch.randn(D, P, device=DEV)/np.sqrt(D)).requires_grad_(True)
    Em = (torch.randn(P, HID, device=DEV)/np.sqrt(HID)).requires_grad_(True)
    b = Ytrue.mean(0).clone().requires_grad_(True); opt = torch.optim.Adam([Dm, Em, b], lr=3e-3)
    for s in range(700):
        z = topk(Xin @ Em.T, K); loss = F.mse_loss(z @ Dm.T + b, Ytrue)
        opt.zero_grad(); loss.backward(); opt.step()
    return Dm.detach(), Em.detach(), b.detach()


def top_subspace(Dm, r):
    # top-r left singular directions of the decoder (the r-dim span the atoms most cover)
    U = torch.linalg.svd(Dm, full_matrices=False)[0]          # (D, P) -> (D, min)
    return U[:, :r]                                            # (D, r) orthonormal


def subspace_overlap(U, V):
    # mean cos of principal angles = mean singular value of U^T V
    s = torch.linalg.svdvals(U.T @ V)
    return float(s.mean())


@torch.no_grad()
def main():
    t0 = time.time(); cl.use_state(PT + 'census_state_diverse.pt')
    rows = cl.fineweb_rows(NFIT); g0 = capture(rows, NFIT)
    W0 = m.transformer.h[0].mlp.Down.weight.data.float().to(DEV); Y0 = g0 @ W0.T
    saes = []
    for s in range(NSEED):
        with torch.enable_grad(): saes.append(train_sae(g0, Y0, seed=s))

    # ATOM stability (reproduce 763)
    Dref = F.normalize(saes[0][0], dim=0)
    atom_match = []
    for Ds, _, _ in saes[1:]:
        atom_match.append(float((Dref.T @ F.normalize(Ds, dim=0)).abs().max(1).values.mean()))
    atom_stab = float(np.mean(atom_match))
    print(f'atom stability {atom_stab:.3f} (763 ~0.40)', flush=True)

    # SUBSPACE stability + random baseline
    res = {}
    g = torch.Generator(device=DEV).manual_seed(0)
    for r in RS:
        subs = [top_subspace(Ds, r) for Ds, _, _ in saes]
        ov = [subspace_overlap(subs[0], subs[i]) for i in range(1, NSEED)]
        # random r-subspace baseline (two independent random orthonormal r-frames in R^D)
        Rr1 = torch.linalg.qr(torch.randn(D, r, generator=g, device=DEV))[0]
        Rr2 = torch.linalg.qr(torch.randn(D, r, generator=g, device=DEV))[0]
        rand_ov = subspace_overlap(Rr1, Rr2)
        res[str(r)] = {'subspace_overlap': round(float(np.mean(ov)), 4), 'random_overlap': round(rand_ov, 4)}
        print(f'r={r:3d}: subspace overlap {np.mean(ov):.3f}  (random {rand_ov:.3f})', flush=True)

    ov64 = res['64']['subspace_overlap']
    pa = ov64 >= 0.8 and ov64 - atom_stab >= 0.3 and ov64 > 2*res['64']['random_overlap']
    null_ok = res['64']['random_overlap'] < 0.5
    out = {'n_seed': NSEED, 'atom_stability': round(atom_stab, 4), 'rs': RS, 'subspace': res,
           'pred_a_rotational': bool(pa), 'null_ok': bool(null_ok), 'runtime_s': time.time()-t0}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f'\n(a) ROTATIONAL (subspace stable >> atoms, >> random): {pa}; NULL random-subspace low: {null_ok}', flush=True)
    print(f'wrote {OUT} ({out["runtime_s"]:.0f}s)')


if __name__ == '__main__':
    main()
