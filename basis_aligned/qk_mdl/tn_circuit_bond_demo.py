"""Lessons 4 & 5 data.
L4 (minimal circuits / honest limit): two maps of equal size — a LOW-RANK one (breaks down: a few
   components reproduce it, chosen >> random) and a REDUNDANT full-rank one (won't break down:
   fidelity climbs ~linearly, chosen ~= random). The random-vs-chosen control is the diagnostic.
L5 (bonds / communication channel): two composed bilinear layers with a bottleneck BOND. Sweep the
   bond dimension and find the width below which behavior breaks — the channel's true capacity.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
rng = np.random.default_rng(0)


def eff_rank(sv):
    p = sv/sv.sum(); return float(np.exp(-(p*np.log(p+1e-12)).sum()))


# ===== L4: low-rank (decomposable) vs redundant (not) =====
d, N = 24, 4000
X = rng.standard_normal((N, d))
# low-rank map: reads only a rank-3 subspace of the input
Ql, _ = np.linalg.qr(rng.standard_normal((d, d)))
Mlow = Ql[:, :3] @ np.diag([3., 2., 1.5]) @ Ql[:, :3].T      # rank-3 symmetric map (d x d)
Ylow = X @ Mlow.T
# redundant map: full-rank ISOTROPIC (orthogonal) — every input direction equally read
Mfull, _ = np.linalg.qr(rng.standard_normal((d, d)))
Yfull = X @ Mfull.T


def rank_k_r2(X, Y, dirs):
    Xc = X - X.mean(0); Xp = Xc @ np.asarray(dirs).T
    W = np.linalg.lstsq(np.c_[Xp, np.ones(len(X))], Y, rcond=1e-8)[0]
    pred = np.c_[Xp, np.ones(len(X))] @ W
    return 1 - ((pred-Y)**2).sum()/((Y-Y.mean(0))**2).sum()


def chosen_dirs(M, k):                                        # top-k right singular vectors of the map
    _, _, Vt = np.linalg.svd(M, full_matrices=False); return Vt[:k]
def rand_dirs(k):
    Q, _ = np.linalg.qr(rng.standard_normal((d, d))); return Q[:k]

ks = [1, 2, 3, 4, 6, 9, 12, 18, 24]
low_chosen = [round(rank_k_r2(X, Ylow, chosen_dirs(Mlow, k)), 3) for k in ks]
full_chosen = [round(rank_k_r2(X, Yfull, chosen_dirs(Mfull, k)), 3) for k in ks]
low_random = [round(float(np.mean([rank_k_r2(X, Ylow, rand_dirs(k)) for _ in range(6)])), 3) for k in ks]
full_random = [round(float(np.mean([rank_k_r2(X, Yfull, rand_dirs(k)) for _ in range(6)])), 3) for k in ks]
svl = np.linalg.svd(Mlow, compute_uv=False); svf = np.linalg.svd(Mfull, compute_uv=False)
L4 = {'ks': ks, 'lowrank_chosen': low_chosen, 'lowrank_random': low_random,
      'redundant_chosen': full_chosen, 'redundant_random': full_random,
      'lowrank_eff_rank': round(eff_rank(svl**2), 2), 'redundant_eff_rank': round(eff_rank(svf**2), 2), 'd': d}
print(f"L4 low-rank: chosen {low_chosen} vs random {low_random} (eff-rank {L4['lowrank_eff_rank']})", flush=True)
print(f"L4 redundant: chosen {full_chosen} vs random {full_random} (eff-rank {L4['redundant_eff_rank']})", flush=True)

# ===== L5: bond dimension sweep in a two-layer bilinear model =====
din, dhid, dout = 24, 40, 6
Nb = 6000
Xb = torch.randn(Nb, din, device=DEV)
# teacher: a genuine 2-layer bilinear computation whose intermediate truly needs ~rbond_true dims
rbond_true = 4
Ut = torch.randn(rbond_true, din, device=DEV); Vt = torch.randn(rbond_true, din, device=DEV)
mid = (Xb @ Ut.T) * (Xb @ Vt.T)                        # (N, rbond_true) true bottleneck code
Wt2 = torch.randn(dout, rbond_true, device=DEV)
Yb = mid @ Wt2.T
Yb = (Yb - Yb.mean(0)) / Yb.std(0).clamp_min(1e-6)


def train_bond(rbond):
    A1 = nn.Linear(din, dhid, bias=False).to(DEV); B1 = nn.Linear(din, dhid, bias=False).to(DEV)
    down = nn.Linear(dhid, rbond, bias=False).to(DEV)  # squeeze to bond width
    up = nn.Linear(rbond, dout).to(DEV)                # consumer reads the bond
    opt = torch.optim.Adam(list(A1.parameters())+list(B1.parameters())+list(down.parameters())+list(up.parameters()), lr=3e-3)
    for step in range(3000):
        h = (A1(Xb)*B1(Xb)); pred = up(down(h))
        loss = F.mse_loss(pred, Yb); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = up(down(A1(Xb)*B1(Xb)))
        return 1 - ((pred-Yb)**2).sum().item()/((Yb-Yb.mean(0))**2).sum().item()


bonds = [1, 2, 3, 4, 6, 8, 12]
bond_r2 = [round(train_bond(rb), 3) for rb in bonds]
L5 = {'bond_widths': bonds, 'bond_r2': bond_r2, 'true_bond': rbond_true, 'hidden': dhid,
      'knee': next((b for b, r in zip(bonds, bond_r2) if r >= 0.95), bonds[-1])}
print(f"L5 bond sweep: {list(zip(bonds, bond_r2))} (true bond {rbond_true}, knee {L5['knee']})", flush=True)

json.dump({'L4': L4, 'L5': L5}, open('tn_circuit_bond_demo.json', 'w'), indent=1)
print('TN CIRCUIT BOND DEMO DONE', flush=True)
