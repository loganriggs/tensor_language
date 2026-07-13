"""Tier 1.1 (correlated pair, isotropic W: accepted merge + novelty detector) and
Tier 1.2 (correlated pair, anisotropic W: mechanism-side rescue vs activation SAE).

Run: python -m mechdecomp.tier1
"""

import torch

from .estep import solve_codes
from .objective import predict, r2
from .tier0 import train
from .toys import correlated_pair_dgp

torch.set_default_dtype(torch.float64)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def decorrelated_points(E, n=400, seed=9):
    """Test points with feature 0 XOR feature 1 (never seen in training at rho=1)."""
    g = torch.Generator().manual_seed(seed)
    Ec = E.cpu()
    X = torch.zeros(Ec.shape[0], n, dtype=Ec.dtype)
    for i in range(n):
        j = i % 2
        alpha = torch.rand(1, generator=g) + 0.5
        back = 2 + torch.randperm(8, generator=g)[:2]
        ab = torch.rand(2, generator=g) + 0.5
        X[:, i] = alpha * Ec[:, j] + (Ec[:, back] * ab).sum(1)
    return X.to(E.device)


def recon_err(W, D, X, lam):
    C = solve_codes(W, D, X, lam)
    return ((predict(W, D, C, X) - W @ X) ** 2).sum(0)


def auroc(pos, neg):
    lab = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    score = torch.cat([pos, neg])
    order = torch.argsort(score, descending=True)
    lab = lab[order]
    tp = torch.cumsum(lab, 0)
    fp = torch.cumsum(1 - lab, 0)
    tpr = tp / pos.numel()
    fpr = fp / neg.numel()
    return torch.trapz(tpr, fpr).item()


def sae_train(X, m=40, lam=3e-3, steps=4000, lr=1e-3, seed=0):
    """Vanilla ReLU SAE on activations x (baseline for 1.2)."""
    g = torch.Generator().manual_seed(seed)
    d = X.shape[0]
    We = (torch.randn(m, d, generator=g) / d ** 0.5).to(X).requires_grad_(True)
    be = torch.zeros(m, dtype=X.dtype, device=X.device, requires_grad=True)
    Wd = (torch.randn(d, m, generator=g) / m ** 0.5).to(X).requires_grad_(True)
    opt = torch.optim.Adam([We, be, Wd], lr=lr)
    for t in range(steps):
        z = torch.relu(We @ X + be[:, None])
        loss = ((Wd @ z - X) ** 2).sum() / X.shape[1] + lam * z.abs().sum() / X.shape[1]
        opt.zero_grad(); loss.backward(); opt.step()
    return (Wd / Wd.norm(dim=0, keepdim=True).clamp_min(1e-9)).detach()


def main():
    lam = 0.05
    # ---- Tier 1.1: isotropic W, rho = 1 → expect merge + novelty spike ----
    W, X, E, S = correlated_pair_dgp(rho=1.0, anisotropic=False, device=DEV)
    W, X, E = W.double(), X.double(), E.double()
    D, C = train(W, X, m=40, lam=lam, verbose=False)
    cos = (E[:, :2].T @ D).abs()
    same_atom = int(cos[0].argmax()) == int(cos[1].argmax())
    merged_dir = (E[:, 0] + E[:, 1]) / 2 ** 0.5
    cos_merged = float((merged_dir @ D[:, cos[0].argmax()]).abs())
    print(f"1.1 merge: pair maps to same atom: {same_atom}; cos to (e0+e1)/√2: {cos_merged:.3f}; atoms {D.shape[1]}")
    Xo = decorrelated_points(E)
    err_ood = recon_err(W, D, Xo, lam)
    err_id = recon_err(W, D, X[:, :400], lam)
    a = auroc(err_ood, err_id)
    print(f"1.1 novelty: OOD err median {err_ood.median():.4f} vs ID {err_id.median():.4f}; AUROC {a:.3f} (gate: high)")

    # ---- Tier 1.2: anisotropic W → expect rescue (two atoms) ----
    W2, X2, E2, S2 = correlated_pair_dgp(rho=1.0, anisotropic=True, device=DEV)
    W2, X2, E2 = W2.double(), X2.double(), E2.double()
    D2, C2 = train(W2, X2, m=40, lam=lam, verbose=False)
    cos2 = (E2[:, :2].T @ D2).abs()
    a0, a1 = int(cos2[0].argmax()), int(cos2[1].argmax())
    print(f"1.2 rescue: e0→atom{a0} (cos {cos2[0].max():.3f}), e1→atom{a1} (cos {cos2[1].max():.3f}); "
          f"distinct: {a0 != a1}")

    # SAE baseline on activations (should merge the pair)
    Dsae = sae_train(X2.float(), m=40).double()
    cs = (E2[:, :2].T @ Dsae).abs()
    s0, s1 = int(cs[0].argmax()), int(cs[1].argmax())
    print(f"1.2 SAE baseline: e0 best-cos {cs[0].max():.3f} (unit {s0}), e1 best-cos {cs[1].max():.3f} "
          f"(unit {s1}); distinct: {s0 != s1}; "
          f"merged-dir max cos {(((E2[:,0]+E2[:,1])/2**0.5) @ Dsae).abs().max():.3f}")


if __name__ == "__main__":
    main()
