"""KNOWN-ANSWER TEST FOR THE CP FITTER (tick 174). Four fitter iterations failed on real
cores (singular ALS, divergent ALS, dead-ReLU Adam, collapsed projected Adam). Before any
further real-data fitting: plant a core with KNOWN sparse nonneg archetypes and require
recovery.

Planted: m=512, R0=24 archetypes, each with 6 active features (positive loadings), weights
lambda log-spread over two decades, plus 1% dense noise. Core = sum lambda_r mu_r^(x3) + noise.
Candidates:
  A. Multiplicative updates (Lee-Seung style for nonneg symmetric CP): no step size,
     monotone for the surrogate. A <- A * (M1 (A o A)) / (A (A^T A)^2), eps-guarded.
  B. Projected Adam, lr = a0/20, init columns from the features of the top core entries.
Metric: greedy-matched mean max |cos| of recovered vs planted archetypes (gate 0.95), plus
relative Frobenius error. Winner gets adopted into qk_stage23.
"""
import sys
import torch
sys.path.insert(0, '/workspace/tensor_language/basis_aligned/qk_mdl')
torch.manual_seed(0)
DEV = 'cuda'
m, R0 = 512, 24
g = torch.Generator().manual_seed(5)

MU = torch.zeros(m, R0, device=DEV)
for r in range(R0):
    sup = torch.randperm(m, generator=g)[:6]
    MU[sup.to(DEV), r] = (0.3 + torch.rand(6, generator=g)).to(DEV)
MU = MU / MU.norm(dim=0, keepdim=True)
LAM = (10 ** (2 * torch.rand(R0, generator=g))).to(DEV)
core = torch.einsum('ar,br,cr->abc', MU * LAM ** (1 / 3), MU * LAM ** (1 / 3), MU * LAM ** (1 / 3))
core = core + 0.01 * core.norm() / m ** 1.5 * torch.randn(m, m, m, generator=g).to(DEV)
core = core / core.norm()
nrm2 = float((core ** 2).sum())
M1 = core.reshape(m, m * m)


def match(U):
    C = (U.T @ MU).abs()
    return float(C.max(0).values.mean())


def fit_mult(R, seed, iters=400):
    gg = torch.Generator().manual_seed(seed)
    A = (0.01 + torch.rand(m, R, generator=gg)).to(DEV) * 0.05
    eps = 1e-12
    for _ in range(iters):
        KR = (A[:, None, :] * A[None, :, :]).reshape(m * m, R)
        num = (M1 @ KR).clamp_min(0)
        den = A @ (A.T @ A) ** 2 + eps
        A = A * (num / den) ** (1 / 3)                        # cube-root damping for order 3
    lam = A.norm(dim=0).clamp_min(1e-12)
    U = A / lam[None, :]
    pred = torch.einsum('ar,br,cr->abc', A, A, A)
    rel = float((pred - core).norm()) / nrm2 ** 0.5
    return U, rel


def fit_padam(R, seed, iters=3000):
    gg = torch.Generator().manual_seed(seed)
    flat = core.reshape(-1)
    top = flat.abs().topk(R).indices
    A0 = torch.zeros(m, R, device=DEV)
    for r, t in enumerate(top.tolist()):
        a, b, c = t // (m * m), (t // m) % m, t % m
        A0[[a, b, c], r] = float(flat[t].clamp_min(1e-6)) ** (1 / 3)
    a0 = float(A0.max())
    A = (A0 + 0.05 * a0 * torch.rand(m, R, generator=gg).to(DEV)).requires_grad_(True)
    opt = torch.optim.Adam([A], lr=a0 / 20)
    for _ in range(iters):
        pred = torch.einsum('ir,jr,kr->ijk', A, A, A)
        loss = ((pred - core) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            A.clamp_(min=0)
    with torch.no_grad():
        lam = A.norm(dim=0).clamp_min(1e-12)
        U = (A / lam[None, :]).detach()
        rel = (float(loss) / nrm2) ** 0.5
    return U, rel


for name, fn in (('mult', fit_mult), ('padam', fit_padam)):
    for R in (24, 32):
        U, rel = fn(R, 0)
        print(f'{name} R={R}: matched-cos {match(U):.4f} rel-err {rel:.4f} '
              f'-> {"PASS" if match(U) > 0.95 else "FAIL"}', flush=True)
print('CP PLANTED DONE', flush=True)
