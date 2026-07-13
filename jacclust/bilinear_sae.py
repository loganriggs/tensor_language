"""tick 42 (Logan): BILINEAR SAE on the SECANT (Thomas Dooms style, two DIFFERENT inputs x^+ and y).

Reconstruct the per-token secant M = y x^+  (x^+ = x/||x||^2, the min-norm W with Wx=y) as a sparse sum of
RANK-1 atoms p_i q_i^T. Bilinear encoder z_i = (q_i·x^+)(p_i·y) = <M, p_i q_i^T> (tied to decoder). TopK.
Optional MIXER after TopK. Loss EXPANDED (Dooms) so d x d M is never instantiated:
  ||M-Mhat||^2 = ||y||^2/||x||^2  - 2 Σ_i z'_i (y·p_i)(x^+·q_i)  +  Σ_ij z'_i z'_j (p_i·p_j)(q_i·q_j).
VERIFY the expansion vs the explicit d x d loss (d=16 small). Metric: per-feature operator-purity (does a
feature fire when one operator is active), chance KACT/N; compare with/without mixer to tick-38 explicit-J
SAE (op-purity 0.898, MMCS 0.926). Same gated-superposition toy as jacsae2 (N=10 ops, 3 active/token).
"""
import sys, torch, numpy as np, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DC, N, KACT, KC, NT = 16, 10, 3, 3, 7000
M, KSP, STEPS = 64, 6, 5000

def make(seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.stack([torch.linalg.qr(torch.randn(DC, DC, generator=g))[0] for _ in range(N)])
    V = F.normalize(torch.randn(N, DC, generator=g), dim=1); W = F.normalize(torch.randn(N, DC, generator=g), dim=1)
    Sc = torch.stack([torch.randperm(N, generator=g)[:KC] for _ in range(NT)])
    coef = torch.rand(NT, KC, generator=g) + 0.5
    c = torch.zeros(NT, DC)
    for t in range(NT): c[t] = (coef[t, :, None] * V[Sc[t]]).sum(0)
    c = c + torch.randn(NT, DC, generator=g) * 0.1
    gate = (c @ W.T) ** 2; topv, topi = gate.topk(KACT, 1)
    Sop = torch.zeros(NT, N).scatter_(1, topi, topv); Sop = Sop / Sop.sum(1, keepdim=True).clamp_min(1e-9)
    op_active = (Sop > 0).float().numpy()
    y = torch.einsum("ti,ijk,tk->tj", Sop, A, c)                     # y = Σ s_i A_i c
    return c, y, op_active, A

class BilinearSAE(torch.nn.Module):
    def __init__(self, d, m, mixer=False):
        super().__init__()
        self.q = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)     # x^+ readers / decoder-right
        self.p = torch.nn.Parameter(torch.randn(m, d) / d ** 0.5)     # y readers / decoder-left
        self.mix = torch.nn.Parameter(torch.eye(m)) if mixer else None
    def encode(self, xp, y):
        return (xp @ self.q.T) * (y @ self.p.T)                       # z_i = (q_i·x^+)(p_i·y)
    def topk(self, z, k):
        val, ix = z.abs().topk(k, 1); zt = torch.zeros_like(z).scatter_(1, ix, torch.gather(z, 1, ix))
        return zt
    def loss(self, xp, y, k):
        z = self.topk(self.encode(xp, y), k)
        zp = z @ self.mix.T if self.mix is not None else z
        yp = y @ self.p.T                                            # (n,m) y·p_i
        xq = xp @ self.q.T                                           # (n,m) x^+·q_i
        cross = (zp * yp * xq).sum(1)                                # Σ_i z'_i (y·p_i)(x^+·q_i)
        Gp = self.p @ self.p.T; Gq = self.q @ self.q.T              # (m,m)
        quad = torch.einsum("ti,ij,tj->t", zp, Gp * Gq, zp)         # Σ_ij z'_i z'_j (p_i·p_j)(q_i·q_j)
        mnorm = (y ** 2).sum(1) / (xp ** 2).sum(1).clamp_min(1e-9) * (xp ** 2).sum(1)  # ||y||^2 ||x^+||^2
        # note ||x^+||^2 = 1/||x||^2, and xp = x/||x||^2 already, so ||M||^2 = ||y||^2 * ||xp||^2:
        mnorm = (y ** 2).sum(1) * (xp ** 2).sum(1)
        return (mnorm - 2 * cross + quad).mean(), z

def verify_expansion(sae, xp, y, k):
    """explicit d x d loss vs expanded (d=16 cheap)."""
    z = sae.topk(sae.encode(xp, y), k); zp = z @ sae.mix.T if sae.mix is not None else z
    Mhat = torch.einsum("ti,ij,ik->tjk", zp, sae.p, sae.q)          # Σ z'_i p_i q_i^T
    Mtrue = torch.einsum("ti,tj->tij", y, xp)                       # y x^+^T
    explicit = ((Mtrue - Mhat) ** 2).sum((1, 2)).mean()
    expanded, _ = sae.loss(xp, y, k)
    return float(explicit), float(expanded)

def op_purity(z, op_active):
    acts = (z.abs() > 1e-6).cpu().numpy(); tot = wsum = 0.0
    for j in range(acts.shape[1]):
        fire = acts[:, j]; w = fire.sum()
        if w < 5: continue
        tot += op_active[fire].mean(0).max() * w; wsum += w
    return tot / max(wsum, 1)

print(f"Bilinear SAE on the SECANT. toy: N={N} ops, {KACT} active/token. chance op-purity {KACT/N:.3f}.")
print(f"(compare tick-38 explicit-J SAE: op-purity 0.898, MMCS 0.926)\n")
res = {}
for mixer in (False, True):
    ps = []
    for seed in range(5):
        c, y, op_active, A = make(seed)
        xp = (c / (c ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); yv = y.to(DEV)
        sae = BilinearSAE(DC, M, mixer=mixer).to(DEV)
        opt = torch.optim.Adam(sae.parameters(), lr=2e-3)
        gg = torch.Generator(device=DEV).manual_seed(seed)
        for step in range(STEPS):
            bi = torch.randint(0, NT, (2048,), generator=gg, device=DEV)
            loss, _ = sae.loss(xp[bi], yv[bi], KSP)
            opt.zero_grad(); loss.backward(); opt.step()
        if seed == 0:
            ex, exp = verify_expansion(sae, xp[:256], yv[:256], KSP)
            print(f"  [mixer={mixer}] loss-expansion check: explicit {ex:.5f} vs expanded {exp:.5f} (rel err {abs(ex-exp)/max(ex,1e-9):.1e})")
        with torch.no_grad(): z = sae.topk(sae.encode(xp, yv), KSP)
        ps.append(op_purity(z, op_active))
    res[mixer] = np.array(ps)
print()
for mixer in (False, True):
    p = res[mixer]; print(f"  bilinear-SAE-on-secant  mixer={str(mixer):5s}  op-purity {p.mean():.3f}±{p.std():.3f}")
print(f"\n  chance {KACT/N:.3f}. Hypothesis: mixer=True recovers operators better (re-bundles rank-1 slices).")
print("DONE")
