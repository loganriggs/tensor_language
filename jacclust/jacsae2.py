"""tick 38 (autonomous): SAE on Jacobian object — the NON-DEGENERATE superposition test (fixes tick-37 caveat).

tick 37: restricted-J SAE recovered mechanism, but the toy was degenerate (one gate/token, restricted-J=A_g,
R2=1.0). Real test: each token fires a SPARSE SET of operators, and which operators fire is a NONLINEAR
function of content (hidden from x). SAEs are designed for exactly this superposition.

Toy: content c = superposition of content features v_j (set S_c). Operator i active iff (w_i·c)^2 in top-k
(NONLINEAR in c, set S_op ≠ S_c). y = Σ_{i∈S_op} s_i A_i c. Objects (per token, normalized):
  x = c            (activation; SAE recovers content features v_j)
  J_full           (full restricted Jacobian ∂y/∂c; operator block + gate-derivative contamination)
  J_op = Σ s_i A_i (operator block only — the ideal mechanism object)
Matched TopK SAEs, 5 seeds. Metrics: per-latent OPERATOR-purity (max_i P(op i active | latent fires);
chance = k_active/N) and MMCS of atoms to the true operator dict {vec(A_i)}. Prediction: J_op-SAE recovers
operators (purity, MMCS high); x-SAE at chance on operators (they're hidden); J_full contaminated (between).
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language")
torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DC, N, KACT, KC = 16, 10, 3, 3          # content dim, #operators, active operators/token, active content feats
NT, M, KSPARSE, STEPS = 7000, 96, 6, 4000

def unit(F): return F / F.norm(dim=1, keepdim=True).clamp_min(1e-9)

def make(seed):
    g = torch.Generator().manual_seed(seed)
    A = torch.stack([torch.linalg.qr(torch.randn(DC, DC, generator=g))[0] for _ in range(N)])   # operators
    V = unit(torch.randn(N, DC, generator=g))              # content feature directions (reuse N of them)
    W = unit(torch.randn(N, DC, generator=g))              # gate readers (nonlinear)
    # content: sparse superposition of KC content features
    Sc = torch.stack([torch.randperm(N, generator=g)[:KC] for _ in range(NT)])
    coef = torch.rand(NT, KC, generator=g) + 0.5
    c = torch.zeros(NT, DC)
    for t in range(NT): c[t] = (coef[t, :, None] * V[Sc[t]]).sum(0)
    c = c + torch.randn(NT, DC, generator=g) * 0.1
    # operator activation: s_i = (w_i·c)^2, keep top-KACT (nonlinear in c -> hidden from x)
    gate = (c @ W.T) ** 2                                  # (NT, N)
    topv, topi = gate.topk(KACT, dim=1)
    Sop = torch.zeros(NT, N); Sop.scatter_(1, topi, topv)  # sparse operator coeffs
    Sop = Sop / Sop.sum(1, keepdim=True).clamp_min(1e-9)
    op_active = (Sop > 0).float().numpy()                  # (NT,N) ground-truth operator-active
    # objects
    Jop = torch.einsum("ti,ijk->tjk", Sop, A).reshape(NT, -1)        # Σ s_i A_i  (operator block)
    # full restricted-J = ∂y/∂c, y=Σ s_i(c) A_i c ; s_i=(w_i·c)^2 (use soft, all i, for the derivative)
    soft = (c @ W.T) ** 2
    y_op = torch.einsum("ti,ijk->tjk", soft, A)                     # Σ s_i A_i (soft, all)
    gate_deriv = torch.einsum("ti,ijk,tk->tij", 2 * (c @ W.T), A, c)  # Σ 2(w_i·c)(A_i c) w_i^T  -> (t,j,i)... build properly
    # gate-deriv term: Σ_i (A_i c) * 2(w_i·c) * w_i^T  -> outer(A_i c, w_i)
    Aic = torch.einsum("ijk,tk->tij", A, c)                        # (t, j=out, i=op) : A_i c per op
    coeff = 2 * (c @ W.T)                                          # (t, i)
    gd = torch.einsum("tij,ti,ik->tjk", Aic, coeff, W)             # (t, out, in)
    Jfull = (y_op + gd).reshape(NT, -1)
    objs = {"x (content, normal SAE)": unit(c), "J_full norm": unit(Jfull), "J_op norm (mechanism)": unit(Jop)}
    return objs, op_active, A

def train_topk_sae(F, seed):
    F = F.to(DEV); n, d = F.shape
    mean = F.mean(0); Fc = F - mean; scale = Fc.norm(dim=1).mean().clamp_min(1e-6); Fc = Fc / scale
    g = torch.Generator(device=DEV).manual_seed(seed)
    We = (torch.randn(d, M, generator=g, device=DEV) / d ** 0.5).requires_grad_()
    Wd = We.detach().clone().T.contiguous().requires_grad_(); b = torch.zeros(M, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([We, Wd, b], lr=2e-3)
    for step in range(STEPS):
        bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
        pre = Fc[bi] @ We + b; val, ix = pre.topk(KSPARSE, 1)
        z = torch.zeros_like(pre).scatter_(1, ix, torch.relu(val)); rec = z @ Wd
        loss = ((rec - Fc[bi]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Wd.data = Wd.data / Wd.data.norm(dim=1, keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        pre = Fc @ We + b; val, ix = pre.topk(KSPARSE, 1)
        Z = torch.zeros_like(pre).scatter_(1, ix, torch.relu(val))
        r2 = 1 - ((Z @ Wd - Fc) ** 2).sum() / (Fc ** 2).sum()
    return Z.cpu().numpy(), Wd.detach().cpu(), float(r2)

def op_purity(acts, op_active):
    """per-latent: max_i P(op i active | latent fires); activity-weighted mean. chance = KACT/N."""
    tot = wsum = 0.0
    for j in range(acts.shape[1]):
        fire = acts[:, j] > 1e-6; w = fire.sum()
        if w < 5: continue
        p = op_active[fire].mean(0).max(); tot += p * w; wsum += w
    return tot / max(wsum, 1)

def mmcs_to_ops(Wd, A):
    """mean over operators of max cosine of any dict atom to vec(A_i). (only meaningful for J-space objects)"""
    if Wd.shape[1] != A.shape[1] * A.shape[2]: return float("nan")
    ops = A.reshape(N, -1); ops = ops / ops.norm(dim=1, keepdim=True); Wn = Wd / Wd.norm(dim=1, keepdim=True)
    sim = (ops @ Wn.T).abs()                                # (N, M)
    return float(sim.max(1).values.mean())

names = ["x (content, normal SAE)", "J_full norm", "J_op norm (mechanism)"]
res = {k: {"p": [], "m": [], "r2": []} for k in names}
for seed in range(5):
    objs, op_active, A = make(seed)
    for name in names:
        acts, Wd, r2 = train_topk_sae(objs[name], seed)
        res[name]["p"].append(op_purity(acts, op_active)); res[name]["m"].append(mmcs_to_ops(Wd, A)); res[name]["r2"].append(r2)

print(f"superposition-of-operators: d_c={DC}, N={N} operators, {KACT} active/token, {KC} content feats/token.")
print(f"chance operator-purity = KACT/N = {KACT/N:.3f}. TopK SAE m={M}, k={KSPARSE}, 5 seeds.\n")
print(f"{'SAE on':28s} {'op-purity':>14s} {'MMCS->operators':>16s} {'recon R2':>10s}")
for name in names:
    p = np.array(res[name]["p"]); mm = np.array(res[name]["m"]); r = np.array(res[name]["r2"])
    mms = f"{np.nanmean(mm):.3f}" if not np.isnan(mm).all() else "  n/a (x-space)"
    print(f"  {name:26s} {p.mean():.3f}±{p.std():.3f}   {mms:>14s}    {r.mean():.3f}")
print(f"\n  chance op-purity {KACT/N:.3f}. WIN = J_op op-purity & MMCS >> x; contamination = J_full < J_op.")
print("DONE")
