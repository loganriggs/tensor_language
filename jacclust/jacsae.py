"""tick 37 (Logan): SAEs on the Jacobian object vs SAEs on raw activations — do they recover MECHANISM?

The k-means clustering work showed the per-token Jacobian J(x) exposes the mechanism (which expert/gate is
active) that the raw activation buries. SAE = the overcomplete/soft version. Question: does an SAE trained on
J(x) recover MECHANISM features (experts) where an SAE on x recovers CONTENT features?

Ground-truth toy (gated superposition): content cluster lc (center μ) processed by expert g -> y = A_g c.
  x = [c ; eps·onehot(g)]     (gate is tiny in x — the whole point)     J(x) = closed-form bilinear Jacobian.
Train matched TopK SAEs (same dict size m, same k, 5 seeds) on x and on J(x). Metric per latent: among the
tokens it fires on, gate-PURITY (max fraction sharing one expert) and content-PURITY (one content cluster),
activity-weighted mean over latents. Chance = 1/G, 1/C. Controls: matched-dim random projection of J, and the
G-embedding G^{1/2}x. Prediction: J-SAE gate-pure >> x-SAE; x-SAE content-pure >> J-SAE.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/workspace/tensor_language")
from jacclust.dgp import build_layer
from jacclust.metric import jacobian, gram

torch.set_default_dtype(torch.float32)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
G, C, D_C, EPS = 6, 8, 24, 0.10
NPC = 170                                   # tokens per (gate,content) cell -> n = G*C*NPC
M, KSPARSE, STEPS = 64, 4, 4000             # SAE dict size, TopK, train steps

def unit(F): return F / F.norm(dim=1, keepdim=True).clamp_min(1e-9)

def make(seed):
    gen = torch.Generator().manual_seed(seed)
    A = torch.stack([torch.linalg.qr(torch.randn(D_C, D_C, generator=gen))[0] for _ in range(G)])
    D, L, R = build_layer(A, G, D_C, EPS)
    mu = torch.randn(C, D_C, generator=gen) * 3.0
    rows = []
    for g in range(G):
        for lc in range(C):
            c = mu[lc] + torch.randn(NPC, D_C, generator=gen) * 0.5
            s = torch.zeros(NPC, G); s[:, g] = EPS
            x = torch.cat([c, s], 1)
            rows.append((x, torch.full((NPC,), g), torch.full((NPC,), lc)))
    X = torch.cat([r[0] for r in rows]); gl = torch.cat([r[1] for r in rows]).numpy(); cl = torch.cat([r[2] for r in rows]).numpy()
    Jfull = torch.stack([jacobian(D, L, R, x) for x in X])             # (n, d_out, d_in)
    J = Jfull.reshape(len(X), -1)
    Jrest = Jfull[:, :, :D_C].reshape(len(X), -1)                      # content columns (the S8 object)
    Gm = gram(D, L, R); w, V = torch.linalg.eigh(Gm); w = w.clamp_min(0)
    Z = (X @ V) * w.sqrt()[None, :]                                    # G-embedding
    objs = {
        "x (normal SAE)": X,
        "J(x) raw": J,
        "J(x) NORMALIZED": unit(J),
        "restricted-J norm": unit(Jrest),
        "G^1/2 x norm": unit(Z),
    }
    return objs, gl, cl

def train_topk_sae(F, seed):
    F = F.to(DEV); n, d = F.shape
    mean = F.mean(0); Fc = F - mean; scale = Fc.norm(dim=1).mean().clamp_min(1e-6); Fc = Fc / scale
    g = torch.Generator(device=DEV).manual_seed(seed)
    We = torch.randn(d, M, generator=g, device=DEV) * (1.0 / d ** 0.5); We.requires_grad_()
    Wd = We.detach().clone().T.contiguous().requires_grad_()
    b = torch.zeros(M, device=DEV, requires_grad=True)
    opt = torch.optim.Adam([We, Wd, b], lr=2e-3)
    idxall = torch.arange(n, device=DEV)
    for step in range(STEPS):
        bi = idxall[torch.randint(0, n, (2048,), generator=g, device=DEV)]
        pre = Fc[bi] @ We + b
        val, ix = pre.topk(KSPARSE, dim=1)
        z = torch.zeros_like(pre).scatter_(1, ix, torch.relu(val))
        rec = z @ Wd
        loss = ((rec - Fc[bi]) ** 2).sum(1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad(): Wd.data = Wd.data / Wd.data.norm(dim=1, keepdim=True).clamp_min(1e-6)
    with torch.no_grad():
        pre = Fc @ We + b; val, ix = pre.topk(KSPARSE, 1)
        Zc = torch.zeros_like(pre).scatter_(1, ix, torch.relu(val))
        r2 = 1 - ((Zc @ Wd - Fc) ** 2).sum() / (Fc ** 2).sum()
    return Zc.cpu().numpy(), float(r2)

def purity(acts, lab, nlab):
    """activity-weighted mean, over latents, of max-fraction-of-firing-tokens sharing one label."""
    tot = 0.0; wsum = 0.0
    for j in range(acts.shape[1]):
        fire = acts[:, j] > 1e-6
        w = fire.sum()
        if w < 5: continue
        labs = lab[fire]; frac = np.bincount(labs, minlength=nlab).max() / w
        tot += frac * w; wsum += w
    return tot / max(wsum, 1)

print(f"gated-superposition toy: G={G} experts, C={C} content, d_c={D_C}, eps={EPS}. TopK SAE m={M}, k={KSPARSE}.")
print(f"chance gate-purity {1/G:.3f}, content-purity {1/C:.3f}\n")
names = ["x (normal SAE)", "J(x) raw", "J(x) NORMALIZED", "restricted-J norm", "G^1/2 x norm"]
res = {k: {"g": [], "c": [], "r2": []} for k in names}
for seed in range(5):
    objs, gl, cl = make(seed)
    for name in names:
        acts, r2 = train_topk_sae(objs[name], seed)
        res[name]["g"].append(purity(acts, gl, G)); res[name]["c"].append(purity(acts, cl, C)); res[name]["r2"].append(r2)

print(f"{'SAE trained on':22s} {'gate-purity':>14s} {'content-purity':>16s} {'recon R2':>10s}")
for name in res:
    g = np.array(res[name]["g"]); c = np.array(res[name]["c"]); r = np.array(res[name]["r2"])
    print(f"  {name:20s} {g.mean():.3f}±{g.std():.3f}   {c.mean():.3f}±{c.std():.3f}    {r.mean():.3f}")
print(f"\n  chance gate {1/G:.3f}, content {1/C:.3f}. WIN = J-SAE gate-purity >> x-SAE gate-purity.")
print("DONE")
