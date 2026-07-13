"""Sanity checks for the Gaussian tensor-similarity metric — RUN BEFORE ANY TRAINING LOOP.

Per the handoff: "L_fid(A,A)=0 and gauge invariance ... the part most likely to be silently wrong."
We verify the closed form against two INDEPENDENT references (exact brute-force Isserlis contraction,
and Monte-Carlo E[y·ŷ]) — an internal consistency check alone could not catch a shared bug.

Checks:
  C1  closed form == brute-force Isserlis contraction on the explicit tensor           (exact)
  C2  closed form == Monte-Carlo E[y·ŷ], x̃~N(0,G)                                      (converges)
  C3  L_fid(A,A) == 0  and  cos(A,A) == 1
  C4  L_fid == relative Gaussian MSE  E‖y-ŷ‖²/E‖y‖²   (the metric's meaning)            (MC)
  C5  GAUGE (must be invariant): hidden permutation; hidden rescaling; L<->R swap
  C6  CONTROL THAT MUST FAIL: random invertible U on the hidden index is NOT a CP gauge
      (if this "passes", the metric is degenerate/broken)
  C7  data-matched metric: G = arbitrary SPD Σ, closed form == MC with x̃~N(0,Σ)
  C8  real layers: jacclust DGP bilinear layer, and a real 500M bilinear MLP layer
"""
import sys, torch
sys.path.insert(0, "/workspace/tensor_language")
sys.path.insert(0, "/workspace/tensor_language/tensor_sim_regularized_bilinear_transcoders")
from tensor_sim import (tensor_inner, norm_sq, fid_loss, cosine_sim,
                        build_tensor, tensor_inner_bruteforce, forward)

torch.set_default_dtype(torch.float64)          # exactness matters here
g = torch.Generator().manual_seed(0)
OK = "\033[32mPASS\033[0m"; BAD = "\033[31mFAIL\033[0m"
fails = []
def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}  {detail}")
    if not cond: fails.append(name)

def rand_layer(K, r, d, gen):
    """Random CP bilinear layer on lifted inputs (d = d_in+1)."""
    D = torch.randn(K, r, generator=gen)
    L = torch.randn(r, d, generator=gen)
    R = torch.randn(r, d, generator=gen)
    return D, L, R

def spd(d, gen):
    M = torch.randn(d, d, generator=gen)
    return M @ M.T / d + 0.5 * torch.eye(d)

# ---------------------------------------------------------------- tiny exact case
K, r, d = 3, 4, 5
D, L, R = rand_layer(K, r, d, g)
D2, L2, R2 = rand_layer(K, 6, d, g)                       # different rank r'=6 (overcomplete)
A, Ah = build_tensor(D, L, R), build_tensor(D2, L2, R2)

print("\nC1  closed form vs brute-force Isserlis contraction (exact)")
for tag, G in [("G=I", None), ("G=Σ", spd(d, g))]:
    cf = tensor_inner(D, L, R, D2, L2, R2, G)
    bf = tensor_inner_bruteforce(A, Ah, G)
    rel = abs(float(cf - bf)) / max(abs(float(bf)), 1e-30)
    check(f"⟨A|Λ|Â⟩ closed==brute  [{tag}]", rel < 1e-10, f"rel err {rel:.2e}")

print("\nC2  closed form vs Monte-Carlo E[y·ŷ]  (x̃ ~ N(0,G))")
for tag, G in [("G=I", None), ("G=Σ", spd(d, g))]:
    n = 4_000_000
    Gm = torch.eye(d) if G is None else G
    Lch = torch.linalg.cholesky(Gm)
    x = torch.randn(n, d, generator=g) @ Lch.T
    mc = (forward(D, L, R, x) * forward(D2, L2, R2, x)).sum(1).mean()
    cf = tensor_inner(D, L, R, D2, L2, R2, G)
    rel = abs(float(cf - mc)) / max(abs(float(cf)), 1e-30)
    check(f"⟨A|Λ|Â⟩ closed==MC     [{tag}]", rel < 0.02, f"closed {float(cf):+.4f} vs MC {float(mc):+.4f} (rel {rel:.1e})")

print("\nC3  L_fid(A,A) == 0   and   cos(A,A) == 1")
for tag, G in [("G=I", None), ("G=Σ", spd(d, g))]:
    lf = float(fid_loss(D, L, R, D, L, R, G)); cs = float(cosine_sim(D, L, R, D, L, R, G))
    check(f"L_fid(A,A)=0 [{tag}]", abs(lf) < 1e-12, f"{lf:.2e}")
    check(f"cos(A,A)=1   [{tag}]", abs(cs - 1) < 1e-12, f"{cs:.12f}")

print("\nC4  L_fid == relative Gaussian MSE  E‖y-ŷ‖²/E‖y‖²   (the metric's meaning)")
n = 4_000_000
x = torch.randn(n, d, generator=g)
y, yh = forward(D, L, R, x), forward(D2, L2, R2, x)
mse_rel = float(((y - yh) ** 2).sum(1).mean() / (y ** 2).sum(1).mean())
lf = float(fid_loss(D, L, R, D2, L2, R2, None))
check("L_fid == E‖y-ŷ‖²/E‖y‖²", abs(lf - mse_rel) / mse_rel < 0.02, f"closed {lf:.4f} vs MC {mse_rel:.4f}")

print("\nC5  GAUGE — must be INVARIANT (L_fid=0, cos=1)")
perm = torch.randperm(r, generator=g)
check("hidden permutation", abs(float(fid_loss(D, L, R, D[:, perm], L[perm], R[perm]))) < 1e-12)
a = torch.rand(r, generator=g) + 0.5; b = torch.rand(r, generator=g) + 0.5
Ls, Rs, Ds = a[:, None] * L, b[:, None] * R, D / (a * b)[None, :]
check("hidden rescaling", abs(float(fid_loss(D, L, R, Ds, Ls, Rs))) < 1e-12)
# L<->R swap transposes each slice: same FUNCTION (quadratic form sees only the symmetric part)
lf_swap = float(fid_loss(D, L, R, D, R, L))
check("L<->R swap (same function)", abs(lf_swap) < 1e-12, f"{lf_swap:.2e}")
# and it really is the same function:
xs = torch.randn(2000, d, generator=g)
check("  ...forward(L,R) == forward(R,L)", torch.allclose(forward(D, L, R, xs), forward(D, R, L, xs), atol=1e-10))

print("\nC6  CONTROL THAT MUST FAIL — general invertible U on hidden index is NOT a CP gauge")
U = torch.randn(r, r, generator=g); U = U + r * torch.eye(r)          # well-conditioned
Lu, Ru, Du = U @ L, U @ R, D @ torch.linalg.inv(U)
lf_u = float(fid_loss(D, L, R, Du, Lu, Ru))
check("random U on hidden index BREAKS invariance", lf_u > 1e-3,
      f"L_fid={lf_u:.4f} (must be >0; if ~0 the metric is degenerate)")

print("\nC7  data-matched metric with G=Σ: gauge still invariant, U still breaks")
Sig = spd(d, g)
check("permutation invariant [G=Σ]", abs(float(fid_loss(D, L, R, D[:, perm], L[perm], R[perm], Sig))) < 1e-12)
check("rescaling invariant   [G=Σ]", abs(float(fid_loss(D, L, R, Ds, Ls, Rs, Sig))) < 1e-12)
check("random U breaks       [G=Σ]", float(fid_loss(D, L, R, Du, Lu, Ru, Sig)) > 1e-3)

print("\nC8  real bilinear layers")
# (a) jacclust DGP hand-built layer
from jacclust.dgp import make_experts, build_layer
gg = torch.Generator().manual_seed(1)
Aexp = make_experts(k_g=3, d_c=6, gen=gg, geometry="orthogonal")
Dd, Ld, Rd = build_layer(Aexp, k_g=3, d_c=6, eps=0.1)
Dd, Ld, Rd = Dd.double(), Ld.double(), Rd.double()
dd = Ld.shape[1]
cf = tensor_inner(Dd, Ld, Rd, Dd, Ld, Rd, None)
bf = tensor_inner_bruteforce(build_tensor(Dd, Ld, Rd), build_tensor(Dd, Ld, Rd), None)
check("DGP layer: closed==brute", abs(float(cf - bf)) / abs(float(bf)) < 1e-10)
check("DGP layer: L_fid(A,A)=0", abs(float(fid_loss(Dd, Ld, Rd, Dd, Ld, Rd))) < 1e-12)
pg = torch.randperm(Ld.shape[0], generator=gg)
check("DGP layer: perm gauge invariant", abs(float(fid_loss(Dd, Ld, Rd, Dd[:, pg], Ld[pg], Rd[pg]))) < 1e-12)

# (b) a REAL 500M bilinear MLP layer (Left/Right/Down) — checks it scales (r=4608)
try:
    import json
    from huggingface_hub import hf_hub_download
    import jacclust.tt_model as TT
    repo = "Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd"
    cfg = json.load(open(hf_hub_download(repo, "config.json"))); cfg.pop("step", None)
    m = TT.GPT(TT.GPTConfig(**cfg)).eval()
    m.load_state_dict(torch.load(hf_hub_download(repo, "pytorch_model.bin"), map_location="cpu", weights_only=True))
    mlp = m.transformer.h[8].mlp
    Lr = mlp.Left.weight.detach().double(); Rr = mlp.Right.weight.detach().double(); Dr = mlp.Down.weight.detach().double()
    print(f"    real L8 MLP: D{tuple(Dr.shape)} L{tuple(Lr.shape)} R{tuple(Rr.shape)}  (r={Lr.shape[0]}, d={Lr.shape[1]})")
    aa = float(norm_sq(Dr, Lr, Rr))
    check("real layer: ‖A‖²_Λ finite & >0", aa > 0 and aa == aa, f"‖A‖²_Λ={aa:.4e}")
    check("real layer: L_fid(A,A)=0", abs(float(fid_loss(Dr, Lr, Rr, Dr, Lr, Rr))) < 1e-10)
    pr = torch.randperm(Lr.shape[0], generator=gg)
    check("real layer: perm gauge invariant", abs(float(fid_loss(Dr, Lr, Rr, Dr[:, pr], Lr[pr], Rr[pr]))) < 1e-10)
    check("real layer: L<->R swap invariant", abs(float(fid_loss(Dr, Lr, Rr, Dr, Rr, Lr))) < 1e-10)
    # MC spot-check on the real layer (G=I)
    xr = torch.randn(20000, Lr.shape[1], generator=gg)
    mc = (forward(Dr, Lr, Rr, xr) ** 2).sum(1).mean()
    rel = abs(float(mc) - aa) / aa
    check("real layer: ‖A‖²_Λ == MC E‖y‖²", rel < 0.05, f"closed {aa:.4e} vs MC {float(mc):.4e} (rel {rel:.1e})")
except Exception as e:
    print(f"    [skip real-model check: {type(e).__name__}: {e}]")

print("\nC9  LIFTED inputs x̃=(1,x) — the handoff's Σ recipe is WRONG; non-central Wick is exact")
from tensor_sim import tensor_inner_mean, fid_loss_mean, lifted_moments
n = 3_000_000
Sx = spd(4, g); mx = torch.tensor([1., -2., .5, 3.])
xr = torch.randn(n, 4, generator=g) @ torch.linalg.cholesky(Sx).T + mx
xt = torch.cat([torch.ones(n, 1), xr], 1)
Sig, mu = lifted_moments(xr)
dl = 5
Dl, Ll, Rl = rand_layer(3, 4, dl, g)
Dl2, Ll2, Rl2 = rand_layer(3, 6, dl, g)
mc = float((forward(Dl, Ll, Rl, xt) * forward(Dl2, Ll2, Rl2, xt)).sum(1).mean())
good = float(tensor_inner_mean(Dl, Ll, Rl, Dl2, Ll2, Rl2, Sig, mu))
bad = float(tensor_inner(Dl, Ll, Rl, Dl2, Ll2, Rl2, (xt.T @ xt) / n))   # handoff: centered formula + uncentered 2nd moment
check("non-central Wick == MC (lifted)", abs(good - mc) / abs(mc) < 0.03,
      f"closed {good:+.2f} vs MC {mc:+.2f} (rel {abs(good-mc)/abs(mc):.1e})")
check("handoff's centered-Σ recipe is BIASED (documents the bug)", abs(bad - mc) / abs(mc) > 0.10,
      f"would give {bad:+.2f} -> rel err {abs(bad-mc)/abs(mc):.0%}")
check("lifted: L_fid(A,A)=0", abs(float(fid_loss_mean(Dl, Ll, Rl, Dl, Ll, Rl, Sig, mu))) < 1e-12)
pl = torch.randperm(4, generator=g)
check("lifted: perm gauge invariant", abs(float(fid_loss_mean(Dl, Ll, Rl, Dl[:, pl], Ll[pl], Rl[pl], Sig, mu))) < 1e-12)
Ul = torch.randn(4, 4, generator=g) + 4 * torch.eye(4)
check("lifted: random U BREAKS (control)",
      float(fid_loss_mean(Dl, Ll, Rl, Dl @ torch.linalg.inv(Ul), Ul @ Ll, Ul @ Rl, Sig, mu)) > 1e-3)
yl, yhl = forward(Dl, Ll, Rl, xt), forward(Dl2, Ll2, Rl2, xt)
rel_mse = float(((yl - yhl) ** 2).sum(1).mean() / (yl ** 2).sum(1).mean())
lfl = float(fid_loss_mean(Dl, Ll, Rl, Dl2, Ll2, Rl2, Sig, mu))
check("lifted: L_fid == E‖y-ŷ‖²/E‖y‖²", abs(lfl - rel_mse) / rel_mse < 0.03, f"closed {lfl:.4f} vs MC {rel_mse:.4f}")

print("\n" + "=" * 60)
print(f"RESULT: {'ALL CHECKS PASSED' if not fails else 'FAILURES: ' + ', '.join(fails)}")
print("=" * 60)
sys.exit(1 if fails else 0)
