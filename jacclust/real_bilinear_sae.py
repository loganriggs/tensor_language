"""tick 43 (Logan): bilinear SAE on the SECANT of REAL bilinear MLPs (the feasible operator-SAE, tick 42).

x = post-norm MLP input h; y = MLP output; secant M = y h^+ (h^+ = h/||h||^2). Bilinear SAE reconstructs M
as a sparse sum of rank-1 atoms, expanded loss (never forms d×d). Metrics (no labels): secant reconstruction
FVU, FUNCTIONAL FVU (does M_hat h reproduce y), vs random-atom control. If bad -> OUTLIER analysis (Logan):
token-norm concentration, per-dim outlier ratios of h/y, effective rank of the secant collection, and whether
SAE capacity/error concentrates on outlier tokens.  block2-dense-seed0 MLP#0 (layers.1), MLP#1 (layers.3).
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from deep_model import DeepModel
from jacclust.bilinear_sae import BilinearSAE
torch.set_default_dtype(torch.float32); DEV = "cuda" if torch.cuda.is_available() else "cpu"

cfg = json.load(open("runs_owt/block2-dense-seed0/config.json"))
model = DeepModel(cfg["vocab"], cfg["d_model"], cfg["n_head"], cfg["spec"], cfg["n_ctx"], d_hidden=512,
                  scale=cfg.get("scale", 0.5), norm=cfg["norm"], residual=cfg["residual"], attention=cfg["attention"])
model.load_state_dict(torch.load("runs_owt/block2-dense-seed0/model.pt", map_location="cpu", weights_only=True)); model.eval()
rms = torch.nn.RMSNorm(cfg["d_model"], elementwise_affine=False)
toks = np.fromfile("data_text/val.bin", dtype=np.uint16).astype(np.int64)
T = cfg["n_ctx"]; nseq = 120
seqs = torch.tensor(toks[:nseq * T].reshape(nseq, T))
with torch.no_grad(): stream = model.residuals(seqs)

d = cfg["d_model"]; M_DICT, KSP, STEPS = 256, 16, 6000
def get_hy(layer_i, in_i):
    mlp = model.layers[layer_i]; Dw, Lw, Rw = mlp.D.weight.detach(), mlp.L.weight.detach(), mlp.R.weight.detach()
    h = rms(stream[in_i].reshape(-1, d)).detach()
    with torch.no_grad(): y = (Dw @ ((Lw @ h.T) * (Rw @ h.T))).T
    return h, y

def train_and_eval(h, y, seed, train_atoms=True):
    n = h.shape[0]; hp = (h / (h ** 2).sum(1, keepdim=True).clamp_min(1e-9)).to(DEV); yv = y.to(DEV)
    sae = BilinearSAE(d, M_DICT, mixer=False).to(DEV)
    if train_atoms:
        opt = torch.optim.Adam(sae.parameters(), lr=2e-3); g = torch.Generator(device=DEV).manual_seed(seed)
        for _ in range(STEPS):
            bi = torch.randint(0, n, (2048,), generator=g, device=DEV)
            loss, _ = sae.loss(hp[bi], yv[bi], KSP); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        z = sae.topk(sae.encode(hp, yv), KSP)
        # secant reconstruction FVU (expanded): ||M-Mhat||^2 / ||M||^2, per token
        yp = yv @ sae.p.T; xq = hp @ sae.q.T
        Gp = sae.p @ sae.p.T; Gq = sae.q @ sae.q.T
        cross = (z * yp * xq).sum(1); quad = torch.einsum("ti,ij,tj->t", z, Gp * Gq, z)
        mnorm = (yv ** 2).sum(1) * (hp ** 2).sum(1)
        err = (mnorm - 2 * cross + quad).clamp_min(0)
        fvu_secant = float(err.sum() / mnorm.sum())
        # functional FVU: Mhat h = Σ z_i p_i (q_i·h); does it reproduce y?
        qh = (hp @ sae.q.T)                      # (n,m) but q reads h^+; want q_i·h -> use h not h^+
        qh_raw = (h.to(DEV) @ sae.q.T)
        Mhat_h = torch.einsum("ti,id->td", z * qh_raw, sae.p)   # Σ z_i (q_i·h) p_i
        fvu_func = float(((yv - Mhat_h) ** 2).sum() / (yv ** 2).sum())
        per_tok_fvu = (err / mnorm.clamp_min(1e-12)).cpu().numpy()
    return fvu_secant, fvu_func, per_tok_fvu

MLPS = {"MLP#0 (layers.1)": (1, 0), "MLP#1 (layers.3)": (3, 2)}
print(f"block2 bilinear-secant SAE on REAL MLPs. d={d}, dict={M_DICT}, k={KSP}, 3 seeds.\n")
for name, (li, ii) in MLPS.items():
    h, y = get_hy(li, ii); n = h.shape[0]
    Mt = (y ** 2).sum(1) * (h ** 2).sum(1) / (h ** 2).sum(1).clamp_min(1e-9)   # ||M_t||^2 = ||y||^2/||h||^2
    Mnorm2 = (y ** 2).sum(1) / (h ** 2).sum(1).clamp_min(1e-9)
    fs = np.array([train_and_eval(h, y, s)[0] for s in range(3)])
    ff = np.array([train_and_eval(h, y, s)[1] for s in range(3)])
    fr = np.array([train_and_eval(h, y, s, train_atoms=False)[0] for s in range(3)])
    print(f"=== {name} (n={n}) ===")
    print(f"  secant recon FVU  trained {fs.mean():.3f}±{fs.std():.3f}   random-atoms {fr.mean():.3f}")
    print(f"  functional FVU (M_hat h vs y): {ff.mean():.3f}±{ff.std():.3f}   (0=perfect, 1=useless)")
    # outlier diagnostics
    mm = Mnorm2.cpu().numpy(); order = np.argsort(-mm)
    top1 = mm[order[:max(1, n // 100)]].sum() / mm.sum()
    hn, yn = h.cpu().numpy(), y.cpu().numpy()
    kurt_h = ((hn - hn.mean(0)) ** 4).mean(0) / (((hn - hn.mean(0)) ** 2).mean(0) ** 2 + 1e-12)
    outdim_h = (np.abs(hn).max(0) / (np.median(np.abs(hn), 0) + 1e-9))
    outdim_y = (np.abs(yn).max(0) / (np.median(np.abs(yn), 0) + 1e-9))
    # effective rank of the secant collection (vec(M_t) = y_t ⊗ h_t^+) via random-probe covariance trace ratio
    Uh, Sh, _ = torch.linalg.svd(h - h.mean(0), full_matrices=False)
    effrank_h = float((Sh.sum() ** 2) / (Sh ** 2).sum())
    print(f"  OUTLIERS: top-1% tokens hold {top1*100:.1f}% of secant mass | max/median dim-ratio h {outdim_h.max():.0f} y {outdim_y.max():.0f}")
    print(f"           #dims with kurtosis>20: {(kurt_h>20).sum()}/{d} | eff-rank(h) {effrank_h:.1f}/{d}")
    per = train_and_eval(h, y, 0)[2]
    corr = np.corrcoef(per, mm)[0, 1]
    print(f"           corr(per-token FVU, ||M_t||^2) = {corr:+.2f} (>0 => error concentrates on high-norm/outlier tokens)\n")
print("DONE")
