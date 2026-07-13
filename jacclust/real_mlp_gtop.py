"""tick 41 (autonomous, priority-1): does the G-TOP PROJECTION recipe help on the REAL bilinear MLPs?

DGP-E recipe: project J's columns off the top-r eigenvectors of G (P4 says that IS the gate subspace);
on DGP-E it beat controls (ARI 0.654 vs 0.388). Does it transfer to the real block2 MLPs? No mechanism
labels, so use the SURROGATE test (per-cluster linear map h->y predicts held-out output; ticks 6-9). Metrics
clustered on: raw h, G-embedding, G-top-projected-J (r swept), G_rand spectrum-matched control (STANDING
RULE). 5 seeds. Targets: block2-dense-seed0 MLP#0 (layers.1) and MLP#1 (layers.3).
"""
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "/workspace/tensor_language")
from deep_model import DeepModel
from jacclust.metric import gram, embed
from jacclust.surrogate import projected_gram, random_spectrum_gram, global_r2
from sklearn.cluster import KMeans

def surrogate_r2_ridge(Ztr, Zte, Xtr, Ytr, Xte, Yte, k, seed, lam=5.0):
    """Ridge version of the per-cluster surrogate (tick-8 lesson: plain lstsq is unstable at d=128)."""
    Zn = Ztr / np.linalg.norm(Ztr, axis=1, keepdims=True).clip(1e-12)
    km = KMeans(k, n_init=6, random_state=seed).fit(Zn); ctr = km.labels_
    cen = km.cluster_centers_; cen = cen / np.linalg.norm(cen, axis=1, keepdims=True).clip(1e-12)
    Zen = Zte / np.linalg.norm(Zte, axis=1, keepdims=True).clip(1e-12)
    cte = np.argmax(Zen @ cen.T, 1)
    d = Xtr.shape[1]; num = 0.0
    for c in range(k):
        itr, ite = ctr == c, cte == c
        if ite.sum() == 0: continue
        if itr.sum() < d + 1:
            A = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(d), Xtr.T @ Ytr)
        else:
            Xc = Xtr[itr]; A = np.linalg.solve(Xc.T @ Xc + lam * np.eye(d), Xc.T @ Ytr[itr])
        num += ((Xte[ite] @ A - Yte[ite]) ** 2).sum()
    return 1.0 - num / ((Yte - Yte.mean(0)) ** 2).sum()
surrogate_r2 = surrogate_r2_ridge

torch.set_default_dtype(torch.float32); DEV = "cpu"
cfg = json.load(open("runs_owt/block2-dense-seed0/config.json"))
model = DeepModel(cfg["vocab"], cfg["d_model"], cfg["n_head"], cfg["spec"], cfg["n_ctx"],
                  d_hidden=512, scale=cfg.get("scale", 0.5), norm=cfg["norm"],
                  residual=cfg["residual"], attention=cfg["attention"])
model.load_state_dict(torch.load("runs_owt/block2-dense-seed0/model.pt", map_location=DEV, weights_only=True))
model.eval()
rms = torch.nn.RMSNorm(cfg["d_model"], elementwise_affine=False)

toks = np.fromfile("data_text/val.bin", dtype=np.uint16).astype(np.int64)
T = cfg["n_ctx"]; nseq = 120
seqs = torch.tensor(toks[:nseq * T].reshape(nseq, T))
with torch.no_grad():
    stream = model.residuals(seqs)                     # list of layer outputs

# MLP#0 = layers[1], input = stream[0]; MLP#1 = layers[3], input = stream[2]
MLPS = {"MLP#0 (layers.1)": (1, 0), "MLP#1 (layers.3)": (3, 2)}
K = 8

def prep(layer_i, in_i):
    mlp = model.layers[layer_i]
    Dw, Lw, Rw = mlp.D.weight.detach(), mlp.L.weight.detach(), mlp.R.weight.detach()
    x_in = stream[in_i].reshape(-1, cfg["d_model"]).detach()
    h = rms(x_in).detach()                             # post-norm datapoint
    with torch.no_grad():
        y = (Dw @ ((Lw @ h.T) * (Rw @ h.T))).T          # MLP output D(Lh⊙Rh)  (n, d)
    return Dw, Lw, Rw, h, y

print("block2-dense-seed0 — G-top projection on real MLPs (surrogate held-out R2, K=8, 5 seeds).")
print("STANDING RULE: G_rand spectrum-matched control on every cell.\n")
for name, (li, ii) in MLPS.items():
    Dw, Lw, Rw, h, y = prep(li, ii)
    n = h.shape[0]; d = cfg["d_model"]
    G = gram(Dw, Lw, Rw)
    w = torch.linalg.eigvalsh(G).clamp_min(0); effrank = float((w.sum() ** 2) / (w ** 2).sum())
    corr = None
    print(f"=== {name}  (n={n} tokens, d={d}, G eff-rank {effrank:.1f}/{d}) ===")
    hn, yn = h.numpy(), y.numpy()
    Zx = hn                                             # raw metric = h itself
    ZG = embed(G, h).numpy()
    Grand = random_spectrum_gram(G, seed=0); Zrand = embed(Grand, h).numpy()
    projobjs = {}
    for r in (4, 8, 16, 32):
        Gp, _ = projected_gram(Dw, Lw, Rw, r); projobjs[f"G-top-proj r={r}"] = embed(Gp, h).numpy()

    def run(Z):
        vals = []
        for s in range(5):
            rs = np.random.RandomState(s); idx = rs.permutation(n); tr, te = idx[:int(.7 * n)], idx[int(.7 * n):]
            vals.append(surrogate_r2(Z[tr], Z[te], hn[tr], yn[tr], hn[te], yn[te], K, s))
        return np.array(vals)
    cells = {"raw h": Zx, "G-embed": ZG, "G_rand (control)": Zrand, **projobjs}
    base = {}
    for cname, Z in cells.items():
        v = run(Z); base[cname] = v
        print(f"  {cname:22s} surrogate R2 {v.mean():+.4f}±{v.std():.4f}")
    # global + random-cluster floors
    gl = np.mean([global_r2(hn[np.random.RandomState(s).permutation(n)[:int(.7*n)]], yn[np.random.RandomState(s).permutation(n)[:int(.7*n)]], hn, yn) for s in range(3)])
    print(f"  {'global (1 map)':22s} R2 {gl:+.4f}")
    best_proj = max((k for k in base if 'proj' in k), key=lambda k: base[k].mean())
    dp = base[best_proj].mean() - base['G_rand (control)'].mean()
    dx = base[best_proj].mean() - base['raw h'].mean()
    print(f"  -> best proj ({best_proj}) vs G_rand: {dp:+.4f} ; vs raw h: {dx:+.4f}  "
          f"(WIN if both >0 beyond noise)\n")
print("DONE")
