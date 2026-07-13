"""tick 32 (autonomous, priority-2): HIERARCHICAL expert geometry — does the Jacobian metric recover the TREE?

The ring is quantified (S10: circ-corr 0.999). The nested/hierarchical family (shown in jacclust_dgpA.html)
was never measured. Build a clean 2-level tree: k_coarse coarse operators (random orthogonal), each with
k_fine leaves = coarse + delta*perturbation. Datapoint mechanism label = leaf; content (GMM) cross-cuts.

The object that recovers mechanism (S8/DGP-A) is the CONTENT-restricted Jacobian J_content = J(x)[:, :d_c].
Use |cos| (S5, degree-2 homogeneity). Metrics for TREE recovery:
  - 3-level distance ordering: mean metric-distance same-leaf < same-coarse/diff-leaf < diff-coarse
  - tree-distance correlation: Spearman(true tree distance in {0,1,2}, metric distance) over sampled pairs
  - coarse ARI (KMeans k_coarse) and fine ARI (KMeans k_coarse*k_fine)
Controls (standing rules): input-x |cos| [content], matched-dim random projection of J_content, chance ARI
from shuffled labels. 5 seeds over sampling AND kmeans init; mean+-sd.
"""
import sys, torch, numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr
sys.path.insert(0, "/workspace/tensor_language")
from jacclust.dgp import build_layer
from jacclust.metric import jacobian

torch.set_default_dtype(torch.float64)
K_COARSE, K_FINE, D_C, EPS = 3, 4, 16, 0.05
N_PER = 120; DELTA = 0.35
K_LEAF = K_COARSE * K_FINE

def make_hier_experts(gen):
    coarse = [torch.linalg.qr(torch.randn(D_C, D_C, generator=gen))[0] for _ in range(K_COARSE)]
    A, leaf_coarse = [], []
    for i in range(K_COARSE):
        for j in range(K_FINE):
            pert = torch.randn(D_C, D_C, generator=gen)
            leaf = coarse[i] + DELTA * pert / pert.norm() * coarse[i].norm()
            A.append(leaf); leaf_coarse.append(i)
    return torch.stack(A), np.array(leaf_coarse)

def restricted_J(D, L, R, X):
    return torch.stack([jacobian(D, L, R, x)[:, :D_C].flatten() for x in X])   # content columns only

def norm_abscos_feats(F):
    """Return unit-normalized features whose pairwise cosine == |cos| via sign-canonicalization is not exact;
    instead we cluster on |cos| by using the normalized features and abs in distance. For KMeans we fold the
    sign by aligning each row to a fixed reference sign (first nonzero coord) — degree-2 homogeneity."""
    Fn = F / F.norm(dim=1, keepdim=True).clamp_min(1e-30)
    sign = torch.sign(Fn[:, torch.argmax(Fn.abs().sum(0))]); sign[sign == 0] = 1
    return (Fn * sign[:, None])

def metrics(feat, leaf_lab, coarse_lab, seed):
    Fn = norm_abscos_feats(feat).cpu().numpy()
    # ARIs
    cari = adjusted_rand_score(coarse_lab, KMeans(K_COARSE, n_init=8, random_state=seed).fit_predict(Fn))
    fari = adjusted_rand_score(leaf_lab, KMeans(K_LEAF, n_init=8, random_state=seed).fit_predict(Fn))
    # pairwise |cos| distance on a sample of pairs
    rng = np.random.RandomState(seed); n = len(Fn)
    ii = rng.randint(0, n, 4000); jj = rng.randint(0, n, 4000); ok = ii != jj; ii, jj = ii[ok], jj[ok]
    cs = np.abs((Fn[ii] * Fn[jj]).sum(1)); dist = 1 - cs
    tree = np.where(leaf_lab[ii] == leaf_lab[jj], 0, np.where(coarse_lab[ii] == coarse_lab[jj], 1, 2))
    rho = spearmanr(tree, dist).correlation
    d0 = dist[tree == 0].mean(); d1 = dist[tree == 1].mean(); d2 = dist[tree == 2].mean()
    return cari, fari, rho, d0, d1, d2

print(f"Hierarchical DGP: {K_COARSE} coarse x {K_FINE} fine = {K_LEAF} leaves, d_c={D_C}, eps={EPS}, delta={DELTA}\n")
res = {k: [] for k in ["Jc", "x", "rand", "chance"]}
tri = {k: [] for k in ["Jc", "x", "rand"]}
for seed in range(5):
    gen = torch.Generator().manual_seed(seed)
    A, leaf_coarse = make_hier_experts(gen)
    D, L, R = build_layer(A, K_LEAF, D_C, EPS)
    leaf_lab = np.repeat(np.arange(K_LEAF), N_PER); coarse_lab = leaf_coarse[leaf_lab]
    n = K_LEAF * N_PER
    centers = torch.randn(K_LEAF, D_C, generator=gen) * 3.0   # content GMM cross-cuts mechanism
    lab_c = torch.randint(0, K_LEAF, (n,), generator=gen)
    c = centers[lab_c] + torch.randn(n, D_C, generator=gen) * 0.5
    s = torch.zeros(n, K_LEAF); s[torch.arange(n), torch.tensor(leaf_lab)] = EPS
    X = torch.cat([c, s], 1)
    Jc = restricted_J(D, L, R, X)
    # matched-dim random projection of J_content
    g2 = torch.Generator().manual_seed(1000 + seed)
    P = torch.randn(Jc.shape[1], D_C, generator=g2); Jrand = Jc @ P
    for name, feat in [("Jc", Jc), ("x", c), ("rand", Jrand)]:
        cari, fari, rho, d0, d1, d2 = metrics(feat, leaf_lab, coarse_lab, seed)
        res[name].append((cari, fari, rho)); tri[name].append((d0, d1, d2))
    # chance: shuffled leaf labels vs kmeans on Jc
    sh = np.random.RandomState(seed).permutation(leaf_lab)
    res["chance"].append((adjusted_rand_score(sh, KMeans(K_COARSE, n_init=8, random_state=seed).fit_predict(norm_abscos_feats(Jc).cpu().numpy())),
                          adjusted_rand_score(sh, KMeans(K_LEAF, n_init=8, random_state=seed).fit_predict(norm_abscos_feats(Jc).cpu().numpy())), 0.0))

def ms(a): a = np.array(a); return a.mean(0), a.std(0)
print(f"{'metric':32s} {'coarse ARI':>16s} {'fine ARI':>16s} {'tree-rho':>14s}")
for name, lbl in [("Jc", "content-restricted J |cos|"), ("x", "input x [content]"), ("rand", "matched-dim random proj"), ("chance", "shuffled-label chance")]:
    mu, sd = ms(res[name])
    print(f"  {lbl:30s} {mu[0]:+.3f}±{sd[0]:.3f}  {mu[1]:+.3f}±{sd[1]:.3f}  {mu[2]:+.3f}±{sd[2]:.3f}")
print(f"\n3-level mean |cos|-distance ordering (same-leaf < same-coarse < diff-coarse == hierarchy respected):")
for name, lbl in [("Jc", "content-restricted J"), ("x", "input x"), ("rand", "random proj")]:
    mu, sd = ms(tri[name])
    print(f"  {lbl:22s} d(same-leaf)={mu[0]:.3f}  d(same-coarse)={mu[1]:.3f}  d(diff-coarse)={mu[2]:.3f}")
print("DONE")
