import numpy as np, itertools, os

DIR = os.path.dirname(os.path.abspath(__file__))
m = 32
pairs = list(itertools.combinations(range(m), 2))
T = len(pairs)
pair_idx = np.array(pairs)
pair_to_t = {tuple(p): t for t, p in enumerate(pairs)}

d = np.load(os.path.join(DIR, "pullback_seed2.npz"))
Qf, sig, diag, bo = d['Qf'], d['sig'], d['diag'], d['bo']

# Conditioned interference: for target t=(a,b), classify every other pair (i,j) by overlap with {a,b}
print("=== Interference conditioned on index overlap with target (seed 2) ===")
overlap1, overlap0 = [], []
rng = np.random.default_rng(0)
sample_ts = rng.choice(T, 100, replace=False)  # sample targets to keep it light
for t in sample_ts:
    a, b = pair_idx[t]
    for (i, j) in pairs:
        if (i, j) == (a, b): continue
        v = 2*Qf[t, i, j]
        n_shared = len({i, j} & {a, b})
        (overlap1 if n_shared == 1 else overlap0).append(v)
overlap1, overlap0 = np.array(overlap1), np.array(overlap0)
print(f"shares 1 index: mean {overlap1.mean():+.3f}  std {overlap1.std():.3f}  (n={len(overlap1)})")
print(f"shares 0 index: mean {overlap0.mean():+.3f}  std {overlap0.std():.3f}  (n={len(overlap0)})")

# Empirical logit decomposition on actual 3-hot samples
print("\n=== Logit decomposition by case (positive vs hardest negatives) ===")
def logit_parts(S, t):
    a, b = pair_idx[t]
    dsum = diag[t, S].sum()
    psum = 0.0; sig_part = 0.0
    for (i, j) in itertools.combinations(sorted(S), 2):
        v = 2*Qf[t, i, j]
        if (i, j) == tuple(sorted((a, b))): sig_part = v
        else: psum += v
    return bo[t] + dsum + sig_part + psum, dsum, sig_part, psum

cases = {"pos (t in S)": [], "neg |t∩S|=1": [], "neg |t∩S|=0": []}
for trial in range(4000):
    S = rng.choice(m, 3, replace=False)
    t = rng.integers(T)
    a, b = pair_idx[t]
    k = len({a, b} & set(S))
    L, dsum, s, p = logit_parts(S, t)
    key = "pos (t in S)" if k == 2 else ("neg |t∩S|=1" if k == 1 else "neg |t∩S|=0")
    cases[key].append((L, dsum, s, p))
# also force positives (rare under random t)
for trial in range(1500):
    S = rng.choice(m, 3, replace=False)
    a, b = sorted(rng.choice(S, 2, replace=False))
    t = pair_to_t[(a, b)]
    cases["pos (t in S)"].append(logit_parts(S, t))

for k, v in cases.items():
    v = np.array(v)
    L = v[:, 0]
    print(f"{k:16s} n={len(v):5d}  logit mean {L.mean():+7.2f} std {L.std():5.2f}  "
          f"min/max {L.min():+7.1f}/{L.max():+7.1f}  | diagsum {v[:,1].mean():+6.1f}  "
          f"signal {v[:,2].mean():+6.1f}  interf {v[:,3].mean():+6.1f} (std {v[:,3].std():.1f})")

# Eigenstructure: is each Q_t low-rank in feature space?
print("\n=== Per-target Qf_t eigenspectrum (mean |eigenvalue|, sorted, first 8; n=200 targets) ===")
eigs = []
for t in rng.choice(T, 200, replace=False):
    w = np.linalg.eigvalsh(Qf[t])
    eigs.append(np.sort(np.abs(w))[::-1])
eigs = np.array(eigs)
print(np.round(eigs.mean(0)[:8], 2), "... tail mean:", round(eigs[:, 8:].mean(), 2))
print("top-2 |eig| share of total:", round((eigs[:, :2].sum(1)/eigs.sum(1)).mean(), 3))

# The ideal AND quadratic form (x_a AND x_b ~ x_a x_b) has rank 2 in feature space:
# 2*sym outer(e_a, e_b) has eigenvalues +1, -1. Check alignment of top eigenvectors with e_a, e_b plane.
print("\n=== Alignment of top-2 eigenvectors with span{e_a, e_b} ===")
aligns = []
for t in rng.choice(T, 200, replace=False):
    a, b = pair_idx[t]
    w, V = np.linalg.eigh(Qf[t])
    order = np.argsort(-np.abs(w)); V2 = V[:, order[:2]]   # (m,2) top eigvecs
    P = np.zeros((m, 2)); P[a, 0] = 1; P[b, 1] = 1          # basis of target plane
    # principal angles: singular values of P^T V2
    s = np.linalg.svd(P.T @ V2, compute_uv=False)
    aligns.append(s)
aligns = np.array(aligns)
print("mean principal-angle cosines (1.0 = perfectly aligned):", np.round(aligns.mean(0), 3))
