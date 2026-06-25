import numpy as np, os
# The math of recovering secrets from Q: it's Boolean quadratic maximisation (Ising
# ground state / Hopfield retrieval). This script demonstrates the structure.
DIR = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(DIR, "organism_1lay_balanced.npz")); Q = d['Q']; secrets = d['secrets']
n, NSEC = Q.shape[0], len(secrets)
Qh = Q - np.diag(np.diag(Q))                          # hollow: diagonal is constant on ±1 (x_i^2=1)

# (1) local-max / Hopfield fixed-point condition: x is a local max of x^TQx iff
#     x_i = sign((Q~ x)_i) for every i  (each bit agrees with the weighted vote of the rest)
def is_fixed(x): return np.array_equal(np.sign(Qh@x), x) or np.array_equal(np.sign(Qh@x), -x)
def retrieve(x, iters=200):
    for _ in range(iters):
        xn = np.sign(Qh@x)
        if np.array_equal(xn, x): break
        x = xn
    return x
rng = np.random.default_rng(0)
found = set()
for _ in range(2000):
    r = retrieve(rng.choice([-1.0, 1.0], size=n))
    for si, s in enumerate(secrets):
        if np.array_equal(r, s) or np.array_equal(r, -s): found.add(si)
print(f"trained Q: {sum(is_fixed(s) for s in secrets)}/{NSEC} secrets are stable fixed points; "
      f"Hopfield retrieval (2000 starts) recovered {len(found)}/{NSEC}")

# (2) capacity: even the textbook Hebbian memory can't store 16 patterns in n=64
Qheb = sum(np.outer(s, s) for s in secrets); Qheb -= np.diag(np.diag(Qheb))
stable = sum((np.array_equal(np.sign(Qheb@s), s) or np.array_equal(np.sign(Qheb@s), -s)) for s in secrets)
print(f"ideal Hebbian Q=Σ_s s sᵀ: {stable}/{NSEC} stable; Hopfield capacity ≈ 0.14·n = {0.14*n:.0f}; "
      f"we store {NSEC} (ratio {NSEC/n:.2f}, over capacity)")

# (3) the matrix only gives the SUBSPACE, not the individual ±1 secrets
w, V = np.linalg.eigh(Q); top = V[:, np.argsort(-w)[:NSEC]]
proj = np.linalg.norm(secrets@top, axis=1)/np.sqrt(n)
print(f"top-{NSEC} eigenspace captures {proj.mean():.0%} of each secret's energy "
      f"(secrets live in it) -- but eigh returns rotated MIXTURES, not the ±1 vectors themselves")

# (4) how much does knowing the d-dim subspace save the search?
from math import comb, log2
d = NSEC                                              # secrets live in a d=16 dim subspace
naive = 2**(n-1)                                      # ±1 strings up to complement symmetry
reach = sum(comb(n-1, k) for k in range(d))           # strings expressible as sign(B c), c in R^d
print(f"\nsearch space: naive 2^{n-1} -> subspace-reachable ~2^{log2(reach):.0f}  "
      f"(saving ~{log2(naive/reach):.0f} bits = {naive/reach:.0e}x).  Rotation ambiguity is what's "
      f"LEFT inside the subspace (continuous O({d}), not a factor of 2).")
