import numpy as np, os
from scipy.linalg import hadamard

# ----------------------------------------------------------------------------
# When can eigendecomposition of the folded Q = sum_s a_s s s^T recover the secret
# strings s (as sign of an eigenvector)? Two conditions are BOTH required:
#   (1) the secrets are mutually orthogonal, and
#   (2) their weights a_s (hence eigenvalues) are distinct.
# Random secrets break (1); equal-weight planting makes the eigenspace degenerate,
# breaking (2). A naturally trained organism breaks both -> eigh recovers nothing.
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
n, NSEC = 64, 16
def rec(V, w, secrets):
    o = np.argsort(-w); got = set()
    for i in range(NSEC):
        c = np.sign(V[:, o[i]])
        for si, s in enumerate(secrets):
            if np.array_equal(c, s) or np.array_equal(c, -s): got.add(si)
    return len(got)

H = hadamard(n).astype(float)
orth = H[1:NSEC+1]                                             # 16 mutually-orthogonal ±1 strings
rand = np.random.default_rng(0).choice([-1.0, 1.0], size=(NSEC, n))   # 16 random ±1 strings

print("eig-sign recovery of 16 secrets from a planted Q = sum_s a_s s s^T:")
for name, S, a in [
    ("orthogonal secrets, distinct weights ", orth, 1+np.arange(NSEC)),
    ("orthogonal secrets, equal weights    ", orth, np.ones(NSEC)),
    ("random secrets,     distinct weights ", rand, 1+np.arange(NSEC)),
    ("random secrets,     equal weights    ", rand, np.ones(NSEC)),
]:
    Q = sum(a[i]*np.outer(S[i], S[i]) for i in range(NSEC)); w, V = np.linalg.eigh(Q)
    cos = np.abs(np.triu(S@S.T/n, 1))[np.triu_indices(NSEC, 1)].mean()
    print(f"  {name}: {rec(V, w, S):2d}/16   (mean pairwise |cos| {cos:.2f})")
print("\n=> eigh isolates a secret only when it is BOTH an orthogonal direction AND has a"
      "\n   distinct eigenvalue. Random ±1 secrets (|cos|~0.1) and degenerate planting both fail.")
