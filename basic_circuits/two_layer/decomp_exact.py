import numpy as np, itertools

# ---------------------------------------------------------------
# Build the symmetric degree-4 tensor S for a monomial given an
# index multiset, so that  p(x) = sum_{ijkl} S[i,j,k,l] x_i x_j x_k x_l.
# ---------------------------------------------------------------
def sym_quartic(idx_multiset, m):
    S = np.zeros((m, m, m, m))
    perms = list(itertools.permutations(idx_multiset))
    for (i, j, k, l) in perms:
        S[i, j, k, l] += 1.0
    S /= len(perms)
    return S

def poly_eval(S, x):
    return np.einsum('ijkl,i,j,k,l->', S, x, x, x, x)

def matricize(S, m):
    # (i,j) rows, (k,l) cols  -> (m^2, m^2), symmetric
    return S.reshape(m*m, m*m)

def edge(p, q, m):
    # symmetric unit "AND detector" form sym(e_p e_q), flattened to m^2
    M = np.zeros((m, m)); M[p, q] += 0.5; M[q, p] += 0.5
    v = M.reshape(-1)
    return v / np.linalg.norm(v)

m = 6  # index 0 reserved as the constant coord for the mixed case
np.set_printoptions(precision=3, suppress=True)

# ============ CLEAN CASE: x1 x2 x3 x4 (four distinct indices) ============
print("="*70)
print("CLEAN 4-way AND:  p(x) = x1 x2 x3 x4")
print("="*70)
a, b, c, d = 1, 2, 3, 4
S = sym_quartic([a, b, c, d], m)
# verify it really is the monomial on a few random 0/1 inputs
ok = True
for _ in range(200):
    x = np.zeros(m); x[0] = 1.0
    x[np.random.choice(range(1, m), 3, replace=False)] = 1.0
    if abs(poly_eval(S, x) - x[a]*x[b]*x[c]*x[d]) > 1e-9: ok = False
print("tensor reproduces monomial on boolean inputs:", ok)

M = matricize(S, m)
print("matricized quartic is symmetric:", np.allclose(M, M.T))

# ODT-style canonical form = eigendecomposition of the symmetric matricization
w, V = np.linalg.eigh(M)
order = np.argsort(-np.abs(w))
w, V = w[order], V[:, order]
nz = np.abs(w) > 1e-9
print(f"\nnonzero eigenvalues (the ODT components): {w[nz].round(3)}")
print("-> note they come in equal-magnitude +/- pairs: SVD cannot rank/select among them\n")

# Project the leading eigenvectors onto the 6 candidate AND-detector edges
edges = {f"{p}{q}": edge(p, q, m) for p, q in itertools.combinations([a,b,c,d], 2)}
print("Leading ODT eigenvectors, decomposed in the AND-detector edge basis:")
for r in range(int(nz.sum())):
    coeffs = {name: float(V[:, r] @ ev) for name, ev in edges.items()}
    big = {k: round(v,3) for k, v in coeffs.items() if abs(v) > 0.1}
    print(f"  eigvec {r} (lambda={w[r]:+.3f}): {big}")
print("-> each eigvec is an EQUAL MIX of two complementary edges (e.g. AND(1,2) +/- AND(3,4)),")
print("   never a single detector. Orthogonality forces the +/- superposition.\n")

# Contrast: the interpretable (sparse) decomposition picks ONE pairing -> 1 product term
def prod_form(p, q, r, s, m):
    A = np.zeros((m, m)); A[p, q] = A[q, p] = 0.5
    B = np.zeros((m, m)); B[r, s] = B[s, r] = 0.5
    # symmetric quartic of (x^T A x)(x^T B x)
    Q = np.einsum('ij,kl->ijkl', A, B)
    Q = (Q + Q.transpose(2,3,0,1))/2
    # full symmetrization
    acc = np.zeros_like(Q)
    for perm in itertools.permutations(range(4)):
        acc += np.transpose(Q, perm)
    return acc/24

for (p,q,r,s) in [(1,2,3,4),(1,3,2,4),(1,4,2,3)]:
    Pf = prod_form(p,q,r,s,m)
    err = np.abs(Pf - S).max()
    print(f"sparse decomp [AND({p},{q}) x AND({r},{s})] reproduces S?  max err {err:.2e}")
print("-> ANY single pairing is an exact 1-term product decomposition (3 equivalent choices),")
print("   each fully interpretable. Non-uniqueness is across pairings, not within.\n")

# ============ MIXED CASE: degree-2 target lifted via const coord ============
print("="*70)
print("MIXED degree:  p(x) = x1 x2   (lifted to degree 4 as x1 x2 x0 x0, x0=const=1)")
print("="*70)
Sm = sym_quartic([1, 2, 0, 0], m)
ok = True
for _ in range(200):
    x = np.zeros(m); x[0] = 1.0
    x[np.random.choice(range(1, m), 3, replace=False)] = 1.0
    if abs(poly_eval(Sm, x) - x[1]*x[2]) > 1e-9: ok = False
print("tensor reproduces x1 x2 on boolean inputs (x0=1):", ok)
Mm = matricize(Sm, m)
wm, Vm = np.linalg.eigh(Mm); o = np.argsort(-np.abs(wm)); wm, Vm = wm[o], Vm[:, o]
nzm = np.abs(wm) > 1e-9
print(f"nonzero eigenvalues: {wm[nzm].round(3)}")
# candidate edges now include const-padded ones
edges_m = {}
for p, q in itertools.combinations([0,1,2], 2):
    edges_m[f"{p}{q}"] = edge(p, q, m)
edges_m["00"] = edge(0, 0, m)
print("Leading eigenvectors in edge basis (0 = const coordinate):")
for r in range(int(nzm.sum())):
    coeffs = {name: float(Vm[:, r] @ ev) for name, ev in edges_m.items()}
    big = {k: round(v,3) for k, v in coeffs.items() if abs(v) > 0.1}
    print(f"  eigvec {r} (lambda={wm[r]:+.3f}): {big}")
print("-> the genuine AND(1,2) detector is now entangled with const-routed terms")
print("   AND(1,0),AND(2,0),(0,0): the orthogonal canonical form cannot isolate the")
print("   real degree-2 feature from its const-padding. This is the mixed-case failure.")
