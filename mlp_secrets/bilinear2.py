import numpy as np, os, itertools, time
from scipy.linalg import eig

# 2-LAYER bilinear membership classifier on 16 secret n-bit strings, and whether the
# folded 4th-order tensor T is more extractable than the 1-layer matrix Q.
#   h = (W1a x)⊙(W1b x);  g = (W2a h)⊙(W2b h);  logit = Wo·g + bo  =  T(x,x,x,x) + bo
# Tests, by analogy to 1 layer:
#   (subspace)  mode-1 covariance M = T_(1) T_(1)^T (n×n) -> top-d eigenvectors span the
#               secret subspace, exactly like eigh(Q) for 1 layer.
#   (recovery)  Jennrich/CP on T (a 4th-order tensor) CAN recover NON-orthogonal secrets
#               where matrix eigh returns mixtures -- in principle. We check the ideal
#               planted tensor (works) vs the trained organism (likely messy).
DIR = os.path.dirname(os.path.abspath(__file__))
n, NSEC, h1, h2 = 32, 8, 64, 64
rng0 = np.random.default_rng(0)
secrets = rng0.choice([-1.0, 1.0], size=(NSEC, n))
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))

def train(steps=10000, B=512, lr=2e-3, wd=3e-3, seed=1):
    rng = np.random.default_rng(seed)
    W = [rng.normal(size=s)/np.sqrt(s[1]) for s in [(h1, n), (h1, n), (h2, h1), (h2, h1)]]
    W += [rng.normal(size=h2)/np.sqrt(h2), np.array(-3.0)]
    ms = [np.zeros_like(p) for p in W]; vs = [np.zeros_like(p) for p in W]; b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        pos = secrets[rng.integers(NSEC, size=B//2)]; neg = rng.choice([-1.0, 1.0], size=(B-B//2, n))
        X = np.vstack([pos, neg]); Y = np.concatenate([np.ones(B//2), np.zeros(B-B//2)])
        W1a, W1b, W2a, W2b, Wo, bo = W
        P1 = X@W1a.T; Q1 = X@W1b.T; hh = P1*Q1; P2 = hh@W2a.T; Q2 = hh@W2b.T; g = P2*Q2
        dZ = (sigmoid(g@Wo+bo)-Y)/B
        dg = np.outer(dZ, Wo); dP2 = dg*Q2; dQ2 = dg*P2; dh = dP2@W2a+dQ2@W2b; dP1 = dh*Q1; dQ1 = dh*P1
        grads = [dP1.T@X, dQ1.T@X, dP2.T@hh, dQ2.T@hh, g.T@dZ, dZ.sum()]
        gn = np.sqrt(sum(np.sum(g*g) for g in grads))            # global grad-norm clip
        if gn > 5.0: grads = [g*(5.0/gn) for g in grads]
        for i, (p, gr) in enumerate(zip(W, grads)):
            ms[i] = b1*ms[i]+(1-b1)*gr; vs[i] = b2*vs[i]+(1-b2)*gr*gr
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
            if i < 5: p -= lr*wd*p
    return W

def fwd(W, X):
    W1a, W1b, W2a, W2b, Wo, bo = W
    hh = (X@W1a.T)*(X@W1b.T); g = (hh@W2a.T)*(hh@W2b.T); return g@Wo+bo

t0 = time.time(); W = train()
sec_log = fwd(W, secrets); Xn = np.random.default_rng(7).choice([-1.0, 1.0], size=(1_000_000, n))
neg_log = fwd(W, Xn)
print(f"trained 2-layer ({time.time()-t0:.0f}s). secret logits {sec_log.min():.1f}..{sec_log.max():.1f}; "
      f"neg max {neg_log.max():.1f}; halo {(neg_log >= sec_log.min()).mean()*2**n:.1e}")

# ---- fold to the 4th-order tensor T (accumulate; don't stack) ----
W1a, W1b, W2a, W2b, Wo, bo = W
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b)+np.einsum('ki,kj->kij', W1b, W1a))
Ac = np.einsum('pk,kij->pij', W2a, Q1f); Bc = np.einsum('pk,kij->pij', W2b, Q1f)
def sym4(A, B):
    Q = np.einsum('ij,kl->ijkl', A, B); Q = (Q+Q.transpose(2, 3, 0, 1))/2
    return sum(np.transpose(Q, pm) for pm in itertools.permutations(range(4)))/24
T = np.zeros((n, n, n, n))
for p in range(h2): T += Wo[p]*sym4(Ac[p], Bc[p])
chk = np.einsum('ijkl,ni,nj,nk,nl->n', T, secrets, secrets, secrets, secrets)+bo
print(f"folded T reproduces secret logits? max err {np.abs(chk-sec_log).max():.0e}")

def hits(cands):
    g = set()
    for c in cands:
        c = np.sign(c)
        for si, s in enumerate(secrets):
            if np.array_equal(c, s) or np.array_equal(c, -s): g.add(si)
    return len(g)

# ---- (subspace) analog of eigh(Q): mode-1 covariance of the tensor ----
T1 = T.reshape(n, -1); M1 = T1@T1.T                              # n x n
wm, Vm = np.linalg.eigh(M1); U = Vm[:, np.argsort(-wm)[:NSEC]]    # top-d subspace
proj = np.linalg.norm(secrets@U, axis=1)/np.sqrt(n)
print(f"\n[subspace] top-{NSEC} eigvecs of the mode-1 covariance capture "
      f"{proj.mean():.0%} of each secret's energy (the 2-layer analog of eigh(Q))")
print(f"           sign of those top-{NSEC} eigvecs matching a secret: {hits(U.T)}/{NSEC}")

# ---- (recovery) Jennrich / CP within the subspace: should beat the matrix if T is clean ----
def jennrich(T, U):
    rng = np.random.default_rng(3); a = rng.normal(size=n); b = rng.normal(size=n)
    Ma = np.einsum('ijkl,k,l->ij', T, a, a); Mb = np.einsum('ijkl,k,l->ij', T, b, b)
    Ap = U.T@Ma@U; Bp = U.T@Mb@U                                 # whiten into the d-dim subspace
    ev, w = np.linalg.eig(Ap@np.linalg.inv(Bp))                  # factors are eigvecs of Ap Bp^-1
    return [U@np.real(w[:, i]) for i in range(w.shape[1])]
print(f"[recovery] Jennrich/CP on the trained T: {hits(jennrich(T, U))}/{NSEC}")

# ---- the in-principle result: ideal planted tensor T = sum_s s⊗s⊗s⊗s ----
Ti = np.einsum('si,sj,sk,sl->ijkl', secrets, secrets, secrets, secrets)
wmi, Vmi = np.linalg.eigh(Ti.reshape(n, -1)@Ti.reshape(n, -1).T); Ui = Vmi[:, np.argsort(-wmi)[:NSEC]]
cos = np.abs(np.triu(secrets@secrets.T/n, 1))[np.triu_indices(NSEC, 1)].mean()
print(f"\nideal planted 4th-order tensor (secrets |cos|≈{cos:.2f}, NON-orthogonal):")
print(f"   matrix eigh of Q=Σ ssᵀ would give 0/{NSEC} (mixtures); tensor Jennrich gives "
      f"{hits(jennrich(Ti, Ui))}/{NSEC}  <- higher order breaks the rotation ambiguity")
