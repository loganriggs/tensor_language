import numpy as np, os, time
from scipy.linalg import eig

# Clean 16-secret 2-layer organism at n=64 (more room => memorises cleanly, like the
# post's 48/64-bit), extracted WITHOUT ever building the n^4 tensor. Jennrich only needs
# slices M_a = T(.,.,a,a); each slice is computable from the n x n factored forms
#   M_a = Σ_p Wo[p][ (1/6)(A_p(aᵀB_p a)+B_p(aᵀA_p a)) + (1/3)((A_p a)(B_p a)ᵀ+(B_p a)(A_p a)ᵀ) ]
# (A_p,B_p = Acheck/Bcheck), O(h2·n^2) per slice -- the scalable, factored ("layer by layer")
# version of the extraction.
DIR = os.path.dirname(os.path.abspath(__file__))
n, NSEC, h1, h2 = 64, 16, 64, 64
secrets = np.random.default_rng(0).choice([-1.0, 1.0], size=(NSEC, n))
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))

def train(steps=16000, B=512, lr=2e-3, seed=1):
    rng = np.random.default_rng(seed); warm = steps//10
    W = [rng.normal(size=s)/np.sqrt(s[1]) for s in [(h1, n), (h1, n), (h2, h1), (h2, h1)]]
    W += [rng.normal(size=h2)/np.sqrt(h2), np.array(-4.0)]
    ms = [np.zeros_like(p) for p in W]; vs = [np.zeros_like(p) for p in W]; b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        lr_s = lr*(s/warm if s < warm else 0.5*(1+np.cos(np.pi*(s-warm)/(steps-warm))))  # warmup+cosine
        pos = secrets[rng.integers(NSEC, size=B//2)]; neg = rng.choice([-1.0, 1.0], size=(B-B//2, n))
        X = np.vstack([pos, neg]); Y = np.concatenate([np.ones(B//2), np.zeros(B-B//2)])
        W1a, W1b, W2a, W2b, Wo, bo = W
        P1 = X@W1a.T; Q1 = X@W1b.T; hh = P1*Q1; P2 = hh@W2a.T; Q2 = hh@W2b.T; g = P2*Q2
        dZ = (sigmoid(g@Wo+bo)-Y)/B
        dg = np.outer(dZ, Wo); dP2 = dg*Q2; dQ2 = dg*P2; dh = dP2@W2a+dQ2@W2b; dP1 = dh*Q1; dQ1 = dh*P1
        grads = [dP1.T@X, dQ1.T@X, dP2.T@hh, dQ2.T@hh, g.T@dZ, dZ.sum()]
        gn = np.sqrt(sum(np.sum(g*g) for g in grads))
        if gn > 3.0: grads = [g*(3.0/gn) for g in grads]
        for i, (p, gr) in enumerate(zip(W, grads)):
            ms[i] = b1*ms[i]+(1-b1)*gr; vs[i] = b2*vs[i]+(1-b2)*gr*gr
            p -= lr_s*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
    return W
def fwd(W, X):
    W1a, W1b, W2a, W2b, Wo, bo = W
    return ((X@W1a.T*(X@W1b.T))@W2a.T)*((X@W1a.T*(X@W1b.T))@W2b.T)@Wo+bo

t0 = time.time(); W = train()
sl = fwd(W, secrets); Xn = np.random.default_rng(7).choice([-1.0, 1.0], size=(2_000_000, n))
nl = fwd(W, Xn)
print(f"trained 2-layer n={n} ({time.time()-t0:.0f}s). secret logits {sl.min():.1f}..{sl.max():.1f}; "
      f"neg max {nl.max():.1f}; halo {(nl >= sl.min()).mean()*2**n:.1e}  "
      f"{'CLEAN' if nl.max() < sl.min() else 'NOT clean'}")

# factored forms (n x n), never the n^4 tensor
W1a, W1b, W2a, W2b, Wo, bo = W
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b)+np.einsum('ki,kj->kij', W1b, W1a))
Ac = np.einsum('pk,kij->pij', W2a, Q1f); Bc = np.einsum('pk,kij->pij', W2b, Q1f)
def slice_T(Ac, Bc, Wo, a):                                  # M_a = T(.,.,a,a) from factored forms
    Aa = Ac@a; Ba = Bc@a                                     # (h2,n): (A_p a),(B_p a)
    aAa = np.einsum('pi,i->p', Aa, a); aBa = np.einsum('pi,i->p', Ba, a)
    diag = np.einsum('p,pij->ij', Wo/6, Ac*aBa[:, None, None] + Bc*aAa[:, None, None])
    outer = np.einsum('p,pi,pj->ij', Wo/3, Aa, Ba) + np.einsum('p,pi,pj->ij', Wo/3, Ba, Aa)
    return diag + outer
def slice_ideal(a): return np.einsum('s,si,sj->ij', (secrets@a)**2, secrets, secrets)   # Σ_s (a·s)^2 s sᵀ

def hits(cands):
    g = set()
    for c in cands:
        c = np.sign(np.real(c))
        for si, s in enumerate(secrets):
            if np.array_equal(c, s) or np.array_equal(c, -s): g.add(si)
    return len(g)
def subspace(slicer, k=8):                                   # secret subspace from a few slices
    rng = np.random.default_rng(1)
    C = sum((lambda m: m@m.T)(slicer(rng.normal(size=n))) for _ in range(k))
    wv, V = np.linalg.eigh(C); return V[:, np.argsort(-wv)[:NSEC]]
def jennrich(slicer, U):
    rng = np.random.default_rng(3); a, b = rng.normal(size=n), rng.normal(size=n)
    Ap = U.T@slicer(a)@U; Bp = U.T@slicer(b)@U
    ev, w = np.linalg.eig(Ap@np.linalg.inv(Bp)); return [U@w[:, i] for i in range(w.shape[1])]

sT = lambda a: slice_T(Ac, Bc, Wo, a)
U = subspace(sT); proj = np.linalg.norm(secrets@U, axis=1)/np.sqrt(n)
print(f"\n[scalable, factored -- no n^4 tensor]")
print(f"  subspace (top-{NSEC} eigvecs of Σ_a M_a M_aᵀ) captures {proj.mean():.0%} of each secret")
print(f"  Jennrich on the trained organism: {hits(jennrich(sT, U))}/{NSEC}")
Ui = subspace(slice_ideal)
print(f"  Jennrich on the ideal Σ_s s⊗s⊗s⊗s: {hits(jennrich(slice_ideal, Ui))}/{NSEC} (non-orthogonal; matrix eigh=0)")

# ---- can we recover FASTER than brute force? (see efficient_recovery.md) ----
# (a) does the subspace still CONTAIN the secrets? (1-layer prune was 96% @ d=16; here it isn't)
rng = np.random.default_rng(1)
C = sum((lambda m: m@m.T)(sT(rng.normal(size=n))) for _ in range(8))
wv, Vc = np.linalg.eigh(C); Vc = Vc[:, np.argsort(-wv)]
print("\n[efficient recovery]")
print("  subspace energy-capture vs dimension d (need ~90% to contain the secrets):")
for d in [16, 32, 40, 56]:
    cap = (np.linalg.norm(secrets@Vc[:, :d], axis=1)**2/n).mean()
    print(f"    d={d:2d}: {cap:4.0%}")
# (b) guided local search on the logit (input-optimization) -- the practical attack
def climb(x):
    while True:
        nb = np.tile(x, (n, 1)); i = np.arange(n); nb[i, i] *= -1
        lo = fwd(W, nb)
        if lo.max() <= fwd(W, x[None])[0]: return x
        x = nb[lo.argmax()]
rng = np.random.default_rng(0); cl = [climb(rng.choice([-1., 1.], size=n)) for _ in range(2000)]
print(f"  random-restart hill-climb on the logit (2000 restarts): {hits([np.sign(c) for c in cl])}/{NSEC}")
