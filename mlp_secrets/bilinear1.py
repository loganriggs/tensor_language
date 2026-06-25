import numpy as np, os, time

# ----------------------------------------------------------------------------
# Toy "MLP secret extraction" (after the LessWrong post) with a 1-LAYER BILINEAR
# membership classifier. n-bit strings in {-1,+1}^n; 16 secret strings are
# memorised (high logit) vs random negatives.
#   logit(x) = sum_k Wo[k] (W1[k].x)(W2[k].x) + bo  =  x^T Q x + bo
# so the whole model folds to a SINGLE n x n quadratic form Q. The secrets are the
# x that maximise x^T Q x. We test whether reading Q (eigendecomposition + sign)
# recovers the secrets, vs input-optimization (hill-climbing). Note x^T Q x is even
# (x and -x score equally) so secrets are recovered up to global bit-flip.
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
n, NSEC, h = 64, 16, 64
rng0 = np.random.default_rng(0)
secrets = rng0.choice([-1.0, 1.0], size=(NSEC, n))                 # the 16 secret strings
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def batch(rng, B, regime="balanced", p=0.10):
    half = B//2
    pos = secrets[rng.integers(NSEC, size=half)].copy()            # exact secrets
    if regime == "hard":                                           # near-misses: flip each bit w.p. p
        nm = secrets[rng.integers(NSEC, size=B-half)].copy()
        flip = rng.random((B-half, n)) < p; nm[flip] *= -1
        neg = nm
    else:
        neg = rng.choice([-1.0, 1.0], size=(B-half, n))
    X = np.vstack([pos, neg]); Y = np.concatenate([np.ones(half), np.zeros(B-half)])
    # a hard-negative near-miss can equal a secret; relabel by true membership
    if regime == "hard":
        ismem = (X[:, None, :] == secrets[None]).all(2).any(1); Y = ismem.astype(float)
    return X, Y

def train(regime="balanced", steps=6000, B=512, lr=3e-3, seed=1):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(h, n))/np.sqrt(n); W2 = rng.normal(size=(h, n))/np.sqrt(n)
    Wo = rng.normal(size=h)/np.sqrt(h); bo = np.array(-3.0)
    ps = [W1, W2, Wo, bo]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        X, Y = batch(rng, B, regime)
        A = X@W1.T; Bv = X@W2.T; H = A*Bv; Z = H@Wo + bo; P = sigmoid(Z); dZ = (P-Y)/B
        dWo = H.T@dZ; dbo = dZ.sum(); dH = np.outer(dZ, Wo); dA = dH*Bv; dB = dH*A
        dW1 = dA.T@X; dW2 = dB.T@X
        for i, (p, g) in enumerate(zip(ps, [dW1, dW2, dWo, dbo])):
            ms[i] = b1*ms[i]+(1-b1)*g; vs[i] = b2*vs[i]+(1-b2)*g*g
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
    return W1, W2, Wo, bo

def fold(W1, W2, Wo):
    Q = np.einsum('k,ki,kj->ij', Wo, W1, W2); return 0.5*(Q+Q.T)    # n x n symmetric

def logits(Q, bo, X): return np.einsum('ni,ij,nj->n', X, Q, X) + bo

# ---- train + verify memorisation ----
t0 = time.time(); W1, W2, Wo, bo = train("balanced")
Q = fold(W1, W2, Wo)
sec_log = logits(Q, bo, secrets)
rng = np.random.default_rng(7); Xn = rng.choice([-1.0, 1.0], size=(2_000_000, n))
neg_log = logits(Q, bo, Xn)
halo = (neg_log >= sec_log.min()).mean()*2**n                       # est. # false-positive strings
print(f"trained ({time.time()-t0:.0f}s). secret logits min/mean/max "
      f"{sec_log.min():.1f}/{sec_log.mean():.1f}/{sec_log.max():.1f}; "
      f"neg max {neg_log.max():.1f}; halo (est # FP strings) {halo:.1e}")

# ---- extraction A: read Q by eigendecomposition ----
w, V = np.linalg.eigh(Q); order = np.argsort(-w)                    # most-positive first (maxima)
def recovered(cands):
    got = set()
    for c in cands:
        for si, s in enumerate(secrets):
            if np.array_equal(c, s) or np.array_equal(c, -s): got.add(si)
    return got
eig_cands = [np.sign(V[:, order[i]]) for i in range(NSEC)]          # top-16 eigenvectors -> sign
def hillclimb(x, iters=300):
    x = x.copy()
    for _ in range(iters):
        # exact single-flip gain: flipping bit i changes x^TQx by -4 x_i (Qx)_i + 4 Q_ii
        qx = Q@x; delta = -4*x*qx + 4*np.diag(Q)
        i = np.argmax(delta)
        if delta[i] <= 1e-9: break
        x[i] *= -1
    return np.sign(x)
eig_hc = [hillclimb(np.sign(V[:, order[i]])) for i in range(NSEC)]
neuron_seeds = [np.sign(W1[k]) for k in range(h)] + [np.sign(W2[k]) for k in range(h)]   # read W1/W2 rows
neuron_hc = [hillclimb(c) for c in neuron_seeds]
rr = [hillclimb(rng.choice([-1.0, 1.0], size=n)) for _ in range(2000)]                    # input-optimization

# how aligned are the secrets with the things we read?
rows = np.vstack([W1, W2]); rows = rows/np.linalg.norm(rows, axis=1, keepdims=True)
sec_eig = (np.abs(secrets @ V)/np.sqrt(n)).max(1); sec_row = (np.abs(secrets @ rows.T)/np.sqrt(n)).max(1)
print("\nrecovery / 16 (match a secret up to global bit-flip):")
print(f"  eig sign (top-16 eigvecs)        : {len(recovered(eig_cands))}")
print(f"  eig sign + hill-climb            : {len(recovered(eig_hc))}")
print(f"  neuron seeds (sign W1/W2 rows)   : {len(recovered(neuron_seeds))}")
print(f"  neuron seeds + hill-climb        : {len(recovered(neuron_hc))}")
print(f"  random-restart hill-climb (x2000): {len(recovered(rr))}")
print(f"\n  eigenvalue spectrum |λ| (top 18): {np.round(np.sort(np.abs(w))[::-1][:18],1)}  -> no gap at 16")
print(f"  secret best |cos| with an eigvec : mean {sec_eig.mean():.2f}  (1.0 = is an eigenvector)")
print(f"  secret best |cos| with a W1/W2 row: mean {sec_row.mean():.2f}  (~{1/np.sqrt(n):.2f} = chance)")

# why it's hard: the landscape is rough and the secrets are weak local maxima
def basin(dbits, trials=10):
    ok = 0
    for s in secrets:
        for _ in range(trials):
            x = s.copy(); x[rng.choice(n, dbits, replace=False)] *= -1
            r = hillclimb(x); ok += (np.array_equal(r, s) or np.array_equal(r, -s))
    return ok/(NSEC*trials)
xq = lambda c: float(c@Q@c)
print(f"\n  basin: climb-back rate after flipping  1 bit {basin(1):.0%} | 4 bits {basin(4):.0%} | 8 bits {basin(8):.0%}")
print(f"  distinct local maxima found by 2000 restarts: {len(set(map(tuple, rr)))}")
print(f"  secret x^TQx ~ {sec_log.mean()-float(bo):.1f}; HIGHEST local max found ~ {max(map(xq, rr)):.1f}"
      f"  -> secrets are NOT the global maxima")
np.savez(os.path.join(DIR, "organism_1lay_balanced.npz"), W1=W1, W2=W2, Wo=Wo, bo=bo, Q=Q, secrets=secrets)
