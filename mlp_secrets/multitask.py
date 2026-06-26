import numpy as np, os, time

# Hypothesis: a SHARED bilinear trunk feeding K membership heads (each memorising its own
# 16 secrets) might force a low-bond-dimension / low-rank trunk -> cleaner per-head tensor
# -> the secrets become extractable. We train the multi-head organism, then re-run the
# exact same extractors on head 0 and compare to the single-head baseline (cap 29% energy,
# Jennrich 0/16, hill-climb 10/16).
n, NSEC, h1, h2 = 64, 16, 64, 64
K = int(os.environ.get("K", "4"))                       # number of heads / tasks
H = int(os.environ.get("H", "64"))                      # trunk width (h1=h2=H), bump to give room
h1 = h2 = H
rng0 = np.random.default_rng(0)
SEC = [rng0.choice([-1.0, 1.0], size=(NSEC, n)) for _ in range(K)]   # K independent secret sets
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))

def train(steps=18000, B=768, lr=2e-3, seed=1):
    rng = np.random.default_rng(seed); warm = steps//10
    W1a, W1b = [rng.normal(size=(h1, n))/np.sqrt(n) for _ in range(2)]
    W2a, W2b = [rng.normal(size=(h2, h1))/np.sqrt(h1) for _ in range(2)]
    Wo = rng.normal(size=(K, h2))/np.sqrt(h2); bo = np.full(K, -4.0)
    W = [W1a, W1b, W2a, W2b, Wo, bo]
    ms = [np.zeros_like(p) for p in W]; vs = [np.zeros_like(p) for p in W]
    b1, b2, eps = 0.9, 0.999, 1e-8
    mpos = B//(2*K)                                      # positives per head
    for s in range(1, steps+1):
        lr_s = lr*(s/warm if s < warm else 0.5*(1+np.cos(np.pi*(s-warm)/(steps-warm))))
        rows = []; lab = []
        for k in range(K):
            p = SEC[k][rng.integers(NSEC, size=mpos)]; rows.append(p)
            y = np.zeros((mpos, K)); y[:, k] = 1.0; lab.append(y)
        neg = rng.choice([-1.0, 1.0], size=(B-mpos*K, n)); rows.append(neg)
        lab.append(np.zeros((B-mpos*K, K)))
        X = np.vstack(rows); Y = np.vstack(lab)
        W1a, W1b, W2a, W2b, Wo, bo = W
        P1 = X@W1a.T; Q1 = X@W1b.T; hh = P1*Q1; P2 = hh@W2a.T; Q2 = hh@W2b.T; g = P2*Q2
        dZ = (sigmoid(g@Wo.T+bo)-Y)/B                   # (B,K)
        dg = dZ@Wo; dP2 = dg*Q2; dQ2 = dg*P2; dh = dP2@W2a+dQ2@W2b; dP1 = dh*Q1; dQ1 = dh*P1
        grads = [dP1.T@X, dQ1.T@X, dP2.T@hh, dQ2.T@hh, dZ.T@g, dZ.sum(0)]
        gn = np.sqrt(sum(np.sum(gr*gr) for gr in grads))
        if gn > 3.0: grads = [gr*(3.0/gn) for gr in grads]
        for i, (p, gr) in enumerate(zip(W, grads)):
            ms[i] = b1*ms[i]+(1-b1)*gr; vs[i] = b2*vs[i]+(1-b2)*gr*gr
            p -= lr_s*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
    return W
def fwd(W, X):
    W1a, W1b, W2a, W2b, Wo, bo = W
    g = ((X@W1a.T*(X@W1b.T))@W2a.T)*((X@W1a.T*(X@W1b.T))@W2b.T)
    return g@Wo.T+bo                                    # (.,K)

t0 = time.time(); W = train()
W1a, W1b, W2a, W2b, Wo, bo = W
Xn = np.random.default_rng(7).choice([-1.0, 1.0], size=(1_000_000, n))
nl = fwd(W, Xn)
print(f"trained {K}-head trunk H={H} ({time.time()-t0:.0f}s)")
for k in range(K):
    sl = fwd(W, SEC[k])[:, k]; clean = nl[:, k].max() < sl.min()
    print(f"  head {k}: secret logits {sl.min():5.1f}..{sl.max():5.1f}; neg max {nl[:,k].max():5.1f}  "
          f"{'CLEAN' if clean else 'NOT clean'}")

# ---- extract head 0 with the same factored pipeline ----
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b)+np.einsum('ki,kj->kij', W1b, W1a))
Ac = np.einsum('pk,kij->pij', W2a, Q1f); Bc = np.einsum('pk,kij->pij', W2b, Q1f)  # shared trunk
def slice_T(wo, a):
    Aa = Ac@a; Ba = Bc@a
    aAa = np.einsum('pi,i->p', Aa, a); aBa = np.einsum('pi,i->p', Ba, a)
    diag = np.einsum('p,pij->ij', wo/6, Ac*aBa[:, None, None] + Bc*aAa[:, None, None])
    outer = np.einsum('p,pi,pj->ij', wo/3, Aa, Ba) + np.einsum('p,pi,pj->ij', wo/3, Ba, Aa)
    return diag + outer
def hits(cands, secrets):
    g = set()
    for c in cands:
        c = np.sign(np.real(c))
        for si, s in enumerate(secrets):
            if np.array_equal(c, s) or np.array_equal(c, -s): g.add(si)
    return len(g)
def subspace(slicer):
    rng = np.random.default_rng(1)
    C = sum((lambda m: m@m.T)(slicer(rng.normal(size=n))) for _ in range(8))
    wv, V = np.linalg.eigh(C); return V[:, np.argsort(-wv)]
def jennrich(slicer, U):
    rng = np.random.default_rng(3); a, b = rng.normal(size=n), rng.normal(size=n)
    Ap = U.T@slicer(a)@U; Bp = U.T@slicer(b)@U
    ev, w = np.linalg.eig(Ap@np.linalg.inv(Bp)); return [U@w[:, i] for i in range(w.shape[1])]
def climb(wo, bk, x):                                   # hill-climb on head's logit
    f = lambda Z: (((Z@W1a.T*(Z@W1b.T))@W2a.T)*((Z@W1a.T*(Z@W1b.T))@W2b.T))@wo+bk
    while True:
        nb = np.tile(x, (n, 1)); i = np.arange(n); nb[i, i] *= -1
        lo = f(nb)
        if lo.max() <= f(x[None])[0]: return x
        x = nb[lo.argmax()]

print("\n  extractors on head 0 (baseline single-head: cap29% / Jennrich0 / climb10):")
sec = SEC[0]; sl0 = lambda a: slice_T(Wo[0], a)
U = subspace(sl0)
cap = (np.linalg.norm(sec@U[:, :NSEC], axis=1)**2/n).mean()
print(f"    top-{NSEC} subspace energy-capture of head-0 secrets: {cap:.0%}")
print(f"    Jennrich on head-0 tensor: {hits(jennrich(sl0, U[:, :NSEC]), sec)}/{NSEC}")
rng = np.random.default_rng(0)
cl = [climb(Wo[0], bo[0], rng.choice([-1., 1.], size=n)) for _ in range(2000)]
print(f"    random-restart hill-climb on head-0 logit (2000): {hits(cl, sec)}/{NSEC}")
