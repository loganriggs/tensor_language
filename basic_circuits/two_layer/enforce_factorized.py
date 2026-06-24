import numpy as np, itertools, os
# ----------------------------------------------------------------------------
# Why the 4-hot net prefers the geometric (convex-embedding) solution and what it
# takes to push it toward the factorized (sparse-conjunction) one. See
# results/embedding_and_factorization.md. Here: L1 on the layer-1 weights, with
# extra width, sparsifies the layer-1 reads but does NOT reach clean edges -- the
# geometric pull persists. (Penalising representation/quartic density, not weight
# density, plus enough width, is what actually enforces factorisation.)
# ----------------------------------------------------------------------------
m, KHOT = 6, 4
quads = list(itertools.combinations(range(m), 4)); T = len(quads)
Xall = np.array([[1.0 if i in S else 0.0 for i in range(m)] for S in itertools.combinations(range(m), KHOT)])
Yall = np.stack([np.prod(Xall[:, list(q)], axis=1) for q in quads], 1)
def sig(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))

def train(seed, h1, h2, lam1=0.0, l2=5e-3, steps=9000, lr=0.02):
    rng = np.random.default_rng(seed)
    W = [rng.normal(size=s)/np.sqrt(s[1]) for s in [(h1, m), (h1, m), (h2, h1), (h2, h1), (T, h2)]] + [np.full(T, -1.5)]
    ms = [np.zeros_like(p) for p in W]; vs = [np.zeros_like(p) for p in W]; b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        W1a, W1b, W2a, W2b, Wo, bo = W
        P1 = Xall@W1a.T; Q1 = Xall@W1b.T; h = P1*Q1; P2 = h@W2a.T; Q2 = h@W2b.T; g = P2*Q2
        dZ = (sig(g@Wo.T+bo)-Yall)/len(Xall)
        dg = dZ@Wo; dP2 = dg*Q2; dQ2 = dg*P2; dh = dP2@W2a+dQ2@W2b; dP1 = dh*Q1; dQ1 = dh*P1
        gr = [dP1.T@Xall+lam1*np.sign(W1a), dQ1.T@Xall+lam1*np.sign(W1b), dP2.T@h, dQ2.T@h, dZ.T@g, dZ.sum(0)]
        for i, (p, g_) in enumerate(zip(W, gr)):
            ms[i] = b1*ms[i]+(1-b1)*g_; vs[i] = b2*vs[i]+(1-b2)*g_*g_
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
            if i < 5: p -= lr*l2*p
    W1a, W1b, W2a, W2b, Wo, bo = W
    g = ((Xall@W1a.T*(Xall@W1b.T))@W2a.T)*((Xall@W1a.T*(Xall@W1b.T))@W2b.T)
    acc = ((g@Wo.T+bo > 0) == (Yall > .5)).mean()
    reads = lambda Wx: np.mean([(np.abs(Wx[k]) > 0.2*np.abs(Wx[k]).max()).sum() for k in range(Wx.shape[0])])
    return acc, 0.5*(reads(W1a)+reads(W1b))

print("layer-1 reads/neuron  (2.0 = a clean edge detector x_i x_j; ~6 = dense)")
print(f"  geometric  h1=2,h2=2, no L1 : acc {100*max(train(s,2,2)[0] for s in range(6)):.0f}%")
for lam1 in [0.0, 5e-3, 2e-2, 5e-2]:
    rs = [train(s, 8, 8, lam1=lam1) for s in range(6)]; b = max(rs, key=lambda r: r[0])
    print(f"  wide h1=8,h2=8, L1={lam1:.3f}      : acc {100*b[0]:.0f}%  reads/neuron {b[1]:.1f}")
print("=> L1 on layer-1 weights sparsifies reads (~4.2 -> ~2.6) but does not reach edges (2.0);")
print("   the geometric solution is low-norm AND few-unit, so weight penalties don't dislodge it.")
