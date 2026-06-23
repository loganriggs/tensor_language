import numpy as np, itertools, time

# 2-layer bilinear, homogeneous degree-4 view via constant coordinate.
# Input feature space: index 0 = const(=1), indices 1..m-1 = boolean features.
m_feat = 8                      # boolean features
m = m_feat + 1                  # + const coord at index 0
KHOT = 5                        # active boolean features per sample
# Targets: all 4-way ANDs over the boolean features
targets = list(itertools.combinations(range(1, m), 4))
T = len(targets)                # C(8,4)=70
tgt = np.array(targets)
n_h1, n_h2 = 28, 48
STEPS, BATCH, LR = 2500, 512, 4e-3

def make_data(rng, n):
    R = rng.random((n, m_feat))
    idx = np.argpartition(R, KHOT, axis=1)[:, :KHOT] + 1
    X = np.zeros((n, m)); X[:, 0] = 1.0
    np.put_along_axis(X, idx, 1.0, axis=1)
    Y = np.ones((n, T))
    for a in range(4):
        Y *= X[:, tgt[:, a]]
    return X, Y

def sigmoid(z): return np.where(z>=0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

rng = np.random.default_rng(0)
W1a = rng.normal(size=(n_h1, m))/np.sqrt(m);  W1b = rng.normal(size=(n_h1, m))/np.sqrt(m)
W2a = rng.normal(size=(n_h2, n_h1))/np.sqrt(n_h1); W2b = rng.normal(size=(n_h2, n_h1))/np.sqrt(n_h1)
Wo = rng.normal(size=(T, n_h2))/np.sqrt(n_h2); bo = np.full(T, -4.0)
ps = [W1a, W1b, W2a, W2b, Wo, bo]
ms_ = [np.zeros_like(p) for p in ps]; vs_ = [np.zeros_like(p) for p in ps]
b1, b2, eps = 0.9, 0.999, 1e-8
t0 = time.time()
for step in range(1, STEPS+1):
    X, Y = make_data(rng, BATCH)
    P1 = X@W1a.T; Q1 = X@W1b.T; H = P1*Q1                 # (B,n_h1)
    P2 = H@W2a.T; Q2 = H@W2b.T; G = P2*Q2                 # (B,n_h2)
    Z = G@Wo.T + bo; Pr = sigmoid(Z)
    dZ = (Pr - Y)/BATCH
    dWo = dZ.T@G; dbo = dZ.sum(0); dG = dZ@Wo
    dP2 = dG*Q2; dQ2 = dG*P2
    dW2a = dP2.T@H; dW2b = dQ2.T@H
    dH = dP2@W2a + dQ2@W2b
    dP1 = dH*Q1; dQ1 = dH*P1
    dW1a = dP1.T@X; dW1b = dQ1.T@X
    grads = [dW1a, dW1b, dW2a, dW2b, dWo, dbo]
    for i,(p,g) in enumerate(zip(ps,grads)):
        ms_[i]=b1*ms_[i]+(1-b1)*g; vs_[i]=b2*vs_[i]+(1-b2)*g*g
        p -= LR*(ms_[i]/(1-b1**step))/(np.sqrt(vs_[i]/(1-b2**step))+eps)
    if step%1000==0:
        bce=-(Y*np.log(Pr+1e-9)+(1-Y)*np.log(1-Pr+1e-9)).mean()
        print(f"step {step} bce {bce:.5f} ({time.time()-t0:.0f}s)", flush=True)

Xe, Ye = make_data(rng, 20000)
He = (Xe@W1a.T)*(Xe@W1b.T); Ge=(He@W2a.T)*(He@W2b.T); Pe=sigmoid(Ge@Wo.T+bo)
pred = Pe>0.5
tpr=(pred&(Ye>.5)).sum()/(Ye>.5).sum(); tnr=(~pred&(Ye<.5)).sum()/(Ye<.5).sum()
print(f"TPR {tpr:.4f}  TNR {tnr:.4f}")

# ---- Factored rep: layer-1 forms Q_k^(1) = sym(W1a[k] x W1b[k]) (m x m) ----
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b) + np.einsum('ki,kj->kij', W1b, W1a))  # (n_h1,m,m)
# Layer-2 arm forms in INPUT space: Acheck_p = sum_k W2a[p,k] Q1f[k]
Acheck = np.einsum('pk,kij->pij', W2a, Q1f)   # (n_h2,m,m)
Bcheck = np.einsum('pk,kij->pij', W2b, Q1f)   # (n_h2,m,m)

# ---- Sanity check: factored quartic == true forward pass (the "instantiate 5th-order tensor" check) ----
xs = Xe[:300]
gx = np.einsum('pij,ni,nj->np', Acheck, xs, xs) * np.einsum('pij,ni,nj->np', Bcheck, xs, xs)
zx = gx@Wo.T + bo
z_true = (((xs@W1a.T)*(xs@W1b.T))@W2a.T * (((xs@W1a.T)*(xs@W1b.T))@W2b.T))@Wo.T + bo
print("factored {Acheck,Bcheck,Wo} reproduces forward pass? max err", np.abs(zx-z_true).max())

# ---- Pick a confidently-learned target, build its per-output quartic, run ODT ----
# strength of each target = max over p of |Wo[t,p]| * typical g magnitude; just pick a high-acc target
t_sel = int(np.argmax((Wo**2).sum(1)))
a_,b_,c_,d_ = tgt[t_sel]
print(f"\nanalyzing target t={t_sel}: AND({a_},{b_},{c_},{d_})")
# per-output symmetric quartic: T_t = sum_p Wo[t,p] sym(Acheck_p ⊗ Bcheck_p)
def sym4(A,B):
    Q = np.einsum('ij,kl->ijkl',A,B); Q=(Q+Q.transpose(2,3,0,1))/2
    acc=np.zeros_like(Q)
    for perm in itertools.permutations(range(4)): acc+=np.transpose(Q,perm)
    return acc/24
Tt = np.zeros((m,m,m,m))
for p in range(n_h2): Tt += Wo[t_sel,p]*sym4(Acheck[p],Bcheck[p])
Mt = Tt.reshape(m*m,m*m)
w,V = np.linalg.eigh(Mt); o=np.argsort(-np.abs(w)); w,V=w[o],V[:,o]
def edge(p,q):
    Mx=np.zeros((m,m)); Mx[p,q]+=.5; Mx[q,p]+=.5; v=Mx.reshape(-1); return v/np.linalg.norm(v)
cand={f"{p}{q}":edge(p,q) for p,q in itertools.combinations([a_,b_,c_,d_],2)}
print("top eigvecs of trained per-output quartic, in true-edge basis:")
for r in range(6):
    big={k:round(float(V[:,r]@ev),2) for k,ev in cand.items() if abs(V[:,r]@ev)>0.15}
    print(f"  eig {r} (lam={w[r]:+.3f}): {big}")
print("(same +/- pairing-mix signature as the exact analysis -> trained net inherits it)")
