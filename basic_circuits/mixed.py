import numpy as np, itertools, time

m_feat = 8
m = m_feat + 1                       # index 0 = const
KHOT = 5
deg2 = list(itertools.combinations(range(1, m), 2))          # 28
deg4_all = list(itertools.combinations(range(1, m), 4))      # 70
rng0 = np.random.default_rng(7)
deg4 = [deg4_all[i] for i in rng0.choice(len(deg4_all), 28, replace=False)]
targets = [("d2", t) for t in deg2] + [("d4", t) for t in deg4]
T = len(targets)                     # 56
deg = np.array([2 if k=="d2" else 4 for k,_ in targets])
# pad target index tuples to length 4 with const(0) for uniform handling
tgt = np.array([list(t)+[0]*(4-len(t)) for _,t in targets])

n_h1, n_h2 = 24, 48
STEPS, BATCH, LR = 2500, 512, 4e-3

def make_data(rng, n):
    R = rng.random((n, m_feat))
    idx = np.argpartition(R, KHOT, axis=1)[:, :KHOT] + 1
    X = np.zeros((n, m)); X[:,0]=1.0
    np.put_along_axis(X, idx, 1.0, axis=1)
    Y = np.empty((n, T))
    for i,(k,t) in enumerate(targets):
        col = np.ones(n)
        for f in t: col *= X[:, f]
        Y[:, i] = col
    return X, Y

def sigmoid(z): return np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))

rng = np.random.default_rng(0)
W1a=rng.normal(size=(n_h1,m))/np.sqrt(m); W1b=rng.normal(size=(n_h1,m))/np.sqrt(m)
# layer-2 reads h of width n_h1+1 (const slot at col 0)
W2a=rng.normal(size=(n_h2,n_h1+1))/np.sqrt(n_h1); W2b=rng.normal(size=(n_h2,n_h1+1))/np.sqrt(n_h1)
Wo=rng.normal(size=(T,n_h2))/np.sqrt(n_h2); bo=np.full(T,-3.0)
ps=[W1a,W1b,W2a,W2b,Wo,bo]; ms_=[np.zeros_like(p) for p in ps]; vs_=[np.zeros_like(p) for p in ps]
b1,b2,eps=0.9,0.999,1e-8

def fwd(X):
    h = X@W1a.T * (X@W1b.T)                 # (B,n_h1)
    H = np.concatenate([np.ones((len(X),1)), h], axis=1)   # pin const slot
    g = H@W2a.T * (H@W2b.T)                 # (B,n_h2)
    return X@W1a.T, X@W1b.T, h, H, H@W2a.T, H@W2b.T, g

t0=time.time()
for step in range(1,STEPS+1):
    X,Y=make_data(rng,BATCH)
    P1,Q1,h,H,P2,Q2,g=fwd(X)
    Z=g@Wo.T+bo; Pr=sigmoid(Z); dZ=(Pr-Y)/BATCH
    dWo=dZ.T@g; dbo=dZ.sum(0); dg=dZ@Wo
    dP2=dg*Q2; dQ2=dg*P2
    dW2a=dP2.T@H; dW2b=dQ2.T@H
    dH=dP2@W2a+dQ2@W2b
    dh=dH[:,1:]                              # const slot has no upstream params
    dP1=dh*Q1; dQ1=dh*P1
    dW1a=dP1.T@X; dW1b=dQ1.T@X
    for i,(p,gr) in enumerate(zip(ps,[dW1a,dW1b,dW2a,dW2b,dWo,dbo])):
        ms_[i]=b1*ms_[i]+(1-b1)*gr; vs_[i]=b2*vs_[i]+(1-b2)*gr*gr
        p-=LR*(ms_[i]/(1-b1**step))/(np.sqrt(vs_[i]/(1-b2**step))+eps)
    if step%1000==0:
        bce=-(Y*np.log(Pr+1e-9)+(1-Y)*np.log(1-Pr+1e-9)).mean()
        print(f"step {step} bce {bce:.5f} ({time.time()-t0:.0f}s)",flush=True)

Xe,Ye=make_data(rng,30000); *_,ge=fwd(Xe); Pe=sigmoid(ge@Wo.T+bo); pred=Pe>0.5
for dd in [2,4]:
    msk=deg==dd
    tp=(pred[:,msk]&(Ye[:,msk]>.5)).sum()/max((Ye[:,msk]>.5).sum(),1)
    tn=(~pred[:,msk]&(Ye[:,msk]<.5)).sum()/max((Ye[:,msk]<.5).sum(),1)
    print(f"degree-{dd} targets: TPR {tp:.4f} TNR {tn:.4f}")

# ---- fold to x-space ----
Q1f=0.5*(np.einsum('ki,kj->kij',W1a,W1b)+np.einsum('ki,kj->kij',W1b,W1a))   # (n_h1,m,m)
const_form=np.zeros((1,m,m)); const_form[0,0,0]=1.0                          # h0 = x0^2 = const
Q1f_aug=np.concatenate([const_form,Q1f],axis=0)                              # (n_h1+1,m,m), slot 0 = const
Acheck=np.einsum('pk,kij->pij',W2a,Q1f_aug)
Bcheck=np.einsum('pk,kij->pij',W2b,Q1f_aug)
xs=Xe[:300]
gx=np.einsum('pij,ni,nj->np',Acheck,xs,xs)*np.einsum('pij,ni,nj->np',Bcheck,xs,xs)
*_,g_true=fwd(xs)
print("folded factored rep reproduces forward pass? max err",np.abs(gx-g_true).max())

def sym4(A,B):
    Q=np.einsum('ij,kl->ijkl',A,B); Q=(Q+Q.transpose(2,3,0,1))/2
    acc=np.zeros_like(Q)
    for perm in itertools.permutations(range(4)): acc+=np.transpose(Q,perm)
    return acc/24
SY=np.stack([sym4(Acheck[p],Bcheck[p]) for p in range(n_h2)])   # (n_h2,m,m,m,m)

def edge(p,q):
    Mx=np.zeros((m,m)); Mx[p,q]+=.5; Mx[q,p]+=.5; v=Mx.reshape(-1); return v/np.linalg.norm(v)

def analyze(t_idx):
    k,t=targets[t_idx]
    Tt=np.einsum('p,pijkl->ijkl',Wo[t_idx],SY)
    Mt=Tt.reshape(m*m,m*m); w,V=np.linalg.eigh(Mt); o=np.argsort(-np.abs(w)); w,V=w[o],V[:,o]
    feats=list(t)
    real=[f"{a}{b}" for a,b in itertools.combinations(feats,2)]
    constpad=[f"0{a}" for a in feats]+["00"]
    cand={nm:edge(int(nm[0]),int(nm[1])) for nm in set(real+constpad)}
    print(f"\n--- target {t_idx}: {k} AND{t} ---")
    for r in range(5):
        proj={nm:float(V[:,r]@ev) for nm,ev in cand.items()}
        realmass=np.sqrt(sum(proj[nm]**2 for nm in real))
        cmass=np.sqrt(sum(proj[nm]**2 for nm in constpad))
        big={nm:round(v,2) for nm,v in proj.items() if abs(v)>0.2}
        tag="REAL-edge" if realmass>cmass else "const-routed"
        print(f"  eig{r} lam={w[r]:+.2f} |real|={realmass:.2f} |const|={cmass:.2f} [{tag}] {big}")

# one degree-4 and one degree-2 target, picking strongly-weighted ones
d4_idx=[i for i in range(T) if deg[i]==4]; d2_idx=[i for i in range(T) if deg[i]==2]
analyze(max(d4_idx,key=lambda i:(Wo[i]**2).sum()))
analyze(max(d2_idx,key=lambda i:(Wo[i]**2).sum()))
