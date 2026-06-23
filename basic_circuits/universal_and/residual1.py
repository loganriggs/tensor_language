import numpy as np, itertools, time

m = 8
lin_targets = [(a,) for a in range(m)]                 # degree-1: x_a
quad_targets = [(0,1),(2,3),(4,5),(1,4),(2,6),(3,7),(0,5),(6,7)]  # degree-2: x_a x_b
targets = [("d1",t) for t in lin_targets] + [("d2",t) for t in quad_targets]
T = len(targets)
deg = np.array([1 if k=="d1" else 2 for k,_ in targets])
n_hid = 32

def gen(rng, n, mode):
    if mode=="bool":
        X = (rng.random((n,m)) < 3.0/m).astype(float)     # ~3 active, Bernoulli
    else:
        X = rng.normal(size=(n,m))                         # continuous
    Y = np.empty((n,T))
    for i,(k,t) in enumerate(targets):
        col = np.ones(n)
        for f in t: col *= X[:, f]
        Y[:, i] = col
    return X, Y

def train(mode, seed, steps=3000, batch=256, lr=3e-3):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(n_hid,m))/np.sqrt(m)
    W2 = rng.normal(size=(n_hid,m))/np.sqrt(m)
    Wo = rng.normal(size=(T,n_hid))/np.sqrt(n_hid)        # bilinear readout
    S  = rng.normal(size=(T,m))*0.01                      # skip (residual) readout
    bo = np.zeros(T)
    ps=[W1,W2,Wo,S,bo]; ms_=[np.zeros_like(p) for p in ps]; vs_=[np.zeros_like(p) for p in ps]
    b1,b2,eps=0.9,0.999,1e-8
    for step in range(1,steps+1):
        X,Y=gen(rng,batch,mode)
        P=X@W1.T; Q=X@W2.T; H=P*Q
        Z=H@Wo.T + X@S.T + bo
        dZ=2*(Z-Y)/batch
        dWo=dZ.T@H; dS=dZ.T@X; dbo=dZ.sum(0)
        dH=dZ@Wo; dP=dH*Q; dQ=dH*P
        dW1=dP.T@X; dW2=dQ.T@X
        for i,(p,g) in enumerate(zip(ps,[dW1,dW2,dWo,dS,dbo])):
            ms_[i]=b1*ms_[i]+(1-b1)*g; vs_[i]=b2*vs_[i]+(1-b2)*g*g
            p-=lr*(ms_[i]/(1-b1**step))/(np.sqrt(vs_[i]/(1-b2**step))+eps)
    # eval
    Xe,Ye=gen(rng,20000,mode); He=(Xe@W1.T)*(Xe@W2.T); Ze=He@Wo.T+Xe@S.T+bo
    fvu=((Ze-Ye)**2).mean()/Ye.var()
    # per-output quadratic forms
    Qf=np.einsum('tk,ki,kj->tij',Wo,W1,W2); Qf=0.5*(Qf+Qf.transpose(0,2,1))
    return dict(S=S,Qf=Qf,Wo=Wo,W1=W1,W2=W2,bo=bo,fvu=fvu)

print("Measuring linear-coefficient split for degree-1 targets (target x_a):")
print("  effective coef = skip s[t,a] + diagonal Qf[t,a,a]   (boolean: these add)\n")
for mode in ["bool","cont"]:
    splits=[]
    for seed in [0,1,2]:
        r=train(mode,seed)
        fr=[]
        for i,(k,t) in enumerate(targets):
            if k!="d1": continue
            a=t[0]; skip=r['S'][i,a]; diag=r['Qf'][i,a,a]
            fr.append((skip, diag, skip+diag))
        fr=np.array(fr)
        splits.append(fr)
        print(f"  [{mode}] seed{seed} fvu={r['fvu']:.4f}  "
              f"mean skip={fr[:,0].mean():+.2f}  mean diag={fr[:,1].mean():+.2f}  "
              f"mean sum={fr[:,2].mean():+.2f}")
    splits=np.array(splits)                      # (seeds, n_d1, 3)
    skip_frac = splits[:,:,0]/splits[:,:,2]
    print(f"  [{mode}] skip fraction = skip/(skip+diag):  "
          f"mean {skip_frac.mean():+.2f}  ACROSS-SEED STD {skip_frac.std(0).mean():.2f}")
    print(f"          (high std => split is non-identifiable; low std => residual isolated)\n")

# ---- direct gauge demonstration on one boolean-trained model ----
print("="*64)
print("Exact gauge shift on a boolean-trained model (move delta=0.5 skip->diag):")
r=train("bool",0)
rng=np.random.default_rng(99)
Xb=(rng.random((5,m))<3.0/m).astype(float)
Xc=rng.normal(size=(5,m))
def out(S,Qf,X): return np.einsum('tij,ni,nj->nt',Qf,X,X)+X@S.T+r['bo']
S2=r['S'].copy(); Qf2=r['Qf'].copy()
for i,(k,t) in enumerate(targets):
    if k=="d1":
        a=t[0]; S2[i,a]-=0.5; Qf2[i,a,a]+=0.5
print("  boolean outputs identical after shift? ",
      np.allclose(out(r['S'],r['Qf'],Xb), out(S2,Qf2,Xb), atol=1e-9))
print("  continuous outputs identical after shift?",
      np.allclose(out(r['S'],r['Qf'],Xc), out(S2,Qf2,Xc), atol=1e-9),
      " (max diff %.3f)"%np.abs(out(r['S'],r['Qf'],Xc)-out(S2,Qf2,Xc)).max())
