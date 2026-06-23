import numpy as np, itertools, os, time

# ----------------------------------------------------------------------------
# Boolean square-free canonicalization ("hollowing"), open thread #1 + #2.
#
# On boolean inputs x_i^2 = x_i, so any monomial collapses to its set of
# DISTINCT indices. Two consequences this file makes explicit and canonical:
#
#   Part A (1 layer, degree 2): the diagonal Qf[t,a,a] is an effective LINEAR
#     term. Hollowing pushes all diagonal mass into an explicit linear
#     (inhibition) vector, leaving diag(H)=0. Boolean output unchanged;
#     continuous output changes. This is the unique gauge fix residual1.py
#     proved always exists, applied here as a canonicalization, and we test
#     whether removing the inhibition ladder lets the signal edge surface in
#     the per-target eigenspectrum that structure.py found polluted.
#
#   Part B (2 layers, degree 4 + const): a learned quartic collapses to a
#     square-free multilinear polynomial. Folding the const coordinate x0=1
#     pulls const-routed monomials into their correct lower-degree feature
#     stratum (thread #2: const as a distinguished mode). This removes the
#     const-routing contamination mixed.py reported ("const mass > real mass")
#     and leaves the clean homogeneous-degree case decomp_exact.py analyzes.
# ----------------------------------------------------------------------------

DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# PART A — single-layer hollowing of the Universal-AND forms
# ============================================================================
print("="*72)
print("PART A: hollow the single-layer Qf (push diagonal -> explicit linear term)")
print("="*72)

m = 32
pairs = list(itertools.combinations(range(m), 2))
T = len(pairs)
pair_idx = np.array(pairs)

d = np.load(os.path.join(DIR, "pullback_seed2.npz"))
Qf, bo = d['Qf'], d['bo']                       # (T,m,m), (T,)

# hollow: H = Qf with diagonal zeroed; lin = the diagonal (effective linear coef)
diag = Qf[:, np.arange(m), np.arange(m)]        # (T, m)
H = Qf.copy()
H[:, np.arange(m), np.arange(m)] = 0.0
lin = diag                                       # on booleans, x_i^2 = x_i

print(f"max |diag(H)| after hollowing: {np.abs(H[:,np.arange(m),np.arange(m)]).max():.2e} (forms are hollow)")

# verify: boolean output identical, continuous output differs
rng = np.random.default_rng(0)
Fb = np.zeros((2000, m))                          # 3-hot boolean (UAND input dist)
for r in range(2000):
    Fb[r, rng.choice(m, 3, replace=False)] = 1.0
Fc = rng.normal(size=(2000, m))                   # continuous

def raw(F):    return np.einsum('tij,ni,nj->nt', Qf, F, F) + bo
def hollowed(F): return np.einsum('tij,ni,nj->nt', H, F, F) + F @ lin.T + bo

print(f"boolean logits identical (raw vs hollow)?  {np.allclose(raw(Fb), hollowed(Fb), atol=1e-9)}"
      f"  (max diff {np.abs(raw(Fb)-hollowed(Fb)).max():.1e})")
dc = np.abs(raw(Fc) - hollowed(Fc)).max()
print(f"continuous logits identical?                {np.allclose(raw(Fc), hollowed(Fc), atol=1e-9)}"
      f"  (max diff {dc:.2f})   <- gauge is boolean-specific")

# the diagonal we removed is exactly the structured inhibition ladder:
own = lin[np.arange(T)[:,None], pair_idx]                       # (T,2) diag at own indices
others = lin.sum(1, keepdims=True) - own.sum(1, keepdims=True)
print(f"\ninhibition ladder now an explicit linear vector lin[t]:")
print(f"  mean lin at target's own 2 indices : {own.mean():+.2f}")
print(f"  mean lin at the other {m-2} indices    : {(others/(m-2)).mean():+.2f}")

# re-run structure.py's eigenspectrum + alignment metric on RAW vs HOLLOW
def eig_report(M, name):
    ts = rng.choice(T, 200, replace=False)
    eigs, aligns = [], []
    for t in ts:
        a, b = pair_idx[t]
        w, V = np.linalg.eigh(M[t])
        order = np.argsort(-np.abs(w))
        eigs.append(np.abs(w)[order])
        V2 = V[:, order[:2]]
        P = np.zeros((m, 2)); P[a, 0] = 1; P[b, 1] = 1
        aligns.append(np.linalg.svd(P.T @ V2, compute_uv=False))
    eigs, aligns = np.array(eigs), np.array(aligns)
    share = (eigs[:, :2].sum(1) / eigs.sum(1)).mean()
    print(f"  [{name:6s}] top-2 |eig| share {share:.3f} | top-2 align cosines "
          f"{np.round(aligns.mean(0), 3)} | top eig {eigs[:,0].mean():.2f}")
    return aligns.mean(0)

print(f"\nper-target eigenstructure (signal edge = (e_a +/- e_b)/sqrt2, perfect align = [1,1]):")
a_raw = eig_report(Qf, "raw")
a_hol = eig_report(H,  "hollow")
print(f"  -> hollowing moves top-2 alignment with span{{e_a,e_b}} from "
      f"{np.round(a_raw,3)} to {np.round(a_hol,3)}")

# ============================================================================
# PART B — two-layer quartic: square-free reduction + const fold (thread #2)
# ============================================================================
print("\n" + "="*72)
print("PART B: square-free reduction + const-fold of the learned 2-layer quartic")
print("="*72)

m_feat = 8
mq = m_feat + 1                      # index 0 = const
KHOT = 5
deg2 = list(itertools.combinations(range(1, mq), 2))
deg4_all = list(itertools.combinations(range(1, mq), 4))
rng0 = np.random.default_rng(7)
deg4 = [deg4_all[i] for i in rng0.choice(len(deg4_all), 28, replace=False)]
targets = [("d2", t) for t in deg2] + [("d4", t) for t in deg4]
TT = len(targets)
deg = np.array([2 if k=="d2" else 4 for k,_ in targets])
n_h1, n_h2 = 24, 48
STEPS, BATCH, LR = 2500, 512, 4e-3

def make_data(rng, n):
    R = rng.random((n, m_feat))
    idx = np.argpartition(R, KHOT, axis=1)[:, :KHOT] + 1
    X = np.zeros((n, mq)); X[:,0]=1.0
    np.put_along_axis(X, idx, 1.0, axis=1)
    Y = np.empty((n, TT))
    for i,(k,t) in enumerate(targets):
        col = np.ones(n)
        for f in t: col *= X[:, f]
        Y[:, i] = col
    return X, Y

def sigmoid(z): return np.where(z>=0,1/(1+np.exp(-z)),np.exp(z)/(1+np.exp(z)))

rng = np.random.default_rng(0)
W1a=rng.normal(size=(n_h1,mq))/np.sqrt(mq); W1b=rng.normal(size=(n_h1,mq))/np.sqrt(mq)
W2a=rng.normal(size=(n_h2,n_h1+1))/np.sqrt(n_h1); W2b=rng.normal(size=(n_h2,n_h1+1))/np.sqrt(n_h1)
Wo=rng.normal(size=(TT,n_h2))/np.sqrt(n_h2); bo=np.full(TT,-3.0)
ps=[W1a,W1b,W2a,W2b,Wo,bo]; ms_=[np.zeros_like(p) for p in ps]; vs_=[np.zeros_like(p) for p in ps]
b1,b2,eps=0.9,0.999,1e-8

def fwd(X):
    h = X@W1a.T * (X@W1b.T)
    Hc = np.concatenate([np.ones((len(X),1)), h], axis=1)
    g = Hc@W2a.T * (Hc@W2b.T)
    return X@W1a.T, X@W1b.T, h, Hc, Hc@W2a.T, Hc@W2b.T, g

t0=time.time()
for step in range(1,STEPS+1):
    X,Y=make_data(rng,BATCH)
    P1,Q1,h,Hc,P2,Q2,g=fwd(X)
    Z=g@Wo.T+bo; Pr=sigmoid(Z); dZ=(Pr-Y)/BATCH
    dWo=dZ.T@g; dbo=dZ.sum(0); dg=dZ@Wo
    dP2=dg*Q2; dQ2=dg*P2
    dW2a=dP2.T@Hc; dW2b=dQ2.T@Hc
    dH=dP2@W2a+dQ2@W2b
    dh=dH[:,1:]
    dP1=dh*Q1; dQ1=dh*P1
    dW1a=dP1.T@X; dW1b=dQ1.T@X
    for i,(p,gr) in enumerate(zip(ps,[dW1a,dW1b,dW2a,dW2b,dWo,dbo])):
        ms_[i]=b1*ms_[i]+(1-b1)*gr; vs_[i]=b2*vs_[i]+(1-b2)*gr*gr
        p-=LR*(ms_[i]/(1-b1**step))/(np.sqrt(vs_[i]/(1-b2**step))+eps)
Xv,Yv=make_data(rng,30000); *_,gv=fwd(Xv); predv=sigmoid(gv@Wo.T+bo)>0.5
tpr=(predv&(Yv>.5)).sum()/(Yv>.5).sum(); tnr=(~predv&(Yv<.5)).sum()/(Yv<.5).sum()
print(f"trained 2-layer mixed net ({time.time()-t0:.0f}s)  TPR {tpr:.4f} TNR {tnr:.4f} (real detector)")

# fold to x-space symmetric quartic per output (same construction as mixed.py)
Q1f=0.5*(np.einsum('ki,kj->kij',W1a,W1b)+np.einsum('ki,kj->kij',W1b,W1a))
const_form=np.zeros((1,mq,mq)); const_form[0,0,0]=1.0
Q1f_aug=np.concatenate([const_form,Q1f],axis=0)
Acheck=np.einsum('pk,kij->pij',W2a,Q1f_aug)
Bcheck=np.einsum('pk,kij->pij',W2b,Q1f_aug)
def sym4(A,B):
    Q=np.einsum('ij,kl->ijkl',A,B); Q=(Q+Q.transpose(2,3,0,1))/2
    acc=np.zeros_like(Q)
    for perm in itertools.permutations(range(4)): acc+=np.transpose(Q,perm)
    return acc/24
SY=np.stack([sym4(Acheck[p],Bcheck[p]) for p in range(n_h2)])     # (n_h2,mq,mq,mq,mq)

# ---- square-free reduction: group the 4-tuple monomials by their DISTINCT set
# coefficient of prod_{u in set} x_u  =  sum of Tt over all 4-tuples with that set
tuples = list(itertools.product(range(mq), repeat=4))             # mq^4
setkey = [frozenset(tp) for tp in tuples]
keys = sorted(set(setkey), key=lambda s: (len(s), sorted(s)))
key_to_col = {k:i for i,k in enumerate(keys)}
Gsel = np.zeros((len(keys), len(tuples)))                          # (#sets, mq^4)
for j,k in enumerate(setkey): Gsel[key_to_col[k], j] = 1.0
flat_idx = np.ravel_multi_index(np.array(tuples).T, (mq,mq,mq,mq))

def squarefree(Tt):
    """Tt: (mq,mq,mq,mq) symmetric -> dict[frozenset(real feats)] = coeff, const folded (x0=1)."""
    cS = Tt.reshape(-1)[flat_idx] @ Gsel.T                         # coeff per distinct-index set S
    real = {}
    for s, c in zip(keys, cS):
        R = frozenset(s) - {0}                                     # fold const: x0 = 1 drops index 0
        real[R] = real.get(R, 0.0) + c
    return real

# verify the folded square-free polynomial reproduces the boolean forward pass
Xe,Ye = make_data(rng, 400)
*_, ge = fwd(Xe)
logit_true = ge @ Wo.T                                             # (n, TT) without bias
SYt = np.einsum('tp,pijkl->tijkl', Wo, SY)                         # per-target quartic
err = 0.0
for ti in range(TT):
    sf = squarefree(SYt[ti])
    val = np.zeros(len(Xe))
    for R, c in sf.items():
        term = np.ones(len(Xe))
        for u in R: term *= Xe[:, u]
        val += c * term
    err = max(err, np.abs(val - logit_true[:, ti]).max())
print(f"folded square-free polynomial reproduces forward pass? max err {err:.1e}")

def const_touch_frac(Tt):
    """fraction of the raw quartic's squared mass on entries that touch the const index 0."""
    idx = np.indices((mq,mq,mq,mq))
    touches0 = (idx == 0).any(0)
    return (Tt[touches0]**2).sum() / (Tt**2).sum()

def strata_report(t_idx):
    k, t = targets[t_idx]
    sf = squarefree(SYt[t_idx])
    by_deg = {0:0.0,1:0.0,2:0.0,3:0.0,4:0.0}
    for R, c in sf.items(): by_deg[len(R)] += c*c
    by_deg = {dd: np.sqrt(v) for dd, v in by_deg.items()}          # L2 mass per stratum
    genuine = frozenset(t)
    gc = sf.get(genuine, 0.0)
    own_deg = len(genuine)
    comp = sorted(((abs(c), R) for R, c in sf.items() if len(R)==own_deg and R!=genuine), reverse=True)
    n_bigger = sum(1 for c,_ in comp if c >= abs(gc))
    print(f"\n--- target {t_idx}: {k} AND{t} ---")
    print(f"  raw-quartic mass touching const index 0: {const_touch_frac(SYt[t_idx]):.2f}"
          f"  (relocated into real strata by the const-fold, verified exact above)")
    print(f"  genuine coeff d{tuple(sorted(genuine))} = {gc:+.2f}  (its own degree-{own_deg} stratum)")
    print(f"  L2 mass by feature-degree stratum: " +
          "  ".join(f"deg{dd}:{by_deg[dd]:.2f}" for dd in range(5)))
    top = comp[0] if comp else (0.0, None)
    print(f"  largest same-stratum competitor: {top[0]:.2f}"
          f"{' (' + str(tuple(sorted(top[1]))) + ')' if top[1] else ''}"
          f"   genuine/competitor = {abs(gc)/max(top[0],1e-9):.2f}x")
    verdict = (f"largest in its stratum but only {abs(gc)/max(top[0],1e-9):.2f}x the runner-up"
               if n_bigger == 0 else
               f"NOT dominant -- {n_bigger} competitors are >= genuine (cross-target interference)")
    print(f"  competitors with |coeff| >= genuine: {n_bigger}  -> genuine detector {verdict}")

d4_idx=[i for i in range(TT) if deg[i]==4]; d2_idx=[i for i in range(TT) if deg[i]==2]
strata_report(max(d4_idx,key=lambda i:(Wo[i]**2).sum()))
strata_report(max(d2_idx,key=lambda i:(Wo[i]**2).sum()))

print("\n" + "-"*72)
print("Interpretation (both canonicalizations are provably exact; both are")
print("necessary gauge fixes, but NEITHER rescues the signal -- a refinement of")
print("the open threads, not the clean win the notes hoped for):")
print(" Part A: hollowing is the unique residual1.py gauge fix as a canonical form")
print("   -- it isolates the inhibition ladder as an explicit linear vector and")
print("   leaves diag(H)=0. But top-2 eigenvector alignment with span{e_a,e_b}")
print("   barely moves (~0.28 -> ~0.35) and the top-2 share DROPS: the diagonal")
print("   was not what hid the signal edge; off-diagonal interference (the")
print("   embedding crosstalk of factorize.py) still dominates the spectrum.")
print(" Part B: the square-free + const-fold form is exact and scalable (size =")
print("   #subsets up to degree 4, never m^4 past the sanity check). It relocates")
print("   all const-routed mass into the correct real strata, so 'const mass >")
print("   real mass' (mixed.py) is no longer an artifact of the representation.")
print("   But the genuine AND coefficient still does NOT dominate its own stratum:")
print("   cross-target interference coefficients are comparable/larger -- the")
print("   degree-4 analog of Part A. Stratification is necessary but not")
print("   sufficient; within a homogeneous stratum you still face interference")
print("   (threads #3/#5) plus the pairing-mix of decomp_exact.py (thread #4).")
print(" Through-line: at BOTH depths the dominant obstruction to reading off the")
print("   detector is interference between superposed targets, not the boolean")
print("   gauge artifacts (diagonal / const-routing) these fixes remove.")
