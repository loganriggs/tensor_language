import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Non-orthogonal sparse pursuit for a symmetric quartic (see sparse_pursuit.md).
# A degree-4 form is matricised to M[(ij),(kl)] (m^2 x m^2, symmetric). Orthogonal
# eigh gives eigen-matrices that are forced ±(e_ab ± e_cd)/sqrt2 MIXTURES of
# complementary pairings (decomp_exact.py). The de-mixing: each +/- eigenvalue pair
# (v+, v-) of equal magnitude comes from ONE product (xᵀ A x)(xᵀ B x); recover
#     A ∝ v+ + v-,   B ∝ v+ - v-
# which un-rotates the 45° mixture back to the (hopefully sparse) factors. When
# several products are present the +/- eigenvectors must be MATCHED; we pick the
# matching that minimises support (disjoint-support / sparsity tie-break).
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m = 6

def matricise(T4): M = T4.reshape(m*m, m*m); return 0.5*(M + M.T)
def nnz(A, frac=0.1): return int((np.abs(A) > frac*np.abs(A).max()).sum())
def edges(A, frac=0.1):
    th = frac*np.abs(A).max(); return [(i, j) for i in range(m) for j in range(i+1, m) if abs(A[i, j]) > th]

def sparse_pursuit(T4, K=None):
    """Return list of (A,B) product factors, de-mixed with a min-support matching."""
    M = matricise(T4); w, V = np.linalg.eigh(M)
    sig = np.abs(w) > 1e-6*np.abs(w).max(); w, V = w[sig], V[:, sig]
    pos = list(np.argsort(-w)[w[np.argsort(-w)] > 0]); neg = list(np.argsort(w)[w[np.argsort(w)] < 0])
    if K: pos, neg = pos[:K], neg[:K]
    n = min(len(pos), len(neg))
    def fac(pi, ni): s = np.sqrt(abs(w[pi])); return (s*(V[:, pi]+V[:, ni])).reshape(m, m), (s*(V[:, pi]-V[:, ni])).reshape(m, m)
    best = min(itertools.permutations(range(len(neg)), n),
               key=lambda perm: sum(nnz(fac(pos[i], neg[perm[i]])[0]) + nnz(fac(pos[i], neg[perm[i]])[1]) for i in range(n)))
    return [fac(pos[i], neg[best[i]]) for i in range(n)], w

# ===== (1) ideal 4-AND tensor: pursuit recovers the disjoint pairings =====
tgt = (0, 1, 2, 3); T4i = np.zeros((m, m, m, m))
for pm in itertools.permutations(tgt): T4i[pm] = 1/24
facs_i, w_i = sparse_pursuit(T4i)
print(f"ideal AND{tgt}: eigvals {np.round(np.sort(w_i),3)}")
for r, (A, B) in enumerate(facs_i): print(f"  term {r}: A edges {edges(A)}  B edges {edges(B)}  (nnz {nnz(A)+nnz(B)})")

# ===== (2) a LEARNED quartic (4-hot toy, h1=2,h2=2) =====
KHOT, h1, h2 = 4, 2, 2
quads = list(itertools.combinations(range(m), 4)); T = len(quads)
Xall = np.array([[1.0 if i in S else 0.0 for i in range(m)] for S in itertools.combinations(range(m), KHOT)])
Yall = np.stack([np.prod(Xall[:, list(q)], axis=1) for q in quads], 1)
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))
def train(seed, wd=2e-2, steps=8000, lr=0.02):
    rng = np.random.default_rng(seed)
    W = [rng.normal(size=s)/np.sqrt(s[1]) for s in [(h1, m), (h1, m), (h2, h1), (h2, h1), (T, h2)]] + [np.full(T, -1.5)]
    ms = [np.zeros_like(p) for p in W]; vs = [np.zeros_like(p) for p in W]; b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        W1a, W1b, W2a, W2b, Wo, bo = W
        P1 = Xall@W1a.T; Q1 = Xall@W1b.T; h = P1*Q1; P2 = h@W2a.T; Q2 = h@W2b.T; g = P2*Q2
        dZ = (sigmoid(g@Wo.T+bo)-Yall)/len(Xall)
        dg = dZ@Wo; dP2 = dg*Q2; dQ2 = dg*P2; dh = dP2@W2a+dQ2@W2b; dP1 = dh*Q1; dQ1 = dh*P1
        grads = [dP1.T@Xall, dQ1.T@Xall, dP2.T@h, dQ2.T@h, dZ.T@g, dZ.sum(0)]
        for i, (p, gr) in enumerate(zip(W, grads)):
            ms[i] = b1*ms[i]+(1-b1)*gr; vs[i] = b2*vs[i]+(1-b2)*gr*gr
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
            if i < 5: p -= lr*wd*p
    W1a, W1b, W2a, W2b, Wo, bo = W
    g = ((Xall@W1a.T*(Xall@W1b.T))@W2a.T)*((Xall@W1a.T*(Xall@W1b.T))@W2b.T)
    return W, ((g@Wo.T+bo > 0) == (Yall > .5)).mean()
W, acc = max((train(s) for s in range(8)), key=lambda r: r[1])
W1a, W1b, W2a, W2b, Wo, bo = W
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b)+np.einsum('ki,kj->kij', W1b, W1a))
Ac = np.einsum('pk,kij->pij', W2a, Q1f); Bc = np.einsum('pk,kij->pij', W2b, Q1f)
def sym4(A, B):
    Q = np.einsum('ij,kl->ijkl', A, B); Q = (Q+Q.transpose(2, 3, 0, 1))/2
    return sum(np.transpose(Q, pm) for pm in itertools.permutations(range(4)))/24
t = 0; T4l = np.einsum('p,pijkl->ijkl', Wo[t], np.stack([sym4(Ac[p], Bc[p]) for p in range(h2)]))
facs_l, w_l = sparse_pursuit(T4l, K=3)
print(f"\nlearned model acc {100*acc:.0f}%; output AND{quads[t]} eigvals {np.round(np.sort(w_l)[:6],1)}...")
for r, (A, B) in enumerate(facs_l): print(f"  term {r}: A edges {edges(A)}  B edges {edges(B)}  (nnz {nnz(A)+nnz(B)})")

# ===== figure: ideal (sparse edges) vs learned (dense) =====
fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
def heat(ax, Mx, ttl):
    lim = max(np.abs(Mx).max(), 1e-9); ax.imshow(Mx, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax.set_xticks(range(m)); ax.set_yticks(range(m)); ax.set_xticklabels(range(m), fontsize=6); ax.set_yticklabels(range(m), fontsize=6); ax.set_title(ttl, fontsize=9)
axes[0, 0].plot(np.sort(w_i), 'o-'); axes[0, 0].axhline(0, color='k', lw=.5); axes[0, 0].set_title(f'ideal AND{tgt}\neigvals (3 ± pairs)', fontsize=9)
heat(axes[0, 1], facs_i[0][0], f'pursuit A = edge {edges(facs_i[0][0])}'); heat(axes[0, 2], facs_i[0][1], f'pursuit B = edge {edges(facs_i[0][1])}')
axes[0, 3].axis('off'); axes[0, 3].text(0, .5, f'SPARSE: recovers the\n3 disjoint pairings\n(single edges each).\n\northogonal eigh would\ngive ±(e_ab±e_cd)/√2\nmixtures instead.', fontsize=9, va='center')
axes[1, 0].plot(np.sort(w_l), 'o-'); axes[1, 0].axhline(0, color='k', lw=.5); axes[1, 0].set_title(f'learned AND{quads[t]}\neigvals', fontsize=9)
heat(axes[1, 1], facs_l[0][0], f'pursuit A (nnz {nnz(facs_l[0][0])})'); heat(axes[1, 2], facs_l[0][1], f'pursuit B (nnz {nnz(facs_l[0][1])})')
axes[1, 3].axis('off'); axes[1, 3].text(0, .5, 'DENSE: the learned\nfactors are not single\nedges -- the model\nsolved it geometrically,\nnot as sparse\nconjunctions.', fontsize=9, va='center')
fig.suptitle('Non-orthogonal sparse pursuit: ideal 4-AND -> clean disjoint-pair edges; learned net -> dense factors', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_sparse_pursuit.png'), dpi=110)
print("figure: fig_sparse_pursuit")
