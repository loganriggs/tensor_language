import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Decompose the 2-layer 4-wise-AND toy into polynomials and look for structure:
#  (1) per-output square-free polynomial (multilinear coeffs, exact on booleans);
#  (2) common structure across outputs -- each output is an INDEPENDENT sigmoid+BCE
#      head (no softmax), so we normalise each output's coeff vector and compare;
#  (3) a cross-output shared component (SVD of the normalised coeff matrix);
#  (4) layer-1 -> layer-2 reuse (W2 columns; the contracted forms).
# Trained with stronger weight decay than toy_2layer.py to get the LOW-NORM
# (interpretable) solution -- the default tiny-wd net hides everything under huge
# cancelling coefficients.
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m, KHOT, h1, h2 = 7, 5, 3, 5
quads = list(itertools.combinations(range(m), 4)); T = len(quads)
Xall = np.array([[1.0 if i in S else 0.0 for i in range(m)] for S in itertools.combinations(range(m), KHOT)])
Yall = np.stack([np.prod(Xall[:, list(q)], axis=1) for q in quads], 1)
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))

def train(seed, wd=0.05, steps=9000, lr=0.02):
    rng = np.random.default_rng(seed)
    W1a = rng.normal(size=(h1, m))/np.sqrt(m); W1b = rng.normal(size=(h1, m))/np.sqrt(m)
    W2a = rng.normal(size=(h2, h1))/np.sqrt(h1); W2b = rng.normal(size=(h2, h1))/np.sqrt(h1)
    Wo = rng.normal(size=(T, h2))/np.sqrt(h2); bo = np.full(T, -0.7)
    ps = [W1a, W1b, W2a, W2b, Wo, bo]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        P1 = Xall@W1a.T; Q1 = Xall@W1b.T; h = P1*Q1; P2 = h@W2a.T; Q2 = h@W2b.T; g = P2*Q2
        Z = g@Wo.T+bo; dZ = (sigmoid(Z)-Yall)/len(Xall)
        dWo = dZ.T@g; dbo = dZ.sum(0); dg = dZ@Wo; dP2 = dg*Q2; dQ2 = dg*P2; dW2a = dP2.T@h; dW2b = dQ2.T@h
        dh = dP2@W2a+dQ2@W2b; dP1 = dh*Q1; dQ1 = dh*P1; dW1a = dP1.T@Xall; dW1b = dQ1.T@Xall
        for i, (p, gr) in enumerate(zip(ps, [dW1a, dW1b, dW2a, dW2b, dWo, dbo])):
            ms[i] = b1*ms[i]+(1-b1)*gr; vs[i] = b2*vs[i]+(1-b2)*gr*gr
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s))+eps)
            if i < 5: p -= lr*wd*p
    g = ((Xall@W1a.T*(Xall@W1b.T))@W2a.T)*((Xall@W1a.T*(Xall@W1b.T))@W2b.T); Z = g@Wo.T+bo
    return dict(W1a=W1a, W1b=W1b, W2a=W2a, W2b=W2b, Wo=Wo, bo=bo, acc=((Z > 0) == (Yall > .5)).mean(),
                norm=sum(np.abs(p).sum() for p in [W1a, W1b, W2a, W2b, Wo]))
# lowest-norm 100% model over seeds (the cleanest interpretable solution)
M = min((r for r in (train(s) for s in range(10)) if r['acc'] == 1.0), key=lambda r: r['norm'])
W1a, W1b, W2a, W2b, Wo, bo = M['W1a'], M['W1b'], M['W2a'], M['W2b'], M['Wo'], M['bo']
print(f"regularised model: acc {100*M['acc']:.0f}%, L1 weight norm {M['norm']:.0f}")

# ---- fold + square-free reduce each output to a multilinear polynomial ----
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b)+np.einsum('ki,kj->kij', W1b, W1a))
Ac = np.einsum('pk,kij->pij', W2a, Q1f); Bc = np.einsum('pk,kij->pij', W2b, Q1f)
def sym4(A, B):
    Q = np.einsum('ij,kl->ijkl', A, B); Q = (Q+Q.transpose(2, 3, 0, 1))/2
    return sum(np.transpose(Q, pm) for pm in itertools.permutations(range(4)))/24
T4 = np.einsum('tp,pijkl->tijkl', Wo, np.stack([sym4(Ac[p], Bc[p]) for p in range(h2)]))
tuples = list(itertools.product(range(m), repeat=4)); flat = np.ravel_multi_index(np.array(tuples).T, (m,)*4)
keys = sorted(set(frozenset(t) for t in tuples), key=lambda s: (len(s), sorted(s)))
k2c = {k: i for i, k in enumerate(keys)}; G = np.zeros((len(keys), m**4))
for j, t in enumerate(tuples): G[k2c[frozenset(t)], j] = 1
coeff = (T4.reshape(T, -1)[:, flat]) @ G.T                  # (T, #subsets) multilinear coeffs
val = sum(np.outer(np.prod(Xall[:, list(k)], axis=1), coeff[:, ci]) for ci, k in enumerate(keys)) + bo
Zf = ((Xall@W1a.T*(Xall@W1b.T))@W2a.T*((Xall@W1a.T*(Xall@W1b.T))@W2b.T))@Wo.T+bo
print(f"square-free polynomial reproduces logits? max err {np.abs(val-Zf).max():.0e}")
degk = np.array([len(k) for k in keys])

# ---- (2) common structure: normalise each output, mean coeff by (degree, overlap) ----
cn = coeff/np.linalg.norm(coeff, axis=1, keepdims=True)        # rescale each output to unit norm
grid = np.full((4, 5), np.nan); cnt = np.zeros((4, 5))
for t in range(T):
    tgt = set(quads[t])
    for ci, k in enumerate(keys):
        dgr, ov = len(k), len(set(k) & tgt)
        if np.isnan(grid[dgr-1, ov]): grid[dgr-1, ov] = 0.0
        grid[dgr-1, ov] += cn[t, ci]; cnt[dgr-1, ov] += 1
grid = grid/np.where(cnt > 0, cnt, 1)
print("\nmean normalised coeff by (degree, #features-in-target):")
for dgr in range(1, 5):
    print("  deg%d: " % dgr + "  ".join(f"ov{ov}:{grid[dgr-1,ov]:+.3f}" for ov in range(dgr+1) if cnt[dgr-1, ov] > 0))

# ---- (3) cross-output shared component ----
U, s, Vt = np.linalg.svd(cn - cn.mean(0), full_matrices=False)
print(f"\ncross-output normalised-coeff SVD: var% of top comps {np.round(100*s[:4]**2/(s**2).sum(),1)}"
      f"  (mean vector L2 {np.linalg.norm(cn.mean(0)):.2f})")

# ============================ FIGURES ============================
# Fig A: common-structure grid + two example output polynomials
fig = plt.figure(figsize=(15, 5)); gs = fig.add_gridspec(1, 3)
ax = fig.add_subplot(gs[0, 0]); lim = np.nanmax(np.abs(grid))
im = ax.imshow(grid, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
for dgr in range(4):
    for ov in range(5):
        if cnt[dgr, ov] > 0: ax.text(ov, dgr, f"{grid[dgr,ov]:+.2f}", ha='center', va='center', fontsize=8)
ax.set_xlabel('# of subset features that are in the target'); ax.set_ylabel('subset degree')
ax.set_yticks(range(4)); ax.set_yticklabels([1, 2, 3, 4]); ax.set_xticks(range(5))
ax.set_title('Common structure (mean over all 35 outputs,\neach normalised): coeff by degree x target-overlap')
fig.colorbar(im, ax=ax, shrink=.8)
for col, t in [(1, 0), (2, 17)]:
    ax = fig.add_subplot(gs[0, col]); tgt = set(quads[t])
    order = np.argsort(-np.abs(coeff[t]))[:10]
    cols = ['#2ca02c' if set(keys[ci]) == tgt else ('#1f77b4' if len(set(keys[ci]) & tgt) == len(keys[ci])
            else '#d62728') for ci in order]
    ax.barh(range(len(order)), coeff[t][order][::-1], color=cols[::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([''.join(map(str, sorted(keys[ci]))) for ci in order][::-1], fontsize=7)
    ax.axvline(0, color='k', lw=.8); ax.set_xlabel('coeff')
    ax.set_title(f'AND{quads[t]} top terms\n(green=genuine 4-AND, blue=⊆target, red=other)', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_polydecomp.png'), dpi=110)

# Fig B: layer-1 -> layer-2 reuse
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
pair_idx = list(itertools.combinations(range(m), 2))
L1pairs = np.array([[Q1f[k][i, j] for (i, j) in pair_idx] for k in range(h1)])   # (h1, 21) off-diag content
for ax, W, ttl in [(axes[0], W2a, 'W2a (layer-2 reads layer-1)'), (axes[1], W2b, 'W2b')]:
    lim = np.abs(W).max(); ax.imshow(W, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    for p in range(h2):
        for k in range(h1): ax.text(k, p, f"{W[p,k]:+.1f}", ha='center', va='center', fontsize=8)
    ax.set_xlabel('layer-1 neuron k'); ax.set_ylabel('layer-2 unit p'); ax.set_xticks(range(h1)); ax.set_yticks(range(h2))
    ax.set_title(ttl, fontsize=10)
lim = np.abs(L1pairs).max(); im = axes[2].imshow(L1pairs, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
axes[2].set_xticks(range(len(pair_idx))); axes[2].set_xticklabels([f'{i}{j}' for (i, j) in pair_idx], fontsize=5, rotation=90)
axes[2].set_yticks(range(h1)); axes[2].set_ylabel('layer-1 neuron k')
axes[2].set_title('layer-1 forms Q1f[k] in the pair basis\n(dense = not aligned to single pairs)', fontsize=10)
fig.colorbar(im, ax=axes[2], shrink=.8)
use = np.linalg.norm(W2a, axis=0) + np.linalg.norm(W2b, axis=0)
fig.suptitle(f'Layer-1 -> layer-2 reuse. Layer-1 usage (||W2a||+||W2b|| per neuron): '
             f'{np.round(use,1).tolist()} (neuron {int(use.argmax())} reused most). '
             'NB layer-1 basis has a GL(h1) bond gauge -> not canonical without bond-canonicalisation (thread #4).', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_reuse.png'), dpi=110)
print("figures: fig_toy2L_polydecomp, fig_toy2L_reuse")
