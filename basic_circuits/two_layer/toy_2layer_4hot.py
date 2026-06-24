import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# 4-hot variant: 2 stacked bilinear layers, 4-wise ANDs, but inputs are exactly
# 4-hot over m=6 -> C(6,4)=15 inputs = 15 outputs, each input lights EXACTLY ONE
# output (mutually exclusive / one-hot detection of "which 4-subset is active").
# On 4-hot the genuine monomial x_a x_b x_c x_d is itself the perfect detector, so
# we expect a clean solution to surface (unlike the entangled 5-hot case).
#   - sweep (h1,h2) for >=99%
#   - decompose to per-output multilinear polynomials; common structure
#   - canonicalize each output's quartic (eigh of the (ij)(kl) matricisation) and
#     check whether the genuine 4-AND structure pops out.
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m, KHOT = 6, 4
quads = list(itertools.combinations(range(m), 4)); T = len(quads)
Xall = np.array([[1.0 if i in S else 0.0 for i in range(m)] for S in itertools.combinations(range(m), KHOT)])
Yall = np.stack([np.prod(Xall[:, list(q)], axis=1) for q in quads], 1)
print(f"{len(Xall)} inputs (4-hot of {m}), {T} four-ANDs; positives/decision = "
      f"{int(Yall.sum())}/{Yall.size} (one-hot: each input lights {int(Yall.sum(1).mean())} output)")
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))

def train(h1, h2, seed, steps=7000, lr=0.02, wd=2e-2):
    rng = np.random.default_rng(seed)
    W1a = rng.normal(size=(h1, m))/np.sqrt(m); W1b = rng.normal(size=(h1, m))/np.sqrt(m)
    W2a = rng.normal(size=(h2, h1))/np.sqrt(h1); W2b = rng.normal(size=(h2, h1))/np.sqrt(h1)
    Wo = rng.normal(size=(T, h2))/np.sqrt(h2); bo = np.full(T, -1.5)
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

# ---- sweep for >=99% ----
H1S, H2S, SEEDS, THRESH = list(range(1, 7)), list(range(1, 9)), 12, 0.99
grid = np.zeros((len(H1S), len(H2S))); best = {}
print(f"sweeping h1 x h2 (best of {SEEDS} seeds), target >= {100*THRESH:.0f}%")
for i, h1 in enumerate(H1S):
    row = []
    for j, h2 in enumerate(H2S):
        rs = [train(h1, h2, s) for s in range(SEEDS)]
        b = max(rs, key=lambda r: r['acc']); grid[i, j] = b['acc']
        ok = [r for r in rs if r['acc'] >= THRESH]
        best[(h1, h2)] = min(ok, key=lambda r: r['norm']) if ok else b
        row.append(f"{100*b['acc']:3.0f}")
    print(f"  h1={h1}: " + " ".join(row))
perfect = [(h1, h2) for h1 in H1S for h2 in H2S if grid[h1-1, h2-1] >= THRESH]
pareto = sorted((a, b) for (a, b) in perfect if not any(c <= a and d <= b and (c, d) != (a, b) for (c, d) in perfect))
sel = min(perfect, key=lambda hh: (hh[0]+hh[1], hh[0]*hh[1]))
print(f"minimal (h1,h2) reaching >={100*THRESH:.0f}%: frontier {pareto}; using {sel}")
M = best[sel]; W1a, W1b, W2a, W2b, Wo, bo = [M[k] for k in ('W1a', 'W1b', 'W2a', 'W2b', 'Wo', 'bo')]; h1, h2 = sel
print(f"  chosen model acc {100*M['acc']:.1f}%  L1 norm {M['norm']:.0f}")

# ---- fold + square-free reduce ----
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b)+np.einsum('ki,kj->kij', W1b, W1a))
Ac = np.einsum('pk,kij->pij', W2a, Q1f); Bc = np.einsum('pk,kij->pij', W2b, Q1f)
def sym4(A, B):
    Q = np.einsum('ij,kl->ijkl', A, B); Q = (Q+Q.transpose(2, 3, 0, 1))/2
    return sum(np.transpose(Q, pm) for pm in itertools.permutations(range(4)))/24
SY = np.stack([sym4(Ac[p], Bc[p]) for p in range(h2)]); T4 = np.einsum('tp,pijkl->tijkl', Wo, SY)
tuples = list(itertools.product(range(m), repeat=4)); flat = np.ravel_multi_index(np.array(tuples).T, (m,)*4)
keys = sorted(set(frozenset(t) for t in tuples), key=lambda s: (len(s), sorted(s)))
k2c = {k: i for i, k in enumerate(keys)}; G = np.zeros((len(keys), m**4))
for j, t in enumerate(tuples): G[k2c[frozenset(t)], j] = 1
coeff = (T4.reshape(T, -1)[:, flat]) @ G.T
val = sum(np.outer(np.prod(Xall[:, list(k)], axis=1), coeff[:, ci]) for ci, k in enumerate(keys)) + bo
print(f"square-free reproduces logits? max err {np.abs(val-Yall*0-( ((Xall@W1a.T*(Xall@W1b.T))@W2a.T*((Xall@W1a.T*(Xall@W1b.T))@W2b.T))@Wo.T+bo )).max():.0e}")

# how dominant is the genuine 4-AND term now? (fraction of each output's L1 coeff mass)
gen_frac = np.array([abs(coeff[t, k2c[frozenset(quads[t])]])/np.abs(coeff[t]).sum() for t in range(T)])
print(f"genuine 4-AND coeff as fraction of |coeff| mass: mean {gen_frac.mean():.2f}  "
      f"(its rank among |coeff|, mean) {np.mean([1+int((np.abs(coeff[t])>abs(coeff[t,k2c[frozenset(quads[t])]])).sum()) for t in range(T)]):.1f} of {len(keys)}")

# common structure (normalised per output) by (degree, overlap)
cn = coeff/np.linalg.norm(coeff, axis=1, keepdims=True)
gridc = np.full((4, 5), np.nan); cnt = np.zeros((4, 5))
for t in range(T):
    tgt = set(quads[t])
    for ci, k in enumerate(keys):
        dgr, ov = len(k), len(set(k) & tgt)
        gridc[dgr-1, ov] = (0 if np.isnan(gridc[dgr-1, ov]) else gridc[dgr-1, ov]) + cn[t, ci]; cnt[dgr-1, ov] += 1
gridc = gridc/np.where(cnt > 0, cnt, 1)

# ============================ FIGURES ============================
fig, ax = plt.subplots(figsize=(7.5, 5))
im = ax.imshow(100*grid, cmap='RdYlGn', vmin=60, vmax=100, aspect='auto')
for i in range(len(H1S)):
    for j in range(len(H2S)): ax.text(j, i, f"{100*grid[i,j]:.0f}", ha='center', va='center', fontsize=8,
                                      fontweight='bold' if grid[i, j] >= THRESH else 'normal')
for (a, b) in pareto: ax.add_patch(plt.Rectangle((H2S.index(b)-.5, H1S.index(a)-.5), 1, 1, fill=False, ec='blue', lw=2.5))
ax.set_xticks(range(len(H2S))); ax.set_xticklabels(H2S); ax.set_yticks(range(len(H1S))); ax.set_yticklabels(H1S)
ax.set_xlabel('h2'); ax.set_ylabel('h1')
ax.set_title(f'4-hot (mutually-exclusive) 2-layer: {len(Xall)} inputs, {T} one-hot outputs\n'
             f'best-of-{SEEDS} accuracy; blue = minimal frontier reaching >={100*THRESH:.0f}%')
fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_4hot_hsweep.png'), dpi=110)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
lim = np.nanmax(np.abs(gridc)); im = axes[0].imshow(gridc, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
for dgr in range(4):
    for ov in range(5):
        if cnt[dgr, ov] > 0: axes[0].text(ov, dgr, f"{gridc[dgr,ov]:+.2f}", ha='center', va='center', fontsize=8)
axes[0].set_yticks(range(4)); axes[0].set_yticklabels([1, 2, 3, 4]); axes[0].set_xlabel('# subset feats in target'); axes[0].set_ylabel('degree')
axes[0].set_title('Common structure (norm. mean coeff)\nby degree x target-overlap'); fig.colorbar(im, ax=axes[0], shrink=.8)
axes[1].hist(gen_frac, bins=12, color='#2ca02c'); axes[1].set_xlabel('genuine 4-AND coeff / total |coeff|')
axes[1].set_ylabel('# outputs'); axes[1].set_title(f'How dominant is the genuine term?\nmean {gen_frac.mean():.2f}')
# canonicalization: eigh of one output's (ij)(kl) matricisation
t = 0; Mt = T4[t].reshape(m*m, m*m); w, V = np.linalg.eigh(Mt); o = np.argsort(-np.abs(w)); w = w[o]
axes[2].bar(range(10), w[:10], color='#1f77b4'); axes[2].axhline(0, color='k', lw=.8)
axes[2].set_xlabel('eigenvalue rank'); axes[2].set_title(f'Canonicalisation: eigvals of AND{quads[t]} quartic\n(matricised (ij)(kl); low rank = clean)')
fig.suptitle(f'4-hot decomposition (model h1={h1},h2={h2}, acc {100*M["acc"]:.0f}%)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_4hot_decomp.png'), dpi=110)

# ---- the structure that actually pops out: the geometric embedding ----
hh = (Xall@W1a.T)*(Xall@W1b.T); gg = (hh@W2a.T)*(hh@W2b.T)        # layer-1 / layer-2 outputs
try:
    from scipy.spatial import ConvexHull; nh = len(set(ConvexHull(gg[:, :2]).vertices))
except Exception: nh = -1
fig, axx = plt.subplots(1, 2, figsize=(13, 5.5))
for a, Z, ttl in [(axx[0], gg, f'layer-2 output g (first 2 dims) — {nh}/{T} on convex hull'), (axx[1], hh, 'layer-1 output h (first 2 dims)')]:
    a.scatter(Z[:, 0], Z[:, 1], s=55, zorder=3)
    for t in range(T): a.annotate(''.join(map(str, quads[t])), (Z[t, 0], Z[t, 1]), fontsize=7, ha='center', va='center', color='white')
    a.set_title(ttl)
fig.suptitle(f'4-hot one-hot toy (h1={h1},h2={h2}): the {T} mutually-exclusive patterns are embedded as a 2-D arrangement\n'
             '(each pattern separable from the rest = one-hot) — a geometric solution, not genuine monomials', fontsize=11)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_4hot_embed.png'), dpi=110)
print("figures: fig_toy2L_4hot_hsweep, fig_toy2L_4hot_decomp, fig_toy2L_4hot_embed")
