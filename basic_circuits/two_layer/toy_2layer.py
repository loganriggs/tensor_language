import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Two stacked bilinear layers (NO residual, NO const), computing 4-wise ANDs.
#   h = (W1a x)⊙(W1b x)   [layer 1: degree 2, width h1]
#   g = (W2a h)⊙(W2b h)   [layer 2: degree 4 in x, width h2]
#   logit = Wo g + bo
# Inputs: 5-hot over m=6 features (C(6,5)=6 inputs, each = "which feature is off").
# Outputs: the C(6,4)=15 four-wise ANDs.  5-hot means each input has C(5,4)=5
# co-active ANDs -> output superposition. Sweep (h1,h2) to find the minimum hidden
# widths that compute all 15 (the answer depends on input & output size).
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m, KHOT, DEG = 7, 5, 4         # 5-hot of 7 -> C(7,5)=21 inputs, C(7,4)=35 four-ANDs
quads = list(itertools.combinations(range(m), DEG)); T = len(quads)          # 15 four-ANDs
Xall = np.array([[1.0 if i in S else 0.0 for i in range(m)]
                 for S in itertools.combinations(range(m), KHOT)])            # (6,6)
Yall = np.stack([np.prod(Xall[:, list(q)], axis=1) for q in quads], 1)        # (6,15)
def sigmoid(z): z = np.clip(z, -60, 60); return 1/(1+np.exp(-z))

def train(h1, h2, seed, steps=6000, lr=0.02, wd=1e-3):
    rng = np.random.default_rng(seed)
    W1a = rng.normal(size=(h1, m))/np.sqrt(m); W1b = rng.normal(size=(h1, m))/np.sqrt(m)
    W2a = rng.normal(size=(h2, h1))/np.sqrt(h1); W2b = rng.normal(size=(h2, h1))/np.sqrt(h1)
    Wo = rng.normal(size=(T, h2))/np.sqrt(h2); bo = np.full(T, -0.7)
    ps = [W1a, W1b, W2a, W2b, Wo, bo]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        P1 = Xall @ W1a.T; Q1 = Xall @ W1b.T; h = P1*Q1
        P2 = h @ W2a.T; Q2 = h @ W2b.T; g = P2*Q2
        Z = g @ Wo.T + bo; dZ = (sigmoid(Z) - Yall)/len(Xall)
        dWo = dZ.T @ g; dbo = dZ.sum(0); dg = dZ @ Wo
        dP2 = dg*Q2; dQ2 = dg*P2; dW2a = dP2.T @ h; dW2b = dQ2.T @ h
        dh = dP2 @ W2a + dQ2 @ W2b
        dP1 = dh*Q1; dQ1 = dh*P1; dW1a = dP1.T @ Xall; dW1b = dQ1.T @ Xall
        for i, (p, gr) in enumerate(zip(ps, [dW1a, dW1b, dW2a, dW2b, dWo, dbo])):
            ms[i] = b1*ms[i] + (1-b1)*gr; vs[i] = b2*vs[i] + (1-b2)*gr*gr
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s)) + eps)
            if i < 5: p -= lr*wd*p
    P1 = Xall @ W1a.T; Q1 = Xall @ W1b.T; h = P1*Q1; g = (h @ W2a.T)*(h @ W2b.T)
    Z = g @ Wo.T + bo
    return dict(W1a=W1a, W1b=W1b, W2a=W2a, W2b=W2b, Wo=Wo, bo=bo, acc=((Z > 0) == (Yall > .5)).mean())

# ---- sweep both hidden widths ----
H1S, H2S, SEEDS = list(range(1, 9)), list(range(1, 11)), 10
grid = np.zeros((len(H1S), len(H2S))); best = {}
print(f"input/output: {len(Xall)} inputs (5-hot of {m}), {T} four-ANDs; sweeping h1 x h2 (best of {SEEDS} seeds)")
for i, h1 in enumerate(H1S):
    row = []
    for j, h2 in enumerate(H2S):
        b = max((train(h1, h2, s) for s in range(SEEDS)), key=lambda r: r['acc'])
        grid[i, j] = b['acc']; best[(h1, h2)] = b; row.append(f"{100*b['acc']:3.0f}")
    print(f"  h1={h1}: " + " ".join(row))
perfect = [(h1, h2) for h1 in H1S for h2 in H2S if grid[h1-1, h2-1] == 1.0]
pareto = sorted((a, b) for (a, b) in perfect if not any(c <= a and d <= b and (c, d) != (a, b) for (c, d) in perfect))
sel = min(perfect, key=lambda hh: (hh[0]+hh[1], hh[0]*hh[1]))
print(f"minimal (h1,h2) reaching 100%: frontier {pareto}; using {sel} for display")
M = best[sel]; W1a, W1b, W2a, W2b, Wo, bo = M['W1a'], M['W1b'], M['W2a'], M['W2b'], M['Wo'], M['bo']
h1, h2 = sel

# ---- fold the 2 layers to an exact degree-4 tensor per output (sanity + signal coeff) ----
Q1f = 0.5*(np.einsum('ki,kj->kij', W1a, W1b) + np.einsum('ki,kj->kij', W1b, W1a))     # (h1,m,m)
Acheck = np.einsum('pk,kij->pij', W2a, Q1f); Bcheck = np.einsum('pk,kij->pij', W2b, Q1f)  # (h2,m,m)
def sym4(A, B):
    Q = np.einsum('ij,kl->ijkl', A, B); Q = (Q + Q.transpose(2, 3, 0, 1))/2
    return sum(np.transpose(Q, pm) for pm in itertools.permutations(range(4)))/24
SY = np.stack([sym4(Acheck[p], Bcheck[p]) for p in range(h2)])                         # (h2,m,m,m,m)
T4 = np.einsum('tp,pijkl->tijkl', Wo, SY)                                              # (T,m,m,m,m)
Zc = np.einsum('tijkl,ni,nj,nk,nl->nt', T4, Xall, Xall, Xall, Xall) + bo
Zf = ((Xall@W1a.T*(Xall@W1b.T))@W2a.T * ((Xall@W1a.T*(Xall@W1b.T))@W2b.T))@Wo.T + bo
print(f"folded degree-4 tensor reproduces forward pass? max err {np.abs(Zc-Zf).max():.0e}")

# ---- show a few matrices from each layer ----
# Layer 1: per-neuron forms Q1f[k] = sym(W1a[k] (x) W1b[k]).  Layer 2: the contracted
# 3rd-order tensor Acheck/Bcheck[p] = sum_k W2a/b[p,k] Q1f[k]  (h2 x m x m), each slice
# an m x m matrix with g_p = (x^T Acheck[p] x)(x^T Bcheck[p] x).
p_show = np.argsort(-np.linalg.norm(Wo, axis=0))[:3]            # 3 most-used layer-2 neurons
ncol = max(h1, 2); fig, axes = plt.subplots(1 + len(p_show), ncol, figsize=(3.1*ncol, 2.9*(1+len(p_show))), squeeze=False)
for ax in axes.ravel(): ax.axis('off')
def show(ax, Mx, title):
    ax.axis('on'); lim = max(np.abs(Mx).max(), 1e-6); ax.imshow(Mx, cmap='RdBu_r', vmin=-lim, vmax=lim)
    for i in range(m):
        for j in range(m):
            if abs(Mx[i, j]) > 0.05: ax.text(j, i, f"{Mx[i,j]:.1f}", ha='center', va='center', fontsize=7)
    ax.set_xticks(range(m)); ax.set_yticks(range(m))
    ax.set_xticklabels([f'x{i}' for i in range(m)], fontsize=6); ax.set_yticklabels([f'x{i}' for i in range(m)], fontsize=6)
    ax.set_title(title, fontsize=9)
for k in range(h1): show(axes[0, k], Q1f[k], f'Layer-1 form Q1f[{k}]')
for r, p in enumerate(p_show):
    show(axes[r+1, 0], Acheck[p], f'Layer-2 Acheck[{p}] (contracted)')
    show(axes[r+1, 1], Bcheck[p], f'Layer-2 Bcheck[{p}] (contracted)')
fig.suptitle(f'A few weight matrices per layer (m={m}, model h1={h1},h2={h2}; each on its own scale)\n'
             'Layer 1: Q1f[k]=sym(W1a[k](x)W1b[k]).   Layer 2: Acheck/Bcheck[p]=Σ_k W2[p,k]·Q1f[k], '
             'g_p=(xᵀAx)(xᵀBx)', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_layers.png'), dpi=110)
np.savez(os.path.join(DIR, 'toy_2layer_model.npz'), W1a=W1a, W1b=W1b, W2a=W2a, W2b=W2b, Wo=Wo, bo=bo)

# ---- logit components per (input, output): bias / signal(genuine 4-AND term) / interference ----
def shares(n, t): return int(sum(Xall[n, f] for f in quads[t]))     # active target feats; 4=pos, 3=neg
rows = []
for n in range(len(Xall)):
    for t in range(T):
        a, b, c, d = quads[t]
        sig = 24*T4[t, a, b, c, d] * Xall[n, a]*Xall[n, b]*Xall[n, c]*Xall[n, d]
        interf = (Zc[n, t] - bo[t]) - sig
        rows.append((shares(n, t), bo[t], sig, interf))
sh = np.array([r[0] for r in rows]); comp = np.array([r[1:] for r in rows])
present = sorted(set(sh)); ypos = {k: i for i, k in enumerate(present)}
clab = {k: ('positive' if k == DEG else f'neg (shares {k})') for k in range(DEG+1)}
ccol = {DEG: '#2ca02c', DEG-1: '#d62728', DEG-2: '#ff7f0e', DEG-3: '#888', DEG-4: '#bbb'}

# ============================ FIGURES ============================
# (1) h1 x h2 accuracy grid
fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(100*grid, cmap='RdYlGn', vmin=50, vmax=100, aspect='auto')
for i, h1v in enumerate(H1S):
    for j, h2v in enumerate(H2S):
        ax.text(j, i, f"{100*grid[i,j]:.0f}", ha='center', va='center', fontsize=8,
                color='k', fontweight='bold' if grid[i, j] == 1 else 'normal')
for (a, b) in pareto:
    ax.add_patch(plt.Rectangle((H2S.index(b)-.5, H1S.index(a)-.5), 1, 1, fill=False, ec='blue', lw=2.5))
ax.set_xticks(range(len(H2S))); ax.set_xticklabels(H2S); ax.set_yticks(range(len(H1S))); ax.set_yticklabels(H1S)
ax.set_xlabel('h2 (layer-2 width)'); ax.set_ylabel('h1 (layer-1 width)')
ax.set_title(f'2-layer bilinear, 4-wise ANDs (5-hot of {m}): {len(Xall)} inputs, {T} outputs\n'
             f'best-of-{SEEDS} accuracy (%); blue = minimal (h1,h2) frontier reaching 100%')
fig.colorbar(im, label='accuracy (%)'); fig.tight_layout()
fig.savefig(os.path.join(RESULTS, 'fig_toy2L_hsweep.png'), dpi=110)

# (2) logit ladder + (3) ladder decomposition
def ladder(ax, L, title):
    for k in present:
        mk = sh == k
        ax.scatter(L[mk], ypos[k] + np.linspace(-.12, .12, max(mk.sum(), 1)), color=ccol[k], s=40, label=clab[k])
    ax.axvline(0, color='k', lw=1.5); ax.set_yticks(list(ypos.values())); ax.set_yticklabels([clab[k] for k in present])
    ax.set_xlabel('logit'); ax.set_title(title)
fig, ax = plt.subplots(figsize=(9, 3.0)); L = comp.sum(1)
ladder(ax, L, f'2-layer toy logit ladder — h1={h1},h2={h2}, {len(L)} decisions, {100*M["acc"]:.0f}% at threshold 0')
ax.legend(fontsize=8, loc='lower right'); fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_ladder.png'), dpi=110)

variants = [("full", [0, 1, 2]), ("no interference\n(bias+signal)", [0, 1]), ("no signal\n(bias+interference)", [0, 2])]
fig, axes = plt.subplots(1, 3, figsize=(16, 3.4), sharex=True, sharey=True)
print("ablation (threshold 0):")
for ax, (ttl, c) in zip(axes, variants):
    L = comp[:, c].sum(1); pos = L[sh == 4]; neg = L[sh != 4]
    acc = ((pos > 0).sum() + (neg <= 0).sum())/len(L)
    q = max(20, 1.25*np.percentile(np.abs(comp.sum(1)), 90)); ladder(ax, L, f"{ttl} — acc {100*acc:.0f}%"); ax.set_xlim(-q, q)
    print(f"  {ttl.splitlines()[0]:18s} acc {100*acc:.0f}%")
fig.suptitle('2-layer toy logit ladder decomposition (degree-4: bias + genuine 4-AND signal + interference)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy2L_ladder_decomp.png'), dpi=110)
print("figures: fig_toy2L_hsweep, fig_toy2L_layers, fig_toy2L_ladder, fig_toy2L_ladder_decomp")
