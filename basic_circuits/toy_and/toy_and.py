import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Minimal Universal-AND: 1 bilinear layer, 3 boolean inputs, h=2 hidden units,
# 3 outputs = the 3 pairwise ANDs (no embedding, no residual).
#   h = (W1 x) ⊙ (W2 x)   (W1,W2: 2x3) ,  logit = Wo h + bo   (Wo: 3x2)
# 3 quadratic-form outputs squeezed through only 2 bilinear neurons -> weight
# superposition at the smallest scale where it is non-trivial. We train on the
# full 8-input truth table and check it gets all 3 ANDs, then display it the same
# way as the big model: Qf decomposition + logit ladder + ladder decomposition.
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m, h, = 3, 2
pairs = list(itertools.combinations(range(m), 2)); T = len(pairs); pi = np.array(pairs)   # (0,1),(0,2),(1,2)
Xall = np.array(list(itertools.product([0, 1], repeat=m)), float)                          # all 8 inputs
Yall = np.stack([Xall[:, a]*Xall[:, b] for a, b in pairs], 1)                              # (8,3) targets
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train(seed, steps=8000, lr=0.03):
    rng = np.random.default_rng(seed)
    W1 = rng.normal(size=(h, m))/np.sqrt(m); W2 = rng.normal(size=(h, m))/np.sqrt(m)
    Wo = rng.normal(size=(T, h))/np.sqrt(h); bo = np.full(T, -2.0)
    ps = [W1, W2, Wo, bo]; ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    b1, b2, eps = 0.9, 0.999, 1e-8
    for s in range(1, steps+1):
        P = Xall @ W1.T; Q = Xall @ W2.T; H = P*Q; Z = H @ Wo.T + bo; Pr = sigmoid(Z)
        dZ = (Pr - Yall)/len(Xall)
        dWo = dZ.T @ H; dbo = dZ.sum(0); dH = dZ @ Wo; dP = dH*Q; dQ = dH*P
        dW1 = dP.T @ Xall; dW2 = dQ.T @ Xall
        for i, (p, g) in enumerate(zip(ps, [dW1, dW2, dWo, dbo])):
            ms[i] = b1*ms[i] + (1-b1)*g; vs[i] = b2*vs[i] + (1-b2)*g*g
            p -= lr*(ms[i]/(1-b1**s))/(np.sqrt(vs[i]/(1-b2**s)) + eps)
    Z = (Xall @ W1.T * (Xall @ W2.T)) @ Wo.T + bo
    bce = -(Yall*np.log(sigmoid(Z)+1e-12) + (1-Yall)*np.log(1-sigmoid(Z)+1e-12)).mean()
    acc = ((Z > 0) == (Yall > .5)).mean()
    return dict(W1=W1, W2=W2, Wo=Wo, bo=bo, acc=acc, bce=bce)

# best over a handful of seeds (does h=2 suffice for 3 ANDs?)
runs = [train(s) for s in range(12)]
n_perfect = sum(r['acc'] == 1.0 for r in runs)
best = min((r for r in runs), key=lambda r: (-(r['acc']), r['bce']))
W1, W2, Wo, bo = best['W1'], best['W2'], best['Wo'], best['bo']
print(f"h=2, 3 ANDs: {n_perfect}/12 seeds reached 100% on all 24 decisions; "
      f"best acc {100*best['acc']:.0f}%  BCE {best['bce']:.4f}")

Qf = np.einsum('tk,ki,kj->tij', Wo, W1, W2); Qf = 0.5*(Qf + Qf.transpose(0, 2, 1))           # (3,3,3)
Zc = np.einsum('tij,ni,nj->nt', Qf, Xall, Xall) + bo
print("Qf reproduces forward pass? max err", np.abs(Zc - ((Xall@W1.T*(Xall@W2.T))@Wo.T + bo)).max())
print("biases: bo =", {f"AND{pairs[t]}": round(float(bo[t]), 1) for t in range(T)},
      " (= the all-zeros-input logit; W1,W2 are biasless)")

print("\ntruth table (input -> logits / predictions):")
print("  x0x1x2 | " + " | ".join(f"AND{p}" for p in pairs))
for n in range(len(Xall)):
    lg = Zc[n]; pr = (lg > 0).astype(int); tg = Yall[n].astype(int)
    mark = "".join("." if pr[t] == tg[t] else "X" for t in range(T))
    print(f"  {''.join(str(int(v)) for v in Xall[n])}    | " +
          " | ".join(f"{lg[t]:+5.1f}{'*' if tg[t] else ' '}" for t in range(T)) + f"   {mark}")

# ---------- (1) Qf decomposition: signal / diagonal inhibition / interference ----------
lim = np.abs(Qf).max()
fig, axes = plt.subplots(T, 4, figsize=(11, 8.5))
cols = ['Qf (full)', '= signal (AND-ing)', '+ diagonal inhibition', '+ interference']
for t in range(T):
    a, b = pi[t]; M = Qf[t]
    Sg = np.zeros_like(M); Sg[a, b] = M[a, b]; Sg[b, a] = M[b, a]
    Dg = np.diag(np.diag(M)); I_ = M - Sg - Dg
    for c, mat in enumerate([M, Sg, Dg, I_]):
        ax = axes[t, c]; ax.imshow(mat, cmap='RdBu_r', vmin=-lim, vmax=lim)
        for i in range(m):
            for j in range(m):
                if abs(mat[i, j]) > 1e-2: ax.text(j, i, f"{mat[i,j]:.1f}", ha='center', va='center', fontsize=8)
        ax.set_xticks(range(m)); ax.set_yticks(range(m)); ax.set_xticklabels([f'x{i}' for i in range(m)], fontsize=7)
        ax.set_yticklabels([f'x{i}' for i in range(m)], fontsize=7)
        if t == 0: ax.set_title(cols[c], fontsize=11)
    axes[t, 0].set_ylabel(f'output AND(x{a},x{b})\nbias bo = {bo[t]:+.1f}', fontsize=10)
fig.suptitle('Toy Qf decomposition — logit = bias + signal + diagonal inhibition + interference\n'
             '(3 outputs through 2 bilinear neurons; W1,W2 are biasless, bo is the only bias)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy_decomp.png'), dpi=110)

# ---------- logit components for every (input, output) ----------
diag = Qf[:, np.arange(m), np.arange(m)]
def comps(n, t):
    a, b = pi[t]; x = Xall[n]; inh = (diag[t]*x).sum(); sig = interf = 0.0
    for (i, j) in pairs:
        v = 2*Qf[t, i, j]*x[i]*x[j]
        if (i, j) == (a, b): sig += v
        else: interf += v
    return bo[t], inh, sig, interf
rows = []
for n in range(len(Xall)):
    for t in range(T):
        a, b = pi[t]; k = int(Xall[n, a] + Xall[n, b]); cls = 'pos' if k == 2 else 'neg1' if k == 1 else 'neg0'
        rows.append((n, t, cls, *comps(n, t)))
cls_arr = np.array([r[2] for r in rows]); comp = np.array([r[3:] for r in rows])   # (24,4): bias,inhib,signal,interf
colors = {'neg0': '#888', 'neg1': '#d62728', 'pos': '#2ca02c'}
ypos = {'neg0': 0, 'neg1': 1, 'pos': 2}; labs = {'neg0': 'neg, shares 0', 'neg1': 'neg, shares 1', 'pos': 'positive'}

# ---------- (2) logit ladder ----------
fig, ax = plt.subplots(figsize=(9, 3.6))
L = comp.sum(1)
for cl in ['neg0', 'neg1', 'pos']:
    mk = cls_arr == cl
    ax.scatter(L[mk], ypos[cl] + np.linspace(-.12, .12, mk.sum()), color=colors[cl], s=45, label=labs[cl])
ax.axvline(0, color='k', lw=1.5)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels([labs[c] for c in ['neg0', 'neg1', 'pos']])
ax.set_xlabel('logit'); ax.set_title(f'Toy logit ladder — all 24 (input,output) decisions '
                                     f'({100*best["acc"]:.0f}% correct at threshold 0)')
ax.legend(fontsize=8, loc='lower right'); fig.tight_layout()
fig.savefig(os.path.join(RESULTS, 'fig_toy_ladder.png'), dpi=110)

# ---------- (3) logit ladder decomposition (ablations) ----------
variants = [("full", [0, 1, 2, 3]), ("no interference", [0, 1, 2]), ("no inhibition", [0, 2, 3])]
fig, axes = plt.subplots(1, 3, figsize=(16, 3.8), sharex=True, sharey=True)
print("\nablation accuracy (threshold 0):")
for ax, (ttl, c) in zip(axes, variants):
    L = comp[:, c].sum(1)
    for cl in ['neg0', 'neg1', 'pos']:
        mk = cls_arr == cl
        ax.scatter(L[mk], ypos[cl] + np.linspace(-.12, .12, mk.sum()), color=colors[cl], s=40)
    ax.axvline(0, color='k', lw=1.5)
    pos = L[cls_arr == 'pos']; neg = L[cls_arr != 'pos']
    acc = ((pos > 0).sum() + (neg <= 0).sum())/len(L)
    ax.set_title(f"{ttl}\nacc {100*acc:.0f}%", fontsize=11); ax.set_xlabel('logit')
    print(f"  {ttl:16s} acc {100*acc:.0f}%")
axes[0].set_yticks([0, 1, 2]); axes[0].set_yticklabels([labs[c] for c in ['neg0', 'neg1', 'pos']])
fig.suptitle('Toy logit ladder decomposition: remove interference / inhibition', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_toy_ladder_decomp.png'), dpi=110)
print("\nfigures: fig_toy_decomp, fig_toy_ladder, fig_toy_ladder_decomp")
