import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m, d0, n_hid = 32, 16, 64
pairs = list(itertools.combinations(range(m), 2)); T = len(pairs)
pair_idx = np.array(pairs)
d = np.load(os.path.join(DIR, "pullback_seed2.npz"))
Qf, sig, diag, bo = d['Qf'], d['sig'], d['diag'], d['bo']
w = np.load(os.path.join(DIR, "uand_seed2.npz"))
W1, W2, Wo, E = w['W1'], w['W2'], w['Wo'], w['E']

plt.rcParams.update({'font.size': 13})

# ---------- FIG 1: anatomy of one target's quadratic form ----------
t = 100; a, b = pair_idx[t]
M = Qf[t].copy()
lim = np.abs(M).max()
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(M, cmap='RdBu_r', vmin=-lim, vmax=lim)
ax.annotate(f'SIGNAL\nQ[{a},{b}] (+{2*M[a,b]:.0f} as 2q)', xy=(b, a), xytext=(b+7, a-4),
            arrowprops=dict(arrowstyle='->', lw=2), fontsize=12, fontweight='bold')
ax.annotate('diagonal = linear terms\n(x_i² = x_i): inhibition',
            xy=(20, 20), xytext=(2, 29.5),
            arrowprops=dict(arrowstyle='->', lw=2), fontsize=12)
ax.annotate(f'own diag ≈ −4', xy=(a, a), xytext=(a+9, a+6),
            arrowprops=dict(arrowstyle='->', lw=1.5), fontsize=11)
ax.set_title(f'Feature-space quadratic form Qf for target t = AND(x{a}, x{b})\n'
             f'logit(x) = xᵀ Qf x + bias,  x ∈ {{0,1}}³²-sparse', fontsize=13)
ax.set_xlabel('feature j'); ax.set_ylabel('feature i')
fig.colorbar(im, shrink=0.8)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig1_qform.png'), dpi=110)

# ---------- FIG 2: the 3-part decomposition ----------
S = np.zeros_like(M); S[a, b] = M[a, b]; S[b, a] = M[b, a]
Dg = np.diag(np.diag(M))
I_ = M - S - Dg
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
for ax_, mat, title in zip(axes, [M, S, Dg, I_],
        ['Qf (full)', '= signal\n(one cross-term, +38)',
         '+ diagonal inhibition\n(own ≈ −4, others ≈ −16)',
         '+ interference\n(mean 0, std ~5/entry)']):
    ax_.imshow(mat, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax_.set_title(title, fontsize=12); ax_.set_xticks([]); ax_.set_yticks([])
fig.suptitle(f'Decomposition of Qf for AND(x{a}, x{b})  —  seed 2', fontsize=14)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig2_decomp.png'), dpi=110)

# ---------- FIG 3: logit ladder ----------
pair_to_t = {tuple(p): i for i, p in enumerate(pairs)}
rng = np.random.default_rng(0)
def logit(S3, t_):
    aa, bb = pair_idx[t_]
    L = bo[t_] + diag[t_, S3].sum()
    for (i, j) in itertools.combinations(sorted(S3), 2):
        L += 2*Qf[t_, i, j]
    return L
Ls = {k: [] for k in ['pos', 'neg1', 'neg0']}
for _ in range(6000):
    S3 = rng.choice(m, 3, replace=False); t_ = rng.integers(T)
    k = len(set(pair_idx[t_]) & set(S3))
    Ls['pos' if k == 2 else 'neg1' if k == 1 else 'neg0'].append(logit(S3, t_))
for _ in range(3000):
    S3 = rng.choice(m, 3, replace=False)
    aa, bb = sorted(rng.choice(S3, 2, replace=False))
    Ls['pos'].append(logit(S3, pair_to_t[(aa, bb)]))
fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(-140, 45, 90)
ax.hist(Ls['neg0'], bins=bins, alpha=.6, label='neg, shares 0 idx', color='#888')
ax.hist(Ls['neg1'], bins=bins, alpha=.6, label='neg, shares 1 idx (hardest)', color='#d62728')
ax.hist(Ls['pos'],  bins=bins, alpha=.6, label='positive (AND true)', color='#2ca02c')
ax.axvline(0, color='k', lw=1)
zz = np.linspace(-140, 45, 300)
ax2 = ax.twinx(); ax2.plot(zz, 1/(1+np.exp(-zz)), 'b--', lw=2, label='sigmoid')
ax2.set_ylabel('sigmoid(logit)', color='b')
ax.set_xlabel('logit'); ax.set_ylabel('count')
ax.set_title('Logit ladder: diagonal inhibition separates cases;\nsigmoid saturates away ±20 interference noise')
ax.legend(loc='upper left')
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig3_ladder.png'), dpi=110)

# ---------- FIG 3b: logit ladder ablations (drop interference / drop inhibition) ----------
# split each sample's logit into (bias, inhibition, signal, interference) so we can
# re-add subsets and see which mechanism actually separates the case populations.
def logit_parts(S3, t_):
    tgt = tuple(sorted(pair_idx[t_]))
    inhib = diag[t_, S3].sum()
    sig = interf = 0.0
    for (i, j) in itertools.combinations(sorted(S3), 2):
        v = 2*Qf[t_, i, j]
        if (i, j) == tgt: sig += v
        else: interf += v
    return bo[t_], inhib, sig, interf            # bias, inhibition, signal, interference

rng = np.random.default_rng(0)
parts = {k: [] for k in ['pos', 'neg1', 'neg0']}
for _ in range(6000):
    S3 = rng.choice(m, 3, replace=False); t_ = rng.integers(T)
    k = len(set(pair_idx[t_]) & set(S3))
    parts['pos' if k == 2 else 'neg1' if k == 1 else 'neg0'].append(logit_parts(S3, t_))
for _ in range(3000):
    S3 = rng.choice(m, 3, replace=False)
    aa, bb = sorted(rng.choice(S3, 2, replace=False))
    parts['pos'].append(logit_parts(S3, pair_to_t[(aa, bb)]))
parts = {k: np.array(v) for k, v in parts.items()}   # each (n,4)

variants = [                                          # which components to sum
    ("full logit\n(bias+inhibition+signal+interference)", [0, 1, 2, 3]),
    ("no interference\n(bias+inhibition+signal)",         [0, 1, 2]),
    ("no inhibition\n(bias+signal+interference)",         [0, 2, 3]),
]
colors = {'neg0': '#888', 'neg1': '#d62728', 'pos': '#2ca02c'}
labels = {'neg0': 'neg, shares 0 idx', 'neg1': 'neg, shares 1 idx (hardest)', 'pos': 'positive (AND true)'}
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
bins = np.linspace(-140, 60, 100)
for ax, (title, cols) in zip(axes, variants):
    val = {k: parts[k][:, cols].sum(1) for k in parts}
    for key in ['neg0', 'neg1', 'pos']:
        ax.hist(val[key], bins=bins, alpha=.6, color=colors[key], label=labels[key])
    ax.axvline(0, color='k', lw=1)
    fp = (np.concatenate([val['neg0'], val['neg1']]) > 0).mean()   # negatives leaking past 0
    tp = (val['pos'] > 0).mean()                                   # positives kept above 0
    ax.set_title(title, fontsize=11); ax.set_xlabel('logit')
    ax.text(0.02, 0.97, f"neg > 0: {100*fp:.1f}%\npos > 0: {100*tp:.1f}%", transform=ax.transAxes,
            va='top', fontsize=10, bbox=dict(boxstyle='round', fc='white', alpha=.85))
axes[0].set_ylabel('count'); axes[0].legend(loc='upper left', fontsize=9)
fig.suptitle('Logit ladder ablations: inhibition gates the negatives (drop it -> FPR 0->41%); '
             'interference only nudges positives (drop it -> TPR 99.9->91.5%)', fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig3b_ladder_ablation.png'), dpi=110)
for title, cols in variants:
    fp = (np.concatenate([parts['neg0'][:, cols].sum(1), parts['neg1'][:, cols].sum(1)]) > 0).mean()
    tp = (parts['pos'][:, cols].sum(1) > 0).mean()
    print(f"  [{title.splitlines()[0]:16s}] TPR (pos>0): {100*tp:5.1f}%   FPR (neg>0): {100*fp:5.2f}%")

# ---------- INTERFERENCE FACTORIZATION ----------
print("=== Interference factorization (seed 2) ===")
iu = np.triu_indices(m, 1)
C = 2*Qf[:, iu[0], iu[1]]            # (T, T) coefficient matrix; C[t,t] = signal
X = C.copy(); X[np.arange(T), np.arange(T)] = 0.0   # interference only

# exact neuron factorization check: C = Wo @ Mcross
A1, A2 = W1 @ E, W2 @ E
Mcross = A1[:, iu[0]]*A2[:, iu[1]] + A1[:, iu[1]]*A2[:, iu[0]]   # (n_hid, T)
print("C = Wo @ Mcross exact?", np.allclose(C, Wo @ Mcross, atol=1e-8))

U, s, Vt = np.linalg.svd(X, full_matrices=False)
tot = (s**2).sum()
cum = np.cumsum(s**2)/tot
pr = (s**2).sum()**2 / (s**4).sum()
print(f"rank(X) numerical: {(s > 1e-8).sum()} (bound 64+? diag-zeroing adds rank)")
print(f"participation ratio (effective rank): {pr:.1f}")
for r in [1, 2, 4, 8, 16, 32, 64]:
    print(f"  top-{r:2d} components: {100*cum[r-1]:.1f}% of interference variance")

# does removing top-r shared component kill the cross-target correlation?
for r in [1, 4, 16]:
    Xr = X - (U[:, :r]*s[:r]) @ Vt[:r]
    cors = []
    for _ in range(200):
        t1, t2 = rng.choice(T, 2, replace=False)
        msk = np.ones(T, bool); msk[[t1, t2]] = False
        cors.append(np.corrcoef(Xr[t1, msk], Xr[t2, msk])[0, 1])
    print(f"  cross-target corr after removing top-{r:2d}: {np.mean(cors):+.3f} (was +0.41); residual entry std {Xr[np.abs(Xr)>0].std():.2f} (was {X[np.abs(X)>0].std():.2f})")

# does the signal live in the same subspace? project diag(C) signal component
sig_vec_energy = []
Csig = np.zeros_like(C); Csig[np.arange(T), np.arange(T)] = C[np.arange(T), np.arange(T)]
for r in [1, 4, 16, 32, 64]:
    P = Vt[:r].T @ Vt[:r]
    frac = np.linalg.norm(Csig @ P)**2 / np.linalg.norm(Csig)**2
    sig_vec_energy.append((r, frac))
    print(f"  signal energy inside top-{r:2d} interference right-subspace: {100*frac:.1f}%")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].semilogy(np.arange(1, len(s)+1), s, 'o-', ms=3)
axes[0].axvline(64, color='r', ls='--', label='n_hidden = 64')
axes[0].set_xlabel('component'); axes[0].set_ylabel('singular value')
axes[0].set_title('Interference matrix spectrum'); axes[0].legend()
axes[1].plot(np.arange(1, len(s)+1), 100*cum, 'o-', ms=3)
axes[1].axvline(64, color='r', ls='--')
axes[1].set_xlabel('rank kept'); axes[1].set_ylabel('% interference variance')
axes[1].set_title(f'Cumulative variance (effective rank ≈ {pr:.0f})')
axes[1].set_xlim(0, 120)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig4_factor.png'), dpi=110)
print("figures saved")
