import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Anatomy of the SPARSE-Qf pullback (the config-B "embed+Qf-L1" model, taken at
# its chosen sparsity BEFORE the L1-free recovery re-densifies Qf).
#   (1) break Qf down into signal / diagonal inhibition / interference (fig2-style)
#   (2) interference effective rank vs the dense model (was ~5.5, top-1 41.5%)
#   (3) component-ablation logit ladders (full / no-interference / no-inhibition)
# The Qf-L1 model is reproduced here deterministically (same seed/schedule as
# factorized_sparsity.py config B, stopped at the chosen 34%-sparse round 4).
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
m, d0, n_hid = 32, 16, 64
pairs = list(itertools.combinations(range(m), 2)); T = len(pairs); pi = np.array(pairs)
pair_to_t = {tuple(p): i for i, p in enumerate(pairs)}
b1, b2, eps = 0.9, 0.999, 1e-8
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def make_data(rng, n):
    R = rng.random((n, m)); idx = np.argpartition(R, 3, axis=1)[:, :3]
    F = np.zeros((n, m)); np.put_along_axis(F, idx, 1.0, axis=1)
    return F, F[:, pi[:, 0]]*F[:, pi[:, 1]]
def pullback(W1, W2, Wo, E):
    A1, A2 = W1 @ E, W2 @ E; Qf = np.einsum('tk,ki,kj->tij', Wo, A1, A2); return 0.5*(Qf + Qf.transpose(0, 2, 1))
def qf_grads(W1, W2, Wo, E, lam):
    A1, A2 = W1 @ E, W2 @ E; S = np.sign(pullback(W1, W2, Wo, E))
    gWo = np.einsum('ki,tij,kj->tk', A1, S, A2); gA1 = np.einsum('tk,tij,kj->ki', Wo, S, A2)
    gA2 = np.einsum('tk,tij,ki->kj', Wo, S, A1)
    return lam*(gA1 @ E.T), lam*(gA2 @ E.T), lam*gWo, lam*(W1.T @ gA1 + W2.T @ gA2)

w = np.load(os.path.join(DIR, "uand_seed2.npz"))
W1, W2, Wo, bo, E = [w[k].copy() for k in ('W1', 'W2', 'Wo', 'bo', 'E')]
Qf_dense = pullback(W1, W2, Wo, E)

# ---- reproduce config B (prune W1,W2,Wo,E + weight-L1 + Qf-L1) to round 4 (34% sparse) ----
ROUNDS, STEPS, FRAC, BATCH, LR, LAM_W, LAM_QF = 4, 500, 0.10, 512, 1.5e-3, 3e-5, 2e-5
P = {'W1': W1.copy(), 'W2': W2.copy(), 'Wo': Wo.copy(), 'bo': bo.copy(), 'E': E.copy()}
keys = ['W1', 'W2', 'Wo', 'bo', 'E']; prune = ['W1', 'W2', 'Wo', 'E']
mask = {k: np.ones_like(P[k]) for k in prune}; st = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in keys}
rng = np.random.default_rng(0); g = 0
SPARSE_NPZ = os.path.join(DIR, "uand_seed2_sparseQf.npz")
if os.path.exists(SPARSE_NPZ):           # reuse the saved checkpoint; skip retraining
    d = np.load(SPARSE_NPZ)
    for k in keys: P[k][:] = d[k]
    ROUNDS = 0
for rd in range(ROUNDS):
    for _ in range(STEPS):
        g += 1; F, Y = make_data(rng, BATCH); X = F @ P['E'].T
        A = X @ P['W1'].T; Bv = X @ P['W2'].T; H = A*Bv; dZ = (sigmoid(H @ P['Wo'].T + P['bo']) - Y)/BATCH
        gr = {'Wo': dZ.T @ H + LAM_W*np.sign(P['Wo']), 'bo': dZ.sum(0)}
        dH = dZ @ P['Wo']; dA = dH*Bv; dB = dH*A
        gr['W1'] = dA.T @ X + LAM_W*np.sign(P['W1']); gr['W2'] = dB.T @ X + LAM_W*np.sign(P['W2'])
        gr['E'] = (dA @ P['W1'] + dB @ P['W2']).T @ F
        q1, q2, qo, qe = qf_grads(P['W1'], P['W2'], P['Wo'], P['E'], LAM_QF)
        gr['W1'] += q1; gr['W2'] += q2; gr['Wo'] += qo; gr['E'] += qe
        for k in keys:
            mm, vv = st[k]; mm[:] = b1*mm + (1-b1)*gr[k]; vv[:] = b2*vv + (1-b2)*gr[k]*gr[k]
            P[k] -= LR*(mm/(1-b1**g))/(np.sqrt(vv/(1-b2**g)) + eps)
        for k in prune:
            P[k] *= mask[k]; st[k][0] *= mask[k]; st[k][1] *= mask[k]
    for k in prune:
        act = np.flatnonzero(mask[k].ravel()); nd = int(np.floor(FRAC*act.size))
        drop = act[np.argsort(np.abs(P[k].ravel()[act]))[:nd]]; mask[k].ravel()[drop] = 0; P[k].ravel()[drop] = 0
sp = 1 - sum(np.count_nonzero(P[k]) for k in prune)/sum(P[k].size for k in prune)
if not os.path.exists(SPARSE_NPZ): np.savez(SPARSE_NPZ, **{k: P[k] for k in keys})
Qf = pullback(P['W1'], P['W2'], P['Wo'], P['E']); bos = P['bo']
print(f"reproduced sparse-Qf model: {100*sp:.0f}% weight-sparse, Qf |.|<0.5 = {100*(np.abs(Qf)<0.5).mean():.0f}% "
      f"(dense {100*(np.abs(Qf_dense)<0.5).mean():.0f}%)")

# ============ (1) decomposition: dense (top) vs sparse-Qf (bottom), each on its own scale ============
t = 100; a, b = pi[t]
def decomp(Q):
    M = Q[t]; Sg = np.zeros_like(M); Sg[a, b] = M[a, b]; Sg[b, a] = M[b, a]
    Dg = np.diag(np.diag(M)); return M, Sg, Dg, M - Sg - Dg
titles = ['Qf (full)', '= signal (the AND-ing)', '+ diagonal inhibition', '+ interference (off-diag)']
fig, axes = plt.subplots(2, 4, figsize=(15, 8))
for row, (Q, name) in enumerate([(Qf_dense, 'DENSE'), (Qf, 'SPARSE-Qf')]):
    mats = decomp(Q); lim = np.abs(mats[0]).max()
    for c, (a_, mat) in enumerate(zip(axes[row], mats)):
        im = a_.imshow(mat, cmap='RdBu_r', vmin=-lim, vmax=lim); a_.set_xticks([]); a_.set_yticks([])
        if row == 0: a_.set_title(titles[c], fontsize=11)
    fig.colorbar(im, ax=axes[row], shrink=.7)
    axes[row, 0].set_ylabel(f'{name}\n(signal 2·Qf={2*mats[0][a,b]:.0f}, peak |Qf|={np.abs(mats[0]).max():.0f})', fontsize=10)
fig.suptitle(f'Qf breakdown for AND(x{a},x{b}): dense vs sparse-Qf model '
             f'(sparse: {100*(np.abs(Qf)<0.5).mean():.0f}% of entries ~0, ~10x smaller magnitudes)', fontsize=13)
fig.savefig(os.path.join(RESULTS, 'fig_sparseQf_decomp.png'), dpi=110, bbox_inches='tight')

# ============ (2) interference effective rank: dense vs sparse ============
iu = np.triu_indices(m, 1)
def interf_spectrum(Q):
    C = 2*Q[:, iu[0], iu[1]]; X = C.copy(); X[np.arange(T), np.arange(T)] = 0.0
    s = np.linalg.svd(X, compute_uv=False)
    pr = (s**2).sum()**2/(s**4).sum(); top1 = 100*(s[0]**2)/(s**2).sum()
    nnz = (np.abs(X) > 1e-9).mean()*100
    return s, pr, top1, nnz
sd, prd, t1d, nzd = interf_spectrum(Qf_dense)
ss, prs, t1s, nzs = interf_spectrum(Qf)
print(f"\ninterference matrix:")
print(f"  dense : participation-ratio (eff. rank) {prd:.1f}  top-1 mode {t1d:.1f}%  nonzero entries {nzd:.0f}%")
print(f"  sparse: participation-ratio (eff. rank) {prs:.1f}  top-1 mode {t1s:.1f}%  nonzero entries {nzs:.0f}%")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
ax[0].semilogy(np.arange(1, len(sd)+1), sd, '.-', label=f'dense (eff.rank {prd:.1f})')
ax[0].semilogy(np.arange(1, len(ss)+1), ss, '.-', label=f'sparse-Qf (eff.rank {prs:.1f})')
ax[0].axvline(64, color='r', ls='--', lw=1, label='n_hidden=64'); ax[0].set_xlabel('component')
ax[0].set_ylabel('singular value'); ax[0].set_title('Interference spectrum'); ax[0].legend(fontsize=9)
cd = np.cumsum(sd**2)/(sd**2).sum(); cs = np.cumsum(ss**2)/(ss**2).sum()
ax[1].plot(np.arange(1, len(cd)+1), 100*cd, '.-', label=f'dense (top-1 {t1d:.0f}%)')
ax[1].plot(np.arange(1, len(cs)+1), 100*cs, '.-', label=f'sparse-Qf (top-1 {t1s:.0f}%)')
ax[1].set_xlim(0, 120); ax[1].set_xlabel('rank kept'); ax[1].set_ylabel('% interference variance')
ax[1].set_title('Cumulative interference variance'); ax[1].legend(fontsize=9)
fig.suptitle('Interference rank: dense vs sparse-Qf model', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_sparseQf_interference_rank.png'), dpi=110)

# ============ (3) component-ablation logit ladders for the sparse model ============
diag = Qf[:, np.arange(m), np.arange(m)]
def logit_parts(S3, t_):
    tgt = tuple(sorted(pi[t_])); inh = diag[t_, S3].sum(); sig = interf = 0.0
    for (i, j) in itertools.combinations(sorted(S3), 2):
        v = 2*Qf[t_, i, j]
        if (i, j) == tgt: sig += v
        else: interf += v
    return bos[t_], inh, sig, interf
rng = np.random.default_rng(0); parts = {'pos': [], 'neg1': [], 'neg0': []}
for _ in range(6000):
    S3 = rng.choice(m, 3, replace=False); t_ = rng.integers(T); k = len(set(pi[t_]) & set(S3))
    parts['pos' if k == 2 else 'neg1' if k == 1 else 'neg0'].append(logit_parts(S3, t_))
for _ in range(3000):
    S3 = rng.choice(m, 3, replace=False); aa, bb = sorted(rng.choice(S3, 2, replace=False))
    parts['pos'].append(logit_parts(S3, pair_to_t[(aa, bb)]))
parts = {k: np.array(v) for k, v in parts.items()}
def best_thr(pos, neg):
    cand = np.unique(np.concatenate([pos, neg]))
    return max((0.5*((pos > c).mean() + (neg <= c).mean()), c) for c in cand)
colors = {'neg0': '#888', 'neg1': '#d62728', 'pos': '#2ca02c'}
labs = {'neg0': 'neg shares 0', 'neg1': 'neg shares 1', 'pos': 'positive'}
variants = [("full\n(bias+inhib+signal+interf)", [0, 1, 2, 3]),
            ("no interference\n(bias+inhib+signal)", [0, 1, 2]),
            ("no inhibition\n(bias+signal+interf)", [0, 2, 3])]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), sharey=True); bins = np.linspace(-30, 30, 90)
for ax_, (ttl, cols) in zip(axes, variants):
    val = {k: parts[k][:, cols].sum(1) for k in parts}
    for key in ['neg0', 'neg1', 'pos']:
        ax_.hist(val[key], bins=bins, alpha=.6, color=colors[key], label=labs[key])
    pos = val['pos']; neg = np.concatenate([val['neg0'], val['neg1']])
    bacc, thr = best_thr(pos, neg); tp = (pos > thr).mean(); fp = (neg > thr).mean()
    ax_.axvline(thr, color='b', ls='--', lw=2, label=f'recal thr {thr:.0f}'); ax_.axvline(0, color='k', lw=1)
    ax_.set_title(f"{ttl}\nrecal TPR {100*tp:.1f}% FPR {100*fp:.1f}% bAcc {100*bacc:.1f}%", fontsize=10)
    ax_.set_xlabel('logit')
axes[0].set_ylabel('count'); axes[0].legend(fontsize=8, loc='upper left')
fig.suptitle('Sparse-Qf model: component-ablation logit ladders (recalibrated threshold)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_sparseQf_ablation_ladder.png'), dpi=110)
# print ablation numbers
for ttl, cols in variants:
    pos = parts['pos'][:, cols].sum(1); neg = np.concatenate([parts['neg0'][:, cols].sum(1), parts['neg1'][:, cols].sum(1)])
    bacc, thr = best_thr(pos, neg)
    print(f"  [{ttl.splitlines()[0]:16s}] recal bAcc {100*bacc:.1f}%  TPR {100*(pos>thr).mean():.1f}%  FPR {100*(neg>thr).mean():.1f}%  thr {thr:.0f}")
print("figures: fig_sparseQf_decomp, fig_sparseQf_interference_rank, fig_sparseQf_ablation_ladder")
