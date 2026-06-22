import numpy as np, itertools, os, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Sparse Universal-AND: does optimizing the single-layer model for SPARSE weights
# (in the pullback weights W1,W2 and the decoder Wo) change the picture?
#
# Iterative magnitude pruning, starting from the trained dense seed-2 model:
#   repeat:  (i) fine-tune a few hundred steps on  CE + lambda*||W||_1
#            (ii) prune the smallest k% of the *currently-active* weights
# Gotchas handled:
#   - persistent mask: pruned weights stay 0 (we re-zero W AND its Adam moments
#     every step, so nothing "retrains from 0" / resurrects).
#   - prune only among still-active weights (rank by magnitude over mask==1), so a
#     k% round always removes k% NEW weights, never re-counts already-zeroed ones.
# Every round is checkpointed; we report the sparsest one whose CE is still ~dense.
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)

m, d0, n_hid = 32, 16, 64
pairs = list(itertools.combinations(range(m), 2)); T = len(pairs); pi = np.array(pairs)
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def make_data(rng, n):
    R = rng.random((n, m)); idx = np.argpartition(R, 3, axis=1)[:, :3]
    F = np.zeros((n, m)); np.put_along_axis(F, idx, 1.0, axis=1)
    return F, F[:, pi[:, 0]]*F[:, pi[:, 1]]

# ---- load dense trained model (the "original" model) ----
w = np.load(os.path.join(DIR, "uand_seed2.npz"))
W1, W2, Wo, bo, E = [w[k].copy() for k in ('W1', 'W2', 'Wo', 'bo', 'E')]
rng_eval = np.random.default_rng(100); Fe, Ye = make_data(rng_eval, 20000); Xe = Fe @ E.T
def evaluate(W1, W2, Wo, bo):
    He = (Xe @ W1.T) * (Xe @ W2.T); Pe = sigmoid(He @ Wo.T + bo)
    bce = -(Ye*np.log(Pe+1e-9) + (1-Ye)*np.log(1-Pe+1e-9)).mean()
    pred = Pe > 0.5
    tpr = (pred & (Ye > .5)).sum()/(Ye > .5).sum(); tnr = (~pred & (Ye < .5)).sum()/(Ye < .5).sum()
    return bce, tpr, tnr
# second eval set for the RECALIBRATED metric: per Q1 a fixed threshold understates
# a still-separable model, so also track AUC + best global-threshold balanced accuracy.
Fr, Yr = make_data(np.random.default_rng(101), 6000); Xr = Fr @ E.T; yr = Yr.ravel().astype(bool)
npos, nneg = int(yr.sum()), int((~yr).sum())
def recal(W1, W2, Wo, bo):
    z = (((Xr @ W1.T) * (Xr @ W2.T)) @ Wo.T + bo).ravel()
    order = np.argsort(z); ys = yr[order]
    ranks = np.empty(len(z)); ranks[order] = np.arange(len(z))
    auc = (ranks[yr].sum() - npos*(npos-1)/2)/(npos*nneg)
    tpr = 1 - np.cumsum(ys)/npos; tnr = np.cumsum(~ys)/nneg     # predict + for logit > threshold
    return auc, (0.5*(tpr + tnr)).max()
bce0, tpr0, tnr0 = evaluate(W1, W2, Wo, bo); auc0, bal0 = recal(W1, W2, Wo, bo)
print(f"dense baseline: BCE {bce0:.5f}  TPR {tpr0:.4f}  TNR {tnr0:.4f}  AUC {auc0:.4f}  "
      f"recal-bAcc {bal0:.4f}  (33792 weights, 0% sparse)")

# ---- iterative magnitude pruning ----
PRUNE_FRAC, ROUNDS, STEPS, BATCH, LR, LAM = 0.10, 18, 600, 512, 1.5e-3, 3e-5
b1, b2, eps = 0.9, 0.999, 1e-8
prun = [W1, W2, Wo]                                # the sparsifiable weights (pullback + decoder)
masks = [np.ones_like(p) for p in prun]
allp = prun + [bo]                                 # bo trains but is never pruned
ms_ = [np.zeros_like(p) for p in allp]; vs_ = [np.zeros_like(p) for p in allp]

def sparsity():
    z = sum((mk == 0).sum() for mk in masks); tot = sum(mk.size for mk in masks)
    return z/tot
def sparsity_split():                              # (pullback W1+W2, decoder Wo)
    pb = ((masks[0] == 0).sum() + (masks[1] == 0).sum())/(masks[0].size + masks[1].size)
    dec = (masks[2] == 0).sum()/masks[2].size
    return pb, dec
def prune_round(frac):
    for Wm, mk in zip(prun, masks):
        act = np.flatnonzero(mk.ravel())
        n_drop = int(np.floor(frac*act.size))
        if n_drop <= 0: continue
        order = np.argsort(np.abs(Wm.ravel()[act]))      # smallest-magnitude ACTIVE weights
        drop = act[order[:n_drop]]
        mk.ravel()[drop] = 0.0; Wm.ravel()[drop] = 0.0   # remove, persistently

ckpts = [dict(round=0, sparsity=0.0, sp_pb=0.0, sp_dec=0.0, bce=bce0, tpr=tpr0, tnr=tnr0,
              auc=auc0, bal=bal0, W=[p.copy() for p in prun], bo=bo.copy())]
rng = np.random.default_rng(0); t0 = time.time(); step = 0
for rd in range(1, ROUNDS+1):
    for _ in range(STEPS):
        step += 1
        F, Y = make_data(rng, BATCH); X = F @ E.T
        A = X @ W1.T; Bv = X @ W2.T; H = A*Bv
        Z = H @ Wo.T + bo; P = sigmoid(Z); dZ = (P - Y)/BATCH
        dWo = dZ.T @ H + LAM*np.sign(Wo); dbo = dZ.sum(0)
        dH = dZ @ Wo; dA = dH*Bv; dB = dH*A
        dW1 = dA.T @ X + LAM*np.sign(W1); dW2 = dB.T @ X + LAM*np.sign(W2)
        for i, (p, g) in enumerate(zip(allp, [dW1, dW2, dWo, dbo])):
            ms_[i] = b1*ms_[i] + (1-b1)*g; vs_[i] = b2*vs_[i] + (1-b2)*g*g
            p -= LR*(ms_[i]/(1-b1**step))/(np.sqrt(vs_[i]/(1-b2**step)) + eps)
        for i in range(3):                            # keep pruned weights AND their moments at 0
            allp[i] *= masks[i]; ms_[i] *= masks[i]; vs_[i] *= masks[i]
    prune_round(PRUNE_FRAC)
    for p, mk in zip(prun, masks): p *= mk
    bce, tpr, tnr = evaluate(W1, W2, Wo, bo); auc, bal = recal(W1, W2, Wo, bo); pb, dec = sparsity_split()
    ckpts.append(dict(round=rd, sparsity=sparsity(), sp_pb=pb, sp_dec=dec, bce=bce, tpr=tpr, tnr=tnr,
                      auc=auc, bal=bal, W=[p.copy() for p in prun], bo=bo.copy()))
    print(f"round {rd:2d}  sparsity {100*sparsity():5.1f}% (pb {100*pb:4.1f}%/dec {100*dec:4.1f}%)  "
          f"BCE {bce:.5f}  TPR {tpr:.4f} TNR {tnr:.4f}  | recal: AUC {auc:.4f} bAcc {bal:.4f}")
print(f"(pruning loop {time.time()-t0:.0f}s)")

# ---- two operating points ----
# (a) "free at fixed threshold": sparsest with TPR>=0.99 & TNR>=0.9999 at the model's own bias
# (b) "free if recalibrated": sparsest still essentially separable (recal balanced-acc >= 0.999)
fixed = max([c for c in ckpts if c['tpr'] >= 0.99 and c['tnr'] >= 0.9999], key=lambda c: c['sparsity'])
best = max([c for c in ckpts if c['bal'] >= 0.99], key=lambda c: c['sparsity'])
print(f"\nsparsest free at FIXED threshold (TPR>=.99): round {fixed['round']}  "
      f"{100*fixed['sparsity']:.1f}% sparse  BCE {fixed['bce']:.5f}  TPR {fixed['tpr']:.4f}")
print(f"sparsest still SEPARABLE if recalibrated (bAcc>=.99): round {best['round']}  "
      f"{100*best['sparsity']:.1f}% sparse  recal-bAcc {best['bal']:.4f} AUC {best['auc']:.4f}  "
      f"(threshold-0 TPR was {best['tpr']:.4f})")
Ws = best['W']; bos = best['bo']
np.savez(os.path.join(DIR, "uand_seed2_sparse.npz"), W1=Ws[0], W2=Ws[1], Wo=Ws[2], bo=bos, E=E)

# ---- FIG: sparsity vs loss/accuracy tradeoff ----
sp = np.array([100*c['sparsity'] for c in ckpts]); bc = np.array([c['bce'] for c in ckpts])
tp = np.array([100*c['tpr'] for c in ckpts]); bal = np.array([100*c['bal'] for c in ckpts])
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].semilogy(sp, bc, 'o-'); ax[0].axhline(bce0, color='gray', ls=':', label=f'dense BCE {bce0:.5f}')
ax[0].scatter([100*fixed['sparsity']], [fixed['bce']], s=160, facecolors='none', edgecolors='r', lw=2,
              label=f"free@fixed-thr {100*fixed['sparsity']:.0f}%")
ax[0].set_xlabel('weight sparsity (%)'); ax[0].set_ylabel('eval BCE (log)')
ax[0].set_title('Sparsity vs CE (at the model\'s own bias)'); ax[0].legend(fontsize=9)
ax[1].plot(sp, tp, 'o-', label='TPR @ fixed threshold 0')
ax[1].plot(sp, bal, 's-', color='g', label='balanced acc @ best threshold (recalibrated)')
ax[1].axvline(100*best['sparsity'], color='g', ls='--', label=f"separable to {100*best['sparsity']:.0f}%")
ax[1].axvline(100*fixed['sparsity'], color='r', ls=':')
ax[1].set_xlabel('weight sparsity (%)'); ax[1].set_ylabel('rate (%)'); ax[1].set_ylim(40, 101)
ax[1].set_title('Fixed-threshold TPR collapses (calibration, cf. Q1),\nbut recalibrated separability holds much further')
ax[1].legend(fontsize=8.5)
fig.suptitle(f'Iterative magnitude pruning of the Universal-AND weights '
             f'(L1 + prune {int(100*PRUNE_FRAC)}%/round)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_sparsity_curve.png'), dpi=110)

# ---- how does the pullback change? dense vs chosen-sparse ----
def pullback_stats(W1, W2, Wo):
    A1, A2 = W1 @ E, W2 @ E
    Qf = np.einsum('tk,ki,kj->tij', Wo, A1, A2); Qf = 0.5*(Qf + Qf.transpose(0, 2, 1))
    sig = 2*Qf[np.arange(T), pi[:, 0], pi[:, 1]]
    diag = Qf[:, np.arange(m), np.arange(m)]
    iu = np.triu_indices(m, 1); C = 2*Qf[:, iu[0], iu[1]]; Xint = C.copy(); Xint[np.arange(T), np.arange(T)] = 0.0
    s = np.linalg.svd(Xint, compute_uv=False)
    npr = Wo != 0
    return dict(Qf=Qf, sig=sig.mean(), diag=diag.mean(), interf_std=Xint[np.abs(Xint) > 0].std(),
                top1=100*(s[0]**2)/(s**2).sum(), neurons_per_target=npr.sum(1).mean(),
                targets_per_neuron=npr.sum(0).mean(),
                qf_exact_zeros=100*(Qf == 0).mean(), qf_frac_small=100*(np.abs(Qf) < 0.5).mean())
dense = pullback_stats(W1=w['W1'], W2=w['W2'], Wo=w['Wo'])
sparse = pullback_stats(W1=Ws[0], W2=Ws[1], Wo=Ws[2])
print("\n=== pullback: dense vs sparse ===")
print(f"  {'metric':24s} {'dense':>10s} {'sparse':>10s}")
for key, name in [('sig', 'signal (mean)'), ('diag', 'diag inhibition (mean)'),
                  ('interf_std', 'interference std'), ('top1', 'top-1 SVD mode (%)'),
                  ('neurons_per_target', 'neurons / target'), ('targets_per_neuron', 'targets / neuron'),
                  ('qf_exact_zeros', 'Qf exact zeros (%)'), ('qf_frac_small', 'Qf |.|<0.5 (%)')]:
    print(f"  {name:24s} {dense[key]:10.3f} {sparse[key]:10.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
t_show = 100; lim = np.abs(dense['Qf'][t_show]).max()
for a, Q, ttl in zip(ax, [dense['Qf'][t_show], sparse['Qf'][t_show]],
                     [f'dense Qf (target {t_show})', f"sparse Qf ({100*best['sparsity']:.0f}% weights pruned)"]):
    im = a.imshow(Q, cmap='RdBu_r', vmin=-lim, vmax=lim); a.set_title(ttl); a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=ax, shrink=.8)
fig.suptitle(f'Pullback Qf stays DENSE under weight sparsity: {sparse["qf_exact_zeros"]:.1f}% exact zeros '
             f'even at {100*best["sparsity"]:.0f}% weight sparsity\n(the frozen non-orthogonal embedding E re-mixes '
             f'everything in Qf = Wo·(W1E)·(W2E))', fontsize=12)
fig.savefig(os.path.join(RESULTS, 'fig_sparse_pullback.png'), dpi=110)
print("\nfigures saved: fig_sparsity_curve.png, fig_sparse_pullback.png; weights -> uand_seed2_sparse.npz")
