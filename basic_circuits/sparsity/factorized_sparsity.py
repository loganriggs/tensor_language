import numpy as np, itertools, os, time, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Sparse Universal-AND, two configs compared side by side:
#   A "weights"   : iterative magnitude pruning of W1,W2,Wo  (E frozen, weight-L1)
#   B "embed+Qf"  : also prune the embedding E, and add an L1 penalty on the
#                   pullback Qf matrix itself (drive REPRESENTATION sparsity)
# For each: iterative (L1 + prune 10% of active weights/round, persistent mask),
# then an L1-FREE recovery fine-tune at the chosen sparsity to claw back CE.
# We track weight-sparsity vs CE/recal-accuracy AND the Qf-sparsity frontier
# (threshold the trained Qf and re-evaluate -- "sparse Qf at minimal CE cost").
# All accuracy uses the recalibrated (best-threshold) metric, since a fixed
# threshold understates a still-separable model (see couplings.md / Q1).
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)
SMOKE = os.environ.get("SMOKE") == "1"

m, d0, n_hid = 32, 16, 64
pairs = list(itertools.combinations(range(m), 2)); T = len(pairs); pi = np.array(pairs)
b1, b2, eps = 0.9, 0.999, 1e-8
def sigmoid(z): return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))
def make_data(rng, n):
    R = rng.random((n, m)); idx = np.argpartition(R, 3, axis=1)[:, :3]
    F = np.zeros((n, m)); np.put_along_axis(F, idx, 1.0, axis=1)
    return F, F[:, pi[:, 0]]*F[:, pi[:, 1]], np.sort(idx, axis=1)

w = np.load(os.path.join(DIR, "..", "universal_and", "uand_seed2.npz"))   # original trained model
W1d, W2d, Wod, bod, Ed = [w[k].copy() for k in ('W1', 'W2', 'Wo', 'bo', 'E')]

# eval sets: BCE/TPR/TNR on 20k; recalibrated AUC/bAcc on 6k; Qf-frontier on 2.5k
Fe, Ye, _ = make_data(np.random.default_rng(100), 20000)
Fr, Yr, ar = make_data(np.random.default_rng(101), 6000); yr = Yr.ravel().astype(bool)
Fq, Yq, aq = make_data(np.random.default_rng(102), 2500); yq = Yq.ravel().astype(bool)
nposr, nnegr = int(yr.sum()), int((~yr).sum())
nposq, nnegq = int(yq.sum()), int((~yq).sum())

def fwd(F, W1, W2, Wo, bo, E):
    X = F @ E.T; return ((X @ W1.T) * (X @ W2.T)) @ Wo.T + bo

def evaluate(W1, W2, Wo, bo, E):
    Z = fwd(Fe, W1, W2, Wo, bo, E); P = sigmoid(Z)
    bce = -(Ye*np.log(P+1e-9) + (1-Ye)*np.log(1-P+1e-9)).mean()
    pred = P > 0.5
    return bce, (pred & (Ye > .5)).sum()/(Ye > .5).sum(), (~pred & (Ye < .5)).sum()/(Ye < .5).sum()

def recal_from_logits(z, y, npos, nneg):
    order = np.argsort(z); ys = y[order]; ranks = np.empty(len(z)); ranks[order] = np.arange(len(z))
    auc = (ranks[y].sum() - npos*(npos-1)/2)/(npos*nneg)
    tpr = 1 - np.cumsum(ys)/npos; tnr = np.cumsum(~ys)/nneg
    bal = 0.5*(tpr + tnr); i = bal.argmax()
    return auc, bal[i], np.sort(z)[i]                 # auc, best balanced acc, best threshold

def recal(W1, W2, Wo, bo, E):
    return recal_from_logits(fwd(Fr, W1, W2, Wo, bo, E).ravel(), yr, nposr, nnegr)

def pullback(W1, W2, Wo, E):
    A1, A2 = W1 @ E, W2 @ E
    Qf = np.einsum('tk,ki,kj->tij', Wo, A1, A2); return 0.5*(Qf + Qf.transpose(0, 2, 1))

# evaluate logits directly from a (possibly thresholded) Qf via the 3-hot active pairs
COL = [0, 0, 0, 1, 1, 1, 2, 2, 2]; ROW = [0, 1, 2, 0, 1, 2, 0, 1, 2]
def qf_logits(Qf, bo, active):                         # active (n,3) -> (n,T)
    ii = active[:, COL]; jj = active[:, ROW]           # (n,9)
    return Qf[:, ii, jj].sum(2).T + bo

def qf_grads(W1, W2, Wo, E, lam):                       # grads of lam*sum|Qf| wrt W1,W2,Wo,E
    A1, A2 = W1 @ E, W2 @ E
    S = np.sign(pullback(W1, W2, Wo, E))                # (T,32,32) symmetric
    gWo = np.einsum('ki,tij,kj->tk', A1, S, A2)
    gA1 = np.einsum('tk,tij,kj->ki', Wo, S, A2)
    gA2 = np.einsum('tk,tij,ki->kj', Wo, S, A1)
    return lam*(gA1 @ E.T), lam*(gA2 @ E.T), lam*gWo, lam*(W1.T @ gA1 + W2.T @ gA2)

ROUNDS, STEPS, PRUNE_FRAC, BATCH, LR, LAM_W, RECOVER = (2, 30, 0.10, 512, 1.5e-3, 3e-5, 30) if SMOKE \
    else (16, 500, 0.10, 512, 1.5e-3, 3e-5, 2000)

def run_experiment(name, prune_E, lam_qf):
    W1, W2, Wo, bo, E = W1d.copy(), W2d.copy(), Wod.copy(), bod.copy(), Ed.copy()
    P = {'W1': W1, 'W2': W2, 'Wo': Wo, 'bo': bo, 'E': E}
    train = ['W1', 'W2', 'Wo', 'bo'] + (['E'] if prune_E else [])
    prune = ['W1', 'W2', 'Wo'] + (['E'] if prune_E else [])
    mask = {k: np.ones_like(P[k]) for k in prune}
    st = {k: [np.zeros_like(P[k]), np.zeros_like(P[k])] for k in train}
    def step(t, lw, lq):
        F, Y, _ = make_data(rng, BATCH); X = F @ E.T
        A = X @ W1.T; Bv = X @ W2.T; H = A*Bv; Z = H @ Wo.T + bo; dZ = (sigmoid(Z) - Y)/BATCH
        g = {'Wo': dZ.T @ H + lw*np.sign(Wo), 'bo': dZ.sum(0)}
        dH = dZ @ Wo; dA = dH*Bv; dB = dH*A
        g['W1'] = dA.T @ X + lw*np.sign(W1); g['W2'] = dB.T @ X + lw*np.sign(W2)
        if 'E' in train: g['E'] = (dA @ W1 + dB @ W2).T @ F
        if lq > 0:
            qg1, qg2, qgo, qge = qf_grads(W1, W2, Wo, E, lq)
            g['W1'] += qg1; g['W2'] += qg2; g['Wo'] += qgo
            if 'E' in train: g['E'] += qge
        for k in train:
            mm, vv = st[k]; mm[:] = b1*mm + (1-b1)*g[k]; vv[:] = b2*vv + (1-b2)*g[k]*g[k]
            P[k] -= LR*(mm/(1-b1**t))/(np.sqrt(vv/(1-b2**t)) + eps)
        for k in prune:
            P[k] *= mask[k]; st[k][0] *= mask[k]; st[k][1] *= mask[k]
    def prune_round(frac):
        for k in prune:
            act = np.flatnonzero(mask[k].ravel()); nd = int(np.floor(frac*act.size))
            if nd > 0:
                drop = act[np.argsort(np.abs(P[k].ravel()[act]))[:nd]]
                mask[k].ravel()[drop] = 0.0; P[k].ravel()[drop] = 0.0
    def sparsity():
        z = sum((mask[k] == 0).sum() for k in prune); return z/sum(mask[k].size for k in prune)
    def snap():
        bce, tpr, tnr = evaluate(W1, W2, Wo, bo, E); auc, bal, thr = recal(W1, W2, Wo, bo, E)
        return dict(sparsity=sparsity(), bce=bce, tpr=tpr, tnr=tnr, auc=auc, bal=bal, thr=thr,
                    W=[P[k].copy() for k in ('W1', 'W2', 'Wo', 'bo', 'E')], mask={k: mask[k].copy() for k in prune})

    ck = [dict(round=0, **{k: v for k, v in snap().items()})]
    rng = np.random.default_rng(0); g = 0; t0 = time.time()
    for rd in range(1, ROUNDS+1):
        for _ in range(STEPS):
            g += 1; step(g, LAM_W, lam_qf)
        prune_round(PRUNE_FRAC)
        for k in prune: P[k] *= mask[k]
        s = snap(); s['round'] = rd; ck.append(s)
        print(f"  [{name}] rd {rd:2d} sparsity {100*s['sparsity']:5.1f}% BCE {s['bce']:.5f} "
              f"TPR {s['tpr']:.3f} | recal AUC {s['auc']:.4f} bAcc {s['bal']:.4f}", flush=True)
    print(f"  [{name}] prune loop {time.time()-t0:.0f}s")

    best = max([c for c in ck if c['bal'] >= 0.99], key=lambda c: c['sparsity'])
    # ---- L1-free recovery fine-tune at the chosen sparsity (mask frozen, lam=0) ----
    for k, arr in zip(('W1', 'W2', 'Wo', 'bo', 'E'), best['W']): P[k][:] = arr
    for k in prune: mask[k][:] = best['mask'][k]
    for k in train: st[k][0][:] = 0; st[k][1][:] = 0
    for j in range(1, RECOVER+1): step(j, 0.0, 0.0)
    rec_bce, rec_tpr, rec_tnr = evaluate(W1, W2, Wo, bo, E); rec_auc, rec_bal, rec_thr = recal(W1, W2, Wo, bo, E)
    recovered = dict(sparsity=best['sparsity'], bce=rec_bce, tpr=rec_tpr, tnr=rec_tnr, auc=rec_auc, bal=rec_bal,
                     thr=rec_thr, W=[P[k].copy() for k in ('W1', 'W2', 'Wo', 'bo', 'E')])
    print(f"  [{name}] chosen {100*best['sparsity']:.0f}% sparse -> recovery: BCE {best['bce']:.5f}->{rec_bce:.5f}  "
          f"TPR {best['tpr']:.3f}->{rec_tpr:.3f}  recal-bAcc {best['bal']:.4f}->{rec_bal:.4f}")
    return dict(name=name, ck=ck, best=best, recovered=recovered)

print(f"dense baseline: BCE {evaluate(W1d,W2d,Wod,bod,Ed)[0]:.5f}")
A = run_experiment("weights", prune_E=False, lam_qf=0.0)
B = run_experiment("embed+Qf", prune_E=True, lam_qf=2e-5)

# ============================ FIGURES ============================
def arr(ex, key): return np.array([c[key] for c in ex['ck']])

# (1) weight-sparsity frontier: A vs B
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for ex, c in [(A, 'C0'), (B, 'C1')]:
    ax[0].semilogy(100*arr(ex, 'sparsity'), arr(ex, 'bce'), 'o-', color=c, label=ex['name'])
    ax[1].plot(100*arr(ex, 'sparsity'), 100*arr(ex, 'bal'), 'o-', color=c, label=ex['name'])
    ax[1].scatter([100*ex['recovered']['sparsity']], [100*ex['recovered']['bal']], marker='*', s=200,
                  color=c, edgecolors='k', zorder=5, label=f"{ex['name']} recovered")
ax[0].set_xlabel('weight sparsity (%)'); ax[0].set_ylabel('eval BCE (log)'); ax[0].set_title('Weight sparsity vs CE'); ax[0].legend(fontsize=8)
ax[1].set_xlabel('weight sparsity (%)'); ax[1].set_ylabel('recal balanced acc (%)'); ax[1].set_ylim(85, 100.5)
ax[1].set_title('Recalibrated separability vs weight sparsity'); ax[1].legend(fontsize=8)
fig.suptitle('Weight-sparsity frontier: prune W1,W2,Wo (A) vs also prune E + L1-on-Qf (B)', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_sparsity_curve.png'), dpi=110)

# (2) Qf-sparsity frontier: threshold the trained Qf and re-evaluate (dense vs A vs B)
def qf_frontier(W):
    Qf = pullback(W[0], W[1], W[2], W[4]); bo = W[3]
    mags = np.abs(Qf); thrs = np.quantile(mags, np.linspace(0, 0.98, 18))
    out = []
    for th in thrs:
        Qz = Qf*(mags >= th)
        z = qf_logits(Qz, bo, aq).ravel()
        _, bal, _ = recal_from_logits(z, yq, nposq, nnegq)
        out.append((100*(Qz == 0).mean(), 100*bal))
    return np.array(out)
fr_d = qf_frontier([W1d, W2d, Wod, bod, Ed])
fr_a = qf_frontier(A['recovered']['W'])
fr_bt = qf_frontier(B['best']['W'])          # pre-recovery: Qf-L1 active, Qf actually sparse
fr_b = qf_frontier(B['recovered']['W'])      # post-recovery: L1-free fine-tune re-densifies Qf
fig, ax = plt.subplots(figsize=(9, 5.5))
for fr, lab, c, ls in [(fr_d, f"dense (BCE {evaluate(W1d,W2d,Wod,bod,Ed)[0]:.5f})", 'gray', '-'),
                       (fr_a, f"A weights, recovered (BCE {A['recovered']['bce']:.5f})", 'C0', '-'),
                       (fr_bt, f"B Qf-L1, pre-recovery (BCE {B['best']['bce']:.5f})", 'C1', '-'),
                       (fr_b, f"B Qf-L1, recovered (BCE {B['recovered']['bce']:.5f})", 'C1', '--')]:
    ax.plot(fr[:, 0], fr[:, 1], marker='o', color=c, ls=ls, ms=4, label=lab)
ax.axhline(99, color='k', ls=':', lw=1); ax.set_xlabel('Qf sparsity (% entries zeroed)')
ax.set_ylabel('recal balanced acc (%)'); ax.set_ylim(70, 101)
ax.set_title('Qf-sparsity frontier: how much of the pullback can be zeroed,\nat minimal accuracy cost (threshold Qf, re-evaluate)')
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_qf_frontier.png'), dpi=110)

# (3) dense vs B-pre-recovery Qf heatmap (the Qf-L1-sparsified form, before recovery re-densifies it)
Qd = pullback(W1d, W2d, Wod, Ed); Qb = pullback(*[B['best']['W'][i] for i in (0, 1, 2)], B['best']['W'][4])
zd = 100*(np.abs(Qd) < 0.5).mean(); zb = 100*(np.abs(Qb) < 0.5).mean()
fig, ax = plt.subplots(1, 2, figsize=(12, 5)); lim = np.abs(Qd[100]).max()
for a, Q, ttl in zip(ax, [Qd[100], Qb[100]], [f'dense Qf (|·|<0.5: {zd:.0f}%)', f"B Qf-L1 pre-recovery (|·|<0.5: {zb:.0f}%)"]):
    im = a.imshow(Q, cmap='RdBu_r', vmin=-lim, vmax=lim); a.set_title(ttl); a.set_xticks([]); a.set_yticks([])
fig.colorbar(im, ax=ax, shrink=.8)
fig.suptitle('Pullback Qf (target 100): L1-on-Qf + embedding pruning visibly sparsifies the feature-space form\n'
             '(but the L1-free recovery fine-tune re-densifies it -- sparse-Qf and recovered-CE trade off)', fontsize=11)
fig.savefig(os.path.join(RESULTS, 'fig_sparse_pullback.png'), dpi=110)

# (4) recalibrated ladder histograms for A-best and B-best
def ladder_cases(rng, n=9000):
    S = np.array([rng.choice(m, 3, replace=False) for _ in range(n)]); ts = rng.integers(T, size=n)
    Sp = np.array([rng.choice(m, 3, replace=False) for _ in range(n//2)])
    tp = np.array([pi.tolist().index(sorted(rng.choice(s, 2, replace=False).tolist())) for s in Sp])
    S = np.vstack([S, Sp]); ts = np.concatenate([ts, tp])
    a_in = (pi[ts][:, 0:1] == S).any(1).astype(int); b_in = (pi[ts][:, 1:2] == S).any(1).astype(int)
    ov = a_in + b_in
    F = np.zeros((len(S), m)); np.put_along_axis(F, S, 1.0, axis=1)
    return F, ts, ov
Fc, tc, ov = ladder_cases(np.random.default_rng(7))
colors = {0: '#888', 1: '#d62728', 2: '#2ca02c'}; labs = {0: 'neg shares 0', 1: 'neg shares 1', 2: 'positive'}
fig, ax = plt.subplots(1, 2, figsize=(15, 5), sharey=True); bins = np.linspace(-140, 60, 110)
for a, ex in zip(ax, [A, B]):
    W = ex['recovered']['W']; thr = ex['recovered']['thr']
    Z = fwd(Fc, W[0], W[1], W[2], W[3], W[4]); lg = Z[np.arange(len(tc)), tc]
    for cl in [0, 1, 2]:
        a.hist(lg[ov == cl], bins=bins, alpha=.6, color=colors[cl], label=labs[cl])
    pos = lg[ov == 2]; neg = lg[ov < 2]
    tp = (pos > thr).mean(); fp = (neg > thr).mean()
    a.axvline(thr, color='b', ls='--', lw=2, label=f'recal thr {thr:.0f}')
    a.axvline(0, color='k', lw=1)
    a.set_xlabel('logit'); a.set_title(f"{ex['name']} @ {100*ex['recovered']['sparsity']:.0f}% sparse (recovered)\n"
                                       f"recal TPR {100*tp:.1f}% FPR {100*fp:.1f}%  bAcc {100*ex['recovered']['bal']:.1f}%")
    a.legend(fontsize=8, loc='upper left')
ax[0].set_ylabel('count')
fig.suptitle('Recalibrated logit ladders of the chosen sparse models', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig_sparsity_ladder.png'), dpi=110)

# save chosen sparse weights + print summary
Wb = B['recovered']['W']
np.savez(os.path.join(DIR, "uand_seed2_sparse.npz"), W1=Wb[0], W2=Wb[1], Wo=Wb[2], bo=Wb[3], E=Wb[4])
print("\n=== summary (recovered, L1-free fine-tuned) ===")
for ex in [A, B]:
    r = ex['recovered']
    print(f"  {ex['name']:10s}: {100*r['sparsity']:.0f}% weight-sparse  BCE {r['bce']:.5f}  "
          f"recal-bAcc {100*r['bal']:.2f}%  Qf(|·|<0.5) {100*(np.abs(pullback(r['W'][0],r['W'][1],r['W'][2],r['W'][4]))<0.5).mean():.1f}%")
print("figures saved: fig_sparsity_curve, fig_qf_frontier, fig_sparse_pullback, fig_sparsity_ladder")
