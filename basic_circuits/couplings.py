import numpy as np, itertools, os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Two follow-ups to the interference ablations in factorize.py / factorize.md:
#   Q1: the "no interference" ladder misclassifies ~8.5% of positives only because
#       the threshold sits at 0. Is the no-interference logit actually separable if
#       we just move the decision line? (Answer: yes, AUC ~ 1.0.)
#   Q2: going no-interference -> full, which couplings X[t,(i,j)] shift the most,
#       and is there structure / an embedding-geometry reason?
# ----------------------------------------------------------------------------
DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(DIR, "results"); os.makedirs(RESULTS, exist_ok=True)

m = 32
pairs = list(itertools.combinations(range(m), 2)); T = len(pairs)
pair_idx = np.array(pairs); pair_to_t = {tuple(p): i for i, p in enumerate(pairs)}
d = np.load(os.path.join(DIR, "pullback_seed2.npz")); Qf, diag, bo = d['Qf'], d['diag'], d['bo']
E = np.load(os.path.join(DIR, "uand_seed2.npz"))['E']           # (16, 32) embedding
G = E.T @ E                                                     # feature gram (32, 32)
iu = np.triu_indices(m, 1)
C = 2*Qf[:, iu[0], iu[1]]; X = C.copy(); X[np.arange(T), np.arange(T)] = 0.0   # interference matrix

# ---- sample 3-hot inputs, recording (no-interference logit, full logit) per case ----
rng = np.random.default_rng(0); rows = {'pos': [], 'neg1': [], 'neg0': []}
def rec(S3, t_):
    a, b = pair_idx[t_]; base = bo[t_] + diag[t_, S3].sum(); interf = 0.0
    for (i, j) in itertools.combinations(sorted(S3), 2):
        if (i, j) == tuple(sorted((a, b))): base += 2*Qf[t_, i, j]   # signal stays in base
        else: interf += 2*Qf[t_, i, j]
    return base, base + interf
for _ in range(6000):
    S3 = rng.choice(m, 3, replace=False); t_ = rng.integers(T); k = len(set(pair_idx[t_]) & set(S3))
    rows['pos' if k == 2 else 'neg1' if k == 1 else 'neg0'].append(rec(S3, t_))
for _ in range(3000):
    S3 = rng.choice(m, 3, replace=False); aa, bb = sorted(rng.choice(S3, 2, replace=False))
    rows['pos'].append(rec(S3, pair_to_t[(aa, bb)]))
R = {k: np.array(v) for k, v in rows.items()}                  # each (n,2): [no_interf, full]
pos = R['pos']; negs = np.vstack([R['neg0'], R['neg1']])
colors = {'neg0': '#888', 'neg1': '#d62728', 'pos': '#2ca02c'}
labels = {'neg0': 'neg, shares 0 idx', 'neg1': 'neg, shares 1 idx (hardest)', 'pos': 'positive (AND true)'}

# ============================================================================
# Q1 — does the no-interference logit separate if we just move the threshold?
# ============================================================================
def auc(pl, nl):
    allv = np.concatenate([pl, nl]); lab = np.concatenate([np.ones(len(pl)), np.zeros(len(nl))])
    order = np.argsort(allv); ranks = np.empty(len(allv)); ranks[order] = np.arange(len(allv))
    return (ranks[lab == 1].sum() - len(pl)*(len(pl)-1)/2) / (len(pl)*len(nl))

print("=== Q1: no-interference logit, sweep the decision threshold ===")
report = {}
for col, name in [(0, 'no_interf'), (1, 'full')]:
    pl, nl = pos[:, col], negs[:, col]
    cand = np.unique(np.concatenate([pl, nl]))
    bacc, bthr = max((0.5*((pl > t).mean() + (nl <= t).mean()), t) for t in cand)
    thr0 = nl.max()                                            # threshold giving FPR = 0
    report[name] = dict(tpr0=(pl > 0).mean(), fpr0=(nl > 0).mean(), bthr=bthr,
                        tpr_b=(pl > bthr).mean(), fpr_b=(nl > bthr).mean(), bacc=bacc,
                        thr_fpr0=thr0, tpr_fpr0=(pl > thr0).mean(), auc=auc(pl, nl))
    r = report[name]
    print(f"  [{name:9s}] thr=0: TPR {100*r['tpr0']:5.1f}% FPR {100*r['fpr0']:4.2f}%  | "
          f"best-bal thr={r['bthr']:+6.1f}: TPR {100*r['tpr_b']:5.1f}% FPR {100*r['fpr_b']:4.2f}% "
          f"(bAcc {100*r['bacc']:.2f}%)  | FPR=0 thr={r['thr_fpr0']:+5.1f}: TPR {100*r['tpr_fpr0']:5.1f}%  | "
          f"AUC {r['auc']:.4f}")

fig, ax = plt.subplots(figsize=(9.5, 5))
bins = np.linspace(-140, 60, 110)
for key in ['neg0', 'neg1', 'pos']:
    ax.hist(R[key][:, 0], bins=bins, alpha=.6, color=colors[key], label=labels[key])
r = report['no_interf']
ax.axvline(0, color='k', lw=2, label='threshold = 0 (model default)')
ax.axvline(r['bthr'], color='b', lw=2, ls='--', label=f"threshold = {r['bthr']:.0f} (best)")
ax.set_xlabel('logit (NO interference)'); ax.set_ylabel('count')
ax.set_title('Q1: the no-interference logit is already separable (AUC %.3f) —\n'
             'moving the line 0 -> %.0f lifts TPR %.1f%% -> %.1f%% (FPR %.2f%%); interference adds no separating power'
             % (r['auc'], r['bthr'], 100*r['tpr0'], 100*r['tpr_b'], 100*r['fpr_b']), fontsize=11)
ax.legend(loc='upper left', fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig5_threshold_shift.png'), dpi=110)

# ============================================================================
# Q2 — which couplings shift most (no_interf -> full) and why?
# ============================================================================
print("\n=== Q2: per-sample interference shift (full - no_interf) ===")
for k in ['pos', 'neg1', 'neg0']:
    sh = R[k][:, 1] - R[k][:, 0]
    print(f"  {k:5s}: mean shift {sh.mean():+.2f}  std {sh.std():.2f}")

# every (target t, off-target pair p) coupling, classified by index-overlap with the target
s1v, s0v, s1g, s0g = [], [], [], []
for t in range(T):
    a, b = pair_idx[t]
    for pc, (i, j) in enumerate(pairs):
        if pc == t: continue
        ov = len({i, j} & {a, b})
        (s1v if ov == 1 else s0v).append(X[t, pc])
        (s1g if ov == 1 else s0g).append(G[i, j])
s1v, s0v, s1g, s0g = map(np.array, (s1v, s0v, s1g, s0g))
print("\n=== Q2: coupling X[t,(i,j)] grouped by target-overlap ===")
print(f"  shares-1 (pair shares one target index): mean {s1v.mean():+.3f} std {s1v.std():.2f} (n={len(s1v)})")
print(f"  shares-0 (pair disjoint from target):    mean {s0v.mean():+.3f} std {s0v.std():.2f} (n={len(s0v)})")
print(f"  corr(coupling, embedding gram G[i,j]): shares-1 {np.corrcoef(s1v, s1g)[0,1]:+.3f}  "
      f"shares-0 {np.corrcoef(s0v, s0g)[0,1]:+.3f}")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# (a) systematic mean coupling by overlap, with SEM error bars (tiny vs std, but cleanly signed)
means = [s0v.mean(), s1v.mean()]; sems = [s0v.std()/np.sqrt(len(s0v)), s1v.std()/np.sqrt(len(s1v))]
axes[0].bar([0, 1], means, yerr=sems, color=['#1f77b4', '#ff7f0e'], capsize=6, width=.6)
axes[0].axhline(0, color='k', lw=1)
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['shares 0 idx\n(disjoint)', 'shares 1 idx\n(touches target)'])
axes[0].set_ylabel('mean coupling 2·Qf[t,(i,j)]')
axes[0].set_title('(a) systematic bias by overlap\n(per-coupling std ≈ 10; the MEAN is what accumulates)')
for x, mn in zip([0, 1], means): axes[0].text(x, mn + (.06 if mn > 0 else -.10), f'{mn:+.2f}', ha='center', fontsize=11)
# (b) per-sample shift (full - no_interf) by population: which samples move, and how far
sb = np.linspace(-80, 80, 90)
for key in ['neg0', 'neg1', 'pos']:
    sh = R[key][:, 1] - R[key][:, 0]
    axes[1].hist(sh, bins=sb, alpha=.6, color=colors[key], density=True,
                 label=f'{labels[key]}  ({sh.mean():+.2f})')
    axes[1].axvline(sh.mean(), color=colors[key], ls='--', lw=1.5)
axes[1].axvline(0, color='k', lw=1)
axes[1].set_xlabel('interference shift  (full − no-interf logit)'); axes[1].set_ylabel('density')
axes[1].set_title('(b) per-sample shift: positives +1.5 (2 shares-1),\nneg0 −2.0 (3 shares-0) — signs follow (a)')
axes[1].legend(fontsize=8)
# (c) the reason: shares-0 couplings ≈ −(embedding overlap); shares-1 flatter
sub = np.random.default_rng(1)
for v, g, c, lab in [(s0v, s0g, '#1f77b4', 'shares 0'), (s1v, s1g, '#ff7f0e', 'shares 1')]:
    idx = sub.choice(len(v), 2500, replace=False)
    axes[2].scatter(g[idx], v[idx], s=5, alpha=.25, color=c, label=f'{lab}  (corr {np.corrcoef(v, g)[0,1]:+.2f})')
axes[2].axhline(0, color='k', lw=.8); axes[2].axvline(0, color='k', lw=.8)
axes[2].set_xlabel('embedding gram overlap  G[i,j] = (Eᵀ E)[i,j]'); axes[2].set_ylabel('coupling 2·Qf[t,(i,j)]')
axes[2].set_title('(c) why: shares-0 ≈ −(embedding overlap),\ninherited crosstalk; shares-1 are flatter')
axes[2].legend(fontsize=9, markerscale=3)
fig.suptitle('Q2: which couplings shift most (no-interf -> full), and why', fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(RESULTS, 'fig6_coupling_structure.png'), dpi=110)
print("\nfigures saved: fig5_threshold_shift.png, fig6_coupling_structure.png")
