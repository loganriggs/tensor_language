# Amortize sparse interaction support searches

Between native spectral updates80and100 only10of4096edges changed, while the
projected gradient remained0.0112. The first schedule spends a full664128edge
search after every update. Test20fixed-support Riemannian conjugate-gradient
updates per exact full search, keeping all1152inputs and all outputs.

For $F(Q)=\max_E f_E(Q)$, an epoch starts with $f_E(Q_0)=F(Q_0)$.
Armijo ascent gives $f_E(Q_t)\ge f_E(Q_0)$; final reselection gives
$F(Q_t)\ge f_E(Q_t)\ge F(Q_0)$. The full objective need not be monotone between
inner updates; record true scores only at exact reselection boundaries.
Refresh support and gradient before every convergence verdict. Use relative
projected-gradient<=1e-6 and positive normalized support gap>1e-12, unchanged.
Restart conjugate directions at each refresh. This is local nonconvex search.

Native comparison planned after interpreting the current run: continue both
frozen final frames, up to600seconds/2000updates each,4096edges. Compare gain,
gradient and updates per second with each old arm; preserve changed-start caveat
for throughput and objective comparisons. No claim of a matched from-scratch
benchmark. Predict both starts converge, bestcapture improves>=10% over oldbest,
and cross-start fullfunction cosine>=.95. Instruments<=1e-8, monotone epoch
scores, fresh final support replay. No data fitting; nativebehavior validation
remains separate. Code/control first; no job implied queued by this note.

If time expires without convergence, retain checkpoints and diagnose actual
gradient progress before any further budget. A fixed orthogonal graph remains
one representation assumption, not a generic arithmetic lower bound. The circuit
target is stable shared readers and interactions suitable for extraction and
joint edits; numerical stationarity alone does not identify them semantically.
