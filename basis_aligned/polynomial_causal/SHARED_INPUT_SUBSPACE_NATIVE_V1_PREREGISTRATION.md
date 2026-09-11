# Shared input features with unrestricted mixed interactions

Weight-only MLP17 plus the entire unembedding. Test a family that retains every
quadratic interaction touching a rank-r input subspace, including interactions
with its complement. This is broader than an inside-only Tucker core. It does
not impose a small output rank. No data or semantic labels enter discovery.

For symmetric token forms Q_v and P=EE^T, extract
Qstar_v=P Q_v+Q_v P-P Q_v P. Its coefficient energy is
2tr(P S)-sum_v||E^T Q_v E||², S=sum_v Q_v². An upper bound for ANY rank-r
input subspace is min(total,2sum_top_r eigenvalues(S)). The spectral subspace
is an exactly solved surrogate, not necessarily the optimizer of extraction
energy. It achieves at least half that upper bound. Retain the orthogonal
coefficient remainder explicitly; its omission is approximation, not exactness.

Run full-U and vocabulary-centered-U metrics separately. Rank list fixed:
1,4,8,11,16,32,64,128. No nonlinear optimization. Record all eigenvalues, spectral
subspaces, energies, bounds and lower bounds on reader counts for50%/90%capture.
The shared-reader program can use r input projections and r dense1152x1152
partner maps: r*(1152+1152²) numbers, r*1152 elementwise products, plus native
background. Rank11 is the largest integer below the native MLP quadratic-weight
budget3*1152*4608. This representation has redundant coefficients; the budget
comparison is specific to this implementation, not a universal complexity bound.

Predictions, scored as written:

- A: planted/operator controls pass; native full/centered total energies and
  rank1 upper bounds match prior receipts within1e-8; subspaces orthonormal1e-10;
  bounds and half-bound guarantee hold1e-9; native rank4 source-removal replay
  on32formal Gaussian probes within1e-10. No text/model body forwards.
- B: rank11 centered upper bound below15%, ruling out50%coefficient capture by
  this small shared-input family regardless of nonlinear optimizer.
- C: rank11 centered spectral capture at least80%of its upper bound, limiting
  the room for improving this surrogate at the same rank. A miss is not a
  convergence failure or evidence against the broader family.

Native S is computed implicitly from exact Left/Right/Down/U weights in FP64.
Dense50,304x1152x1152 tensor is never built. Cache small S/eigenbasis matrices in
/dev/shm; one managed lane1 job with a1200-second alarm. CPU controls and dryrun
before hash-bound enqueue. Data fitting remains deferred.

Red-team: distinguish a proved energy ceiling for all rank-r touch families from
failure of one spectral frame; centered and full metrics must not be interchanged.
Low global weight energy does not imply weak behavioral effects. Multiple local
families, more readers, conditional structure, cheaper structured partners and
coupled upstream variables are outside a low-rank global-family rejection.

The trace extremum is the Ky Fan variational principle; our mixed-interaction
projection and bound are derived for this tensor, not a claim of semantic recovery.
[Primary mathematical source](https://pmc.ncbi.nlm.nih.gov/articles/PMC8589322/).
