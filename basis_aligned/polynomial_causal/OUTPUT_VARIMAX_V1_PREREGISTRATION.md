# Weight-only overlapping token usage: output varimax

Registered10September before native execution. Distinct from the failed16-leaf
whole-token mean hierarchy in the MLP17 dossier: tokens may use multiple shared
quadratic summands, with signed loadings. Campaign hypotheses8–10. No token
labels or natural states in discovery; no hard clustering or nonnegativity.

Take the exact centered-unembedding coefficient tensor's optimal rank128 output
projection. Compute its SVD implicitly from the1152-dimensional output covariance
and the native4608-product Gram. Write the projection as loadings A[V,128]
times orthonormal quadratic functions H[128,input,input]. Keep the exact common
output function separately. For orthogonal R, A R and R^T H preserve the tensor.
No rank-one or low-input-rank restriction on H. The original token-specific
remainder outside the projection remains charged and is not discarded as zero.

Maximize raw varimax over R: sum over columns of the variance across tokens of
the squared loadings. Divide A by its single global RMS for numerical units;
do not normalize rows, whiten columns or introduce token-frequency weighting.
This favors concentrated signed token usage but can favor a few large-weight
rows. Report participation ratios, top4 usage and padded-row energy explicitly.
The orthogonal function-basis restriction excludes general oblique/overcomplete
dictionary structure; it is one hypothesis, not the entire25-method campaign.

Riemannian PR+ ascent uses the existing Stiefel projection/retraction, beta[0,10],
descent-related restart with sign reversed for ascent, Armijo1e-4 and25halvings.
Initial step0.01*score/slope, capped by displacement1 and step1e6; subsequent
hint1.5*accepted step. Diagnose every accepted step, print every5. Require
relative projected gradient norm*sqrt128/score<=1e-5 and five consecutive
diagnostics' relative progress<=1e-7. No time-limit-as-convergence claim.

CPU controls: dense tensor/SVD projection, orthogonal-function Gram, exact
joint rotation, finite directional derivative; independent overlapping sparse
planted factors recovered with alignment>.99 and convergence;20steps versus
10+10 exact resumed parameters. A first control reached stationarity before
five widely spaced diagnostics; diagnosing every accepted step repaired that
measurement lag before native registration, without relaxing numerical bars.

Native240seconds;900second overall alarm. All50304weight rows,128functions,
one spectral start. Save compact core mixtures/writers/rotation and resumable
state (~6.5MB), not50304×128dense loadings. Literal execution still retains
native L/R and U unless later factored; not a cheaper adopted program.

- A: implicit native total, quadratic-function orthogonality, loading Gram,
  fixed-input rotation replay, energy preservation, rotation orthogonality
  and maximum objective decrease all<=1e-10.
- B: A plus the registered relative-gradient and plateau conditions.
- C: A plus median nonzero-token factor participation falls at least25%,
  and at least50%of loading energy lies in each token's strongest4factors.

For row a, participation=(sum_j a_j²)²/sum_j a_j⁴. It is an effective factor
count, not a thresholded number of nonzero factors. Zero rows reported separately.
Top4 is an aggregate energy statistic; it does not establish per-token fidelity.
Outside-rank128 remainder and common output remain explicit.

Red-team a failed C with optimization, rank-subspace, raw-versus-normalized
weighting and orthogonal-versus-oblique assumptions. A passed C is not causal
identification. Independent rotation starts, input-function complexity and
native selective interventions remain necessary.

Primary method context: [Kaiser1958](https://cda.psych.uiuc.edu/psychometrika_highly_cited_articles/kaiser_1958.pdf)
and [Rohe–Zeng2023](https://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Rohe23.pdf).
The latter establishes inference under specified latent-factor distributions,
including a leptokurtic condition. Learned vocabulary weight rows have not
been shown to satisfy those assumptions; its recovery guarantee is not ours.
