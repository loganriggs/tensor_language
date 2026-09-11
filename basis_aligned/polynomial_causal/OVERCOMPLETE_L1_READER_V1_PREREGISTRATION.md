# Native overcomplete L1 input-reader dictionary

This changes the discovery objective and allows2304features in1152input
dimensions. It reuses `quadratic_token_dictionary_v1` accelerated proximal
code/dictionary solves. The earlier512token-function dictionary was not an
overcomplete input-reader dictionary. No new optimizer is being introduced.

Native L/R rows are normalized individually. Split paired native products
with seed700:3072products/6144readers fit the basis,1536products/3072readers
are held out. Seeds0/937 initialize2304atoms by sampling training readers
without replacement. Fit

$$
\frac{1}{n}\left[\frac12\|Y-ZB\|_F^2+.05\|Z\|_1\right],
\qquad \|B_{k,:}\|_2\le1.
$$

The fixed penalty is inherited from the planted control, not tuned on native
held-out weights. Equal normalized-reader weighting is an assumption; this
does not optimize full-U functional importance. Native product pairings and
Down remain fixed. Overcomplete features may still memorize individual readers.

For each start allow1800soft fit seconds/max2000outer cycles. Each conditional
solve has up to300steps at1e-6relative stationarity. Begin FP32; switch once to
FP64 when FP32 joint stationarity<=1e-4, a conditional solver reports a
nondecreasing-step stop, or900seconds have elapsed. Evaluate
original FP64 normalized weights every5cycles and at termination. Joint
convergence requires two consecutive FP64 checks <=1e-5 relative stationarity
and their relative objective change <=1e-8. Save codes, basis, precision stage,
history and seed to a new atomic checkpoint at least every60seconds at cycle
boundaries, and before scoring. A timeout is unfinished optimization, not a
structural null. Overall job alarm7200seconds includes encoding/scoring.

Compare each fitted dictionary to exactly its own untrained sampled dictionary.
For both, solve Lasso(.05) codes on all reader vectors (row-separable;
max5000steps/tolerance1e-7), select128largest code magnitudes, then solve values
by least squares on that support. Multiply back original reader norms through
the LS targets. This gives equal128-term execution cost; support selection is
still heuristic. Training-reader and held-out-reader squared normalized errors
are separate. Full folded coefficient capture includes all products and is
not a held-out full-tensor score. No text, activations, corpus, labels or CE fit.

Registered predictions:

- A: all saved/replayed sparse programs agree with dense reconstructed readers
  to1e-8relative error; all support normal residuals<=1e-8; finite FP64 scores
  and dictionary row norms<=1+1e-6.
- B: both basis fits satisfy the joint criterion; all four scoring encoders
  satisfy relative stationarity<=1e-7 and maximum Lasso KKT residual<=1e-5.
- C: both fitted dictionaries' held-out128-term capture exceeds the best of
  the two untrained dictionaries by>=.02absolute.
- D: both fitted dictionaries' full-U coefficient capture exceeds the best
  untrained counterpart by>=.01absolute.

Keep convergence and quality verdicts separate, and report combined eligibility.
C/D failures only constrain these fits, penalty and encoder. Negative results
must distinguish local minima, support error, and missing shared features.
Untrained sampled dictionaries are a strong finite-weight baseline, not a
random-isotropic substitute for native geometry.

Compiled program price is9,142,272matrix coefficients:2304x1152feature map,
9216x128sparse values and1152x4608Down; plus1,179,648support indices and1152bias
entries. The complete1152-feature dictionary had7,815,168matrix coefficients;
do not claim matched capacity with it. U and native background remain charged.
`RectangularSparseReaderProgram` inherits the existing execution/removal/dose
algebra; interventions are internal feature edits and can leave the subspace
of features reachable by changing only the raw input. This supplies an explicit
candidate interface, not evidence of native causal selectivity or OOD behavior.

Managed lane1 only, after existing queued work. Source and dependency bindings
are frozen before enqueue. CPU controls cover rectangular execution and the
reused Lasso/support encoder; native hardware preflight precedes long fitting.
