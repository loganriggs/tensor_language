# Three-hour mathematical review — 2026-08-30 10:31 UTC

## Outcome first

The strict scientific ledger is unchanged:

- certified removable storage: **29,196,288 / 545,904,054 = 5.348245316%**;
- named deletion-CE: **0.57968 / 5.30682 = 10.923302467%**;
- unexplained deletion-CE: **4.72714 nat = 89.076697533%**;
- terminal circuit actions jointly passing extraction, selective removal, and OOD:
  **0 / 68**.

The signed FIT causal-response tensor has completed and is preserved, but this review
did **not** deserialize its values.  Its full training transaction received an exact
independent **NO-GO** at commit `2b9a2bd3`: 67 ordinary tests passed, yet all four
adversarial attacks reproduced.  The attacks are post-manifest input/source drift, a
terminal-only state when the second hardlink fails, stale authority binding on the
failure path, and caller-forged authority accepted by the public loader.  The audit
also found two omitted transitive runtime sources.  This is now the precise blocker
to real response factorization; it is not missing data, GPU availability, or slow
loading.

The current GPU remains occupied by the two-extra-seed a8 learned-grouping job.  The
outcome-blind CPU interval was therefore used for a new exact hierarchy gate.

## Ranked top three genuinely new mathematical moves

1. **Dimension-tree selection from exact tensor-network cut ranks**, implemented and
   tested here.
2. **Signed separable sparse nonnegative tensor factorization**, to test the user's
   “many global circuits, few active per document” hypothesis without top-k routing.
3. **ICA/varimax gauge fixing of dense document-code blocks**, to turn an arbitrary
   low-rank coordinate system into a potentially stable sparse semantic coordinate
   system at zero reconstruction cost.

These are complementary.  Move 1 asks whether a hierarchy can be small at all.  Move
2 asks whether the response is an additive parts library with sparse use.  Move 3 asks
whether an existing continuous code contains a simpler basis hidden by a legal gauge.

## 1. Dimension-tree selection from exact cut ranks — executed

### Exact object in bilin18

Apply this first to the dense tensor represented by any fitted response program,

$$
\widehat R\in\mathbb R^{2\times49\times49\times K_d},
$$

with modes `phase`, `source circuit`, `target circuit`, and `document code`.  It can
also be applied to a complete rectangular response subtensor.  It must **not** be
applied to the observed tensor after replacing missing cells by zero.

There are three binary four-leaf trees, represented by their central cuts:

$$
(P,S)\mid(T,D),\qquad
(P,T)\mid(S,D),\qquad
(P,D)\mid(S,T).
$$

For a cut $A\mid A^c$, reshape the physical tensor into a matrix
$\widehat R^{(A)}$.  Every tensor network whose graph contains that edge obeys

$$
\operatorname{bond\ dim}(A\mid A^c)
\;\geq\;
\operatorname{rank}\widehat R^{(A)}.
$$

Conversely, for a tree tensor network, the ranks of the edge matricizations are the
minimal exact hierarchical ranks.  They are invariant under every invertible change
of coordinates at the leaves and every internal tensor-network gauge.  They therefore
measure the represented tensor rather than a particular factorization.

For leaf ranks $r_i$ and central rank $r$, the literal scalar storage of the four-leaf
hierarchical Tucker tree used by the new gate is

$$
P_{\rm HT}
=\sum_{i=0}^3 n_i r_i
+r_0r_1r+r_2r_3r+r^2.
$$

This is literal storage, not quotient dimension.  Both should eventually be reported:
literal storage prices deployment, while quotient dimension prices the number of
observable degrees of freedom after gauges are removed.

The relevant established result is hierarchical SVD: a dimension tree defines its
hierarchical ranks through matricizations and admits SVD-based quasi-best
approximations with rigorous Frobenius error control.  See [Grasedyck,
*Hierarchical Singular Value Decomposition of Tensors*
(2010)](https://doi.org/10.1137/090764189).  Tensor train is the path-tree special
case; TT-SVD gives the corresponding constructive decomposition and error bounds
([Oseledets 2011](https://doi.org/10.1137/090752286)).

### Assumptions that may fail

- The observed response tensor is masked.  Exact cut-rank theorems apply to a complete
  tensor, not a zero-filled mask.  The first lawful real use is on a fitted dense
  program or a prospectively defined complete rectangle.
- A low Frobenius tail need not mean low CE harm, selective removal, or OOD transport.
  The tree is only a candidate executable topology until causal consequence tests
  pass.
- The best tree for the response tensor may not be the best tree for folded MLP
  weights or for the whole residual program.
- Numerical rank needs a stability check across document halves, seeds, and tolerance.
  A slowly decaying spectrum is not a certified small bond.
- Four coarse modes may hide a better hierarchy inside source/target owner groups.
  A positive result should be refined, not treated as the final model graph.

### Prediction beyond reconstruction

If one tree has materially smaller stable cut ranks, it predicts a reusable interface:
for example, a small $(P,S)\mid(T,D)$ bond says a small source-side message should
predict target-by-document responses.  That interface must then:

1. infer held-out document responses from the frozen physical arm panel;
2. transport across document halves and EVAL without changing the tree;
3. support atom or subtree removal with lower unrelated-target damage than a
   price-matched flat basis;
4. compose under the planned pair-intervention test.

The cut rank also supplies a falsifiable lower bound: a proposed tree with a smaller
bond cannot reproduce all registered response coordinates, regardless of optimizer.

### Cheapest falsifier

After the audited transaction is repaired, fit the already frozen candidates and
controls.  For each dense candidate tensor, compute all three central spectra plus the
four singleton spectra on each document half.  Reject a hierarchy if:

- no tree is a unique literal-price winner;
- its selected ranks change by more than two across halves or seeds;
- its normalized spectrum changes by more than `0.10`;
- its truncated tree loses to the price-matched dense SVD control on calibrated
  missing-cell prediction or worst owner-pair NRMSE;
- its subtrees fail the later composition/removal tests.

### Executed proof check

`response_tree_rank_diagnostics.py` and four tests now implement physical
matricization, exact and energy ranks, literal tree prices, all three topology choices,
invertible leaf-gauge invariance, and fail-closed handling of incomplete/malformed
inputs.

The planted tensor has shape $5\times6\times7\times8$ and was generated by the
$(0,1)\mid(2,3)$ tree.  The gate recovered:

| central cut | bond rank | minimal literal HT storage | dense fraction |
|---|---:|---:|---:|
| $(0,1)\mid(2,3)$ | 2 | 97 | 0.0577381 |
| $(0,2)\mid(1,3)$ | 6 | 175 | 0.1041667 |
| $(0,3)\mid(1,2)$ | 6 | 175 | 0.1041667 |

All four tests pass in 2.01 seconds.  This is a planted known answer, not evidence that
bilin18 has this hierarchy.

## 2. Signed separable sparse nonnegative tensor factorization

### Exact object in bilin18

The signed response cannot be fed honestly to ordinary NMF.  Instead make an explicit
sign channel:

$$
R^+=\max(R,0),\qquad R^-=\max(-R,0),\qquad R=R^+-R^-.
$$

Fit the nonnegative five-mode tensor

$$
X_{q,p,s,t,d}\approx
\sum_{k=1}^K
Q_{qk}A_{pk}B_{sk}C_{tk}H_{dk},
\qquad Q,A,B,C,H\ge0,
$$

and penalize document-code sparsity in $H$.  Each component remains a rank-one tensor
program and the complete output is a sum of products.  There is no input-dependent
top-k branch.  Sparse $H_d$ means few library parts are used on a document, while the
overall model may have high rank across all documents.

This is much closer to the user's proposed sparse circuit library than a weight SAE.
The learned atoms live in causal response space, and their value is judged by new
interventions rather than activation reconstruction.

Under separability or filled-facet conditions, NMF can identify parts rather than only
an arbitrary rotated subspace.  The assumptions and geometry are developed by
[Donoho and Stodden (2003)](https://papers.neurips.cc/paper_files/paper/2003/hash/1843e35d41ccf6e63273495ba42df3c1-Abstract.html)
and weakened to subset separability by
[Ge and Zou (2015)](https://proceedings.mlr.press/v37/geb15.html).  For nonnegative
tensors, best-approximation uniqueness is generically better behaved than unconstrained
CP in important cases, although rank-$r$ fitting is still not obtained by naive
deflation ([Qi, Comon, and Lim 2016](https://arxiv.org/abs/1410.8129)).

### Assumptions that may fail

- Circuit responses may rely on cancellation; splitting signs can double the library
  and produce paired positive/negative atoms rather than mechanisms.
- There may be no anchor source, target, or document exposing one component cleanly,
  so separability and identifiability fail.
- Sparse use may be false: shared services can make every document code dense.
- An $\ell_1$ penalty chooses a scale unless factor norms are fixed; all columns must be
  normalized and literal metadata charged.
- Sparse calibrated codes do not create free conditional compute.  Unless a router is
  itself compiled, deployment still evaluates the fixed tensor sum.

### Measurable consequence beyond reconstruction

A real sparse-parts result requires all of the following at a matched literal price:

- held-out documents use substantially fewer active components than $K$;
- support and sign replicate across document halves and OOD domains;
- source/target anchor loadings predict which physical interventions activate each
  part;
- deleting one part damages its predicted targets more than matched negatives and
  off-targets;
- unseen pair interventions are predicted at least as well as by the signed CP
  baseline.

### Cheapest falsifier

On the 229 training documents only, fit $K\in\{8,16,32\}$ with three frozen seeds and
a short frozen sparsity grid.  Before validation, reject if no component has a stable
anchor, median active-code fraction stays above 50%, or factors fail seed alignment.
Only survivors may enter the existing 114-document calibrated-response and later
fresh-intervention gates.  This uses no new model forwards for the first stage.

## 3. ICA/varimax gauge fixing of document-code blocks

### Exact object in bilin18

For an unstructured low-rank or Tucker/block-term response block,

$$
R\approx BH^\top,
$$

there is an exact gauge

$$
(B,H)\mapsto(BG,HG^{-\top}),\qquad G\in GL(K).
$$

SVD chooses orthogonal variance directions, not semantic directions.  Use the gauge
without changing $BH^\top$: whiten $H$, then choose an orthogonal rotation by either
ICA (maximal non-Gaussian independence) or varimax (simple sparse loadings).  Apply the
inverse rotation to $B$.  This is not available across arbitrary CP atoms, whose
continuous gauge is mostly scaling, but it is legal inside a dense/Tucker block.

Under the linear ICA model, mutually independent sources with at most one Gaussian
component are identifiable up to scale and permutation; see
[Comon (1994)](https://doi.org/10.1016/0165-1684(94)90029-9).  Varimax is a weaker
simple-structure criterion rather than a generative identifiability theorem
([Kaiser 1958](https://doi.org/10.1007/BF02289233)).

### Assumptions that may fail

- Document codes may be dependent, Gaussian, or organized into dependent subspaces.
- Whitening amplifies small singular directions.
- A sparse code rotation can make the response basis dense and less editable.
- Better kurtosis or varimax score is not a causal result.
- This does not reduce rank or literal storage by itself; it earns simplicity only if
  codes quantize/compress better or axes support better edits.

### Measurable consequence beyond reconstruction

Because reconstruction is identical by construction, the rotation has to improve a
different quantity: seed/bootstrap axis alignment, code sparsity or prequential
codelength, anchor predictability, and selective atom removal.  This is a clean test of
whether a continuous rank-$K$ code conceals a semantic coordinate system.

### Cheapest falsifier

After one dense/Tucker response candidate exists, rotate only its 229 training codes.
Require exact response replay, then compare SVD, ICA, and varimax coordinates on
split-half axis alignment, active-code fraction, quantized codelength, and physical
anchor enrichment.  Reject before any GPU intervention if the axes are unstable or no
simplicity price improves.  A survivor gets the same held-out and selective-removal
tests as the original basis.

## Requested mathematics reconsidered and pruned

| Family | Decision in this review |
|---|---|
| Tensor/arithmetic-circuit rank | Promote four-mode tree cut ranks and literal HT storage. Raw CP rank alone remains optimizer-sensitive. |
| Simultaneous factorization/shared dictionaries | The frozen signed shared/private CP is still the main baseline. Add sparse signed NTF only as a prospectively frozen alternative. |
| Polynomial invariant theory/gauge quotients | Quotient-Jacobian and orbit-closure work already exists. The new use is exact gauge fixing of dense code blocks by ICA/varimax, with reconstruction held constant. |
| Algebraic complexity | Report literal HT storage, bond ranks, contraction operations, and quotient dimension separately. Do not collapse them into one favorable score. |
| System identification/minimal realization | Action-Hankel needs composed interventions; single-edit responses are not a transition-closed action system. Defer. |
| Hankel/automata | Keep circuit-family-specific parity/successor tests; generic token and MLP0 Hankel entry points remain pruned. |
| MDL/prequential coding | Use to decide whether sparse codes are actually cheaper after charging atoms, precision, router/calibration, and search. It cannot establish mechanism semantics. |
| Causal abstraction/bisimulation | Still downstream of an intervention-complete action set. Similar response vectors alone are not a congruence. |
| Information bottleneck | Still pruned: mutual information does not provide an executable decoder or edit semantics. |
| Sparse program synthesis | Restrict to signed sparse NTF with explicit tensor execution and held-out causal gates. No unconstrained semantic search. |
| Approximation certificates | HSVD supplies an exact Frobenius cut-tail certificate. Whole-model CE certification still requires the existing nonlinear hybrid telescope; global Lipschitz products remain vacuous. |
| Norm minimization before HOSVD | Useful for conditioning a chosen gauge, not for choosing the physical dimension tree or proving semantic components. |

## Priority after this review

1. Repair the five exact lifecycle defects and obtain a fresh independent GO.  Until
   then, opening FIT values would invalidate every downstream mathematical claim.
2. Once the sanitized 229-document artifact exists, run the frozen signed CP/control
   grid and the new tree-rank diagnostic on dense fitted programs.
3. If the CP frontier is dense or unstable, run the small signed sparse-NTF gate.  If a
   dense block is predictive but arbitrary, run the exact-replay ICA/varimax gauge
   test.

The executed scientific action in this review is the planted hierarchy proof gate.
It changes no fraction of the model explained and opens no FIT response, validation,
EVAL, model, row, token, activation, logit, or target.
