# Weights-first discovery: actual methods, assumptions, and bottlenecks

**Latest result:** [09:18 completed dictionary and output-fit results](#update0918).

**11 September, 09:02 UTC.** Discovery remains weight-based. Freeze candidate
factorizations before testing their behavior on FineWeb; reserve Pile for a
separately labelled corpus-shift check. Historical fits adapted on Pile cannot
then claim untouched Pile validation. A million-token pass is not a prerequisite
for the next discovery experiment.

The main weakness is currently the mathematical objective and its restrictions,
not a shortage of text. Some methods fit the entire folded polynomial; the latest
reader dictionaries instead fit a proxy: sparse representations of the model's
existing input readers. That proxy can miss structure requiring new products.
We have tried several serious solvers, but have not exhausted the structural
families or established globally best fits. Calling this a comprehensive
state-of-the-art search would overstate the evidence.

## The object being factored

At the last MLP's already-normalized input x, the bilinear contribution is

$$
q(x)=D[(Lx)\odot(Rx)]+b.
$$

Here x has 1152 coordinates, L and R each contain 4608 reader vectors, D maps
the 4608 products back to 1152 residual coordinates, and U is the unembedding.
Folding U into D gives token loadings C=UD. The quadratic matrix for token v is

$$
Q_v=\sum_{j=1}^{4608}C_{vj}
\frac{\ell_jr_j^\top+r_j\ell_j^\top}{2},
\qquad [Uq(x)]_v=x^\top Q_vx+(Ub)_v.
$$

This is already an exact shared factorization across all tokens. Discovery asks
whether a different set of products, shared readers, overlapping blocks, or
structured stages explains those matrices more simply and stably.

We do not construct the roughly 66.8-billion-entry tensor. Inner products of
two symmetric product matrices reduce exactly to reader dot products:

$$
\langle\operatorname{sym}(ab^\top),\operatorname{sym}(cd^\top)\rangle
=\tfrac12[(a^\top c)(b^\top d)+(a^\top d)(b^\top c)].
$$

Together with the cached output metric U^T U, this permits full-unembedding
coefficient losses without text or the dense tensor. The 50,304 token slices
are not 50,304 independent observations: they are linear combinations of at
most 1152 residual-output quadratic matrices.

This polynomial describes the folded MLP contribution, not the final logits
by itself. Actual prediction also includes the residual stream, final RMS
normalization and the model's tanh logit cap.

## What has actually been used

| Structural assumption | Discovery / optimization method | Evidence and remaining limitation |
|---|---|---|
| A small shared bank of arbitrary two-reader products | Alternating least squares, damped Gauss–Newton, and a penalized joint fit | The penalized 128-product fit reached local stationarity, about 8.70% full coefficient capture. Other solver runs remained unfinished. The penalty and small bank are restrictions. |
| Signed squares of shared linear features | Joint fitting followed by numerical polishing | A 256-square fit locally converged around 9.65%. Squares can express products through polarization, so this is not an entirely independent hypothesis at unequal capacity. |
| Sparse interactions in one common input basis | Riemannian conjugate gradient: update a basis while maintaining orthogonality | Local fits obtained, but functions differed between restarts. Orthogonality and sparse-edge allocation restrict the answer. |
| Several small, potentially overlapping quadratic blocks | Manifold conjugate gradient, exact-Hessian trust regions, and conditional linear/core solves | Broad block fits still missed convergence. A different small-frame family converged around 5.94%, with a restrictive 92-dimensional total input span. Neither settles general overlapping blocks. |
| Tokens sparsely reuse shared quadratic functions | Alternating sparse coding and dictionary updates, then fixed-support least-squares debiasing | A 512-function fit locally converged: 16.56% centered coefficient capture, 20.72% after debiasing. The individual functions remained dense; sparse token usage alone did not yield simple computation. |
| Native input readers are sparse in a complete shared basis | Fourth-moment maximization by matching, stretching and projection (MSP) | Both orthogonal starts gave about 47.31% held-out reader capture, below PCA's 47.54%, and missed combined convergence. A cheap sample-alignment null explains much of the training objective. |
| Those shared features are nonorthogonal | Covariance or robust Tyler-shape whitening, MSP rotation, then sparse encoding with exact fixed-support least squares | Both ordinary-covariance starts converged; held-out reader capture was 43.79% and 43.71%. Tyler fits remain in progress at this snapshot. Better geometry has not yet produced better generalization. |
| More reusable features than input dimensions | **Queued:** 2304-feature dictionary in 1152 dimensions, alternating accelerated proximal sparse-code and bounded-dictionary updates | Changes the objective and allows overlapping features. Two starts, FP64 stationarity checks, recoverable checkpoints. Fixed L1 penalty and native product pairings remain assumptions. No native result yet. |
| Existing candidate readers are useful but their output weights need adjustment | **Queued:** exact conditional Down least-squares solve through the product Gram eigendecomposition | Tests eight frozen dictionaries at unchanged output capacity. It solves the linear subproblem, not the nonlinear reader-discovery problem. |
| Shared computation uses multiple full-rank linear stages | Mixed-radix structured transforms surrounding a product bank, joint L-BFGS fitting | Both continuations remain unconverged, about 3.51% and 3.31% capture. Fixed wiring and difficult optimization remain plausible explanations. |

Capture means one minus relative squared reconstruction error in the stated
metric. It is not behavioral accuracy. Capacities, penalties, and centered
versus full targets differ; these numbers are not a fair leaderboard.
[Primary receipts and additional families](../../WEIGHT_ONLY_METHODS_INDEX.md).

MSP has recovery results under a particular random sparse generative model;
we have not established those assumptions for learned network weights.
[Primary MSP paper](https://www.jmlr.org/papers/v21/19-755.html).
Sparse dictionary learning has a substantial optimization literature; our
native queued implementation is an alternating proximal solver, not a claim
to have implemented every algorithm in that literature or Mairal's exact
online algorithm. [Sparse-coding reference](https://www.jmlr.org/papers/v11/mairal10a.html).
Eliminating linear variables before nonlinear optimization is the variable
projection principle; the queued output solve uses the conditional linear
step, not a complete reduced-objective reader optimizer.
[Variable projection](https://www.cs.umd.edu/users/oleary/software/varpro.pdf).

## The assumptions that most need challenging

**Reader sparsity is a proxy for polynomial simplicity.** The latest dictionary
fits minimize reconstruction of individually normalized L/R rows. They do not
use U or D to discover their features, and preserve the original product
pairings. Two large reader errors might cancel downstream; two small errors
might alter an important interaction. Full folded scoring afterward detects
that mismatch but does not repair the discovery objective. Earlier free-product
fits did optimize the folded object, so the gap is adequate flexible joint
optimization, not the complete absence of such an attempt.

**Orthogonality, equal block sizes, small global subspaces and fixed wiring can
exclude the answer.** Circuit parts may overlap, have unequal sizes, and share
intermediate computations without giving a low-rank output tensor. A rank
bound on one representation does not bound general arithmetic reuse. Our
constructive counterexample already demonstrates that distinction.

**Sparse coding and feature discovery can fail separately.** On a planted toy,
changing only the encoder raised reconstruction from 98.09% to 99.96%. Native
support selection among thousands of features cannot be exhaustively searched.
The queued L1 method therefore uses converged sparse coding followed by an
exact value solve on the chosen support, while acknowledging that support
selection remains heuristic. Its penalty 0.05 was inherited from a small toy;
a native fixed-basis audit already found a dimension-scaled penalty better.
That is a known limitation to report, not evidence against sparse structure if
the frozen run fails.

**The approximation norm itself is an inductive bias.** Uniform coefficient
error does not equal error on normalized input vectors. For a symmetric error
matrix E and x uniformly distributed on the radius-sqrt(d) sphere,

$$
\mathbb E[(x^\top E x)^2]
=\frac{d}{d+2}\left(2\|E\|_F^2+(\operatorname{tr}E)^2\right).
$$

The trace is the sum of diagonal entries. It supplies a radial term proportional
to the squared input norm. Keeping that term exactly in three frozen reader
fits raises ideal-sphere capture from roughly 45–46% to 74%, while changing
coefficient capture by only about 0.13–0.14 percentage points. However, the
radial term plus bias alone scores 63.22%. This is a useful metric warning,
not a discovered 74%-complete circuit or a FineWeb result.
[Frozen correction receipt](../../FROZEN_READER_RADIAL_V1_AUDIT.json).
Actual model states need not be uniformly distributed on that sphere.

**Convergence and identification differ.** Small objective changes alone do
not establish stationarity; stationarity does not establish global recovery;
global reconstruction would still leave sign, scale, permutation and sometimes
larger ambiguities. Held-out weight rows test reuse of a dictionary, but are
not independent corpus validation and can penalize legitimate specialized
features used by only a few products.

## Bottlenecks and next priorities

The large tensor is handled implicitly. Remaining costs include repeated dense
matrix contractions, eigendecompositions/SVDs, difficult nonlinear geometry,
sparse support search, and FP64 certification on a consumer GPU. More tokens
do not remove those bottlenecks. Long runs that hit a time limit remain
unfinished; the user asked for convergence, so they cannot be reported as
negative structural conclusions.

The immediate managed queue contains the exact conditional output comparison
and the overcomplete L1 discovery fit, behind the live oblique comparison.
The most important subsequent weight-based decision is whether better encoding
and output adjustment make those features useful, or whether discovery must
move directly to a more flexible joint folded-polynomial objective with
adaptive/overlapping components. This should follow the receipts rather than
an automatic identical continuation.

No new million-token discovery sweep is being added. FineWeb comes after
candidate freezing for prediction and intervention checks; Pile is a separate
shift test. If weight-only approaches remain inconclusive, any later
data-weighted discovery will be identified as a different objective. None of
the reconstruction scores above establishes OOD prediction, extraction,
selective removal, or composition/reuse.


<a id="update0918"></a>
## 09:18 update: completed dictionaries and exact output fits

**The nonorthogonal dictionary comparison did not resolve the weakness of the
orthogonal method.** Both ordinary-covariance starts converged, while both
Tyler starts hit their time limits. Held-out reader capture was 43.64–43.79%,
below the orthogonal and PCA controls. Numerical checks passed; the registered
convergence and quality predictions did not.
[Completed comparison](../../OBLIQUE_READER_DICTIONARY_V1_RESULT.json).

The subsequent output-weight experiment solved eight conditional linear
problems exactly in about 17 seconds overall. Each eigensolve took roughly
0.8 seconds. This is now a completed comparison, not a queued proposal:

| Frozen reader dictionary | Original Down: coefficient capture | Refitted Down: coefficient capture |
|---|---:|---:|
| Identity |23.25%|24.21%|
| PCA |28.29%|29.17%|
| Orthogonal, two starts |29.56–29.58%|30.32–30.33%|
| Ordinary nonorthogonal, two starts |29.85–29.91%|30.71–30.76%|
| Tyler nonorthogonal, two starts |29.75–29.79%|30.60–30.65%|

All candidates improve, and each learned family beats the refitted PCA control
by at least one percentage point in both starts. However, no learned family
gets the registered one-percentage-point improvement from changing Down itself.
That prediction remains a miss. All numerical checks pass; exact Down fitting
does not turn an unconverged reader fit into a converged one.
[Output-fit receipt](../../READER_CONDITIONAL_WRITER_V1_RESULT.json).

We then checked the strongest apparent positive: the two ordinary-covariance
fits have converged bases, exact output solves and almost identical capture.
Do they represent the same computation? **Their complete folded functions have
cosine similarity only 0.368**, missing the registered 0.9 threshold. This
comparison includes input interactions and output writers, so it is not merely
a disagreement about feature signs, ordering or coordinate names. It weakens
stable-identification claims despite the modest reconstruction advantage.
[Function stability](../../REFITTED_READER_FUNCTION_STABILITY_V1_AUDIT.json).

A separate CPU experiment replaced independent reader fitting with full folded
coefficient fitting on 128 products, keeping their supports and Down fixed.
It reached joint local convergence after a saved-state continuation; full
capture rose from 29.906% to 29.944%. The initial timeout remains recorded.
This settles that small conditional problem, not all products or feature
selection. [Converged result](../../FOLDED_SUPPORT_CONTINUE_V2_RESULT.json).

The overcomplete L1 discovery experiment is now running. Its joint gradient is
still above threshold, even though its conditional code and dictionary solves
are converging; no native verdict yet. No identical MSP continuation is queued.
After L1, a small frozen-program FineWeb check will test whether the known
radial correction helps actual readout behavior. It uses 64 cached sequences,
8,192 prediction positions, eight fixed alternatives and ten body forwards.
This is a diagnostic of known weak candidates and the error metric, not their
promotion to identified circuits. It fits nothing from text and includes the
strong radial-only baseline. [Frozen protocol](../../FROZEN_RADIAL_FINEWEB_V1_PREREGISTRATION.md).
