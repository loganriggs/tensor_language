# Three-hourly mathematical review — 2026-08-30 13:45 UTC

## Bottom line

The completed rank grid looked like a rank problem because every candidate's largest
absolute interface error was `m16 -> m16`. A tensor-rank lower-bound certificate now
shows that this diagnosis was wrong in the useful sense: `m16 -> m16` is about **9.68×
higher-energy** than the next source owner on matched support, but its normalized
rank-16 unfolding tail is only **1.10×** the next owner. It is not exceptionally
higher-rank. It is exceptionally large.

The right next mathematical model is therefore not “give `m16` a bigger private CP
branch.” It is a tensor program evaluated and, if validation warrants, fitted under
both absolute error and block-relative causal error. This preserves the tensor network
while preventing pooled MSE from buying one large interface and ignoring many small
ones.

Strict whole-model ledgers do not move: 5.348245316% certified removable storage,
10.923302467% named deletion cross-entropy, 4.72714 nat / 89.076697533% unexplained,
and 0/68 terminal circuits.

## Evidence inspected

- Git HEAD and origin were current; the response grid and analysis are published.
- The response grid has 51/51 healthy FIT cells and no validation/EVAL access.
- Shared rank 32 has pooled FIT MSE 0.016047438 and reconstructs 65.17% of pooled
  response energy at $P=3200,C=32$.
- A separate GPU job, `head_grain_expressibility.py`, was live at about 6.4 GB. The
  analysis here was CPU-only and completed in 3.26 seconds.
- The first production certificate attempt failed closed because only 70.52% of FIT
  cells are valid and no document contains all 49 targets. No spectrum or receipt was
  produced from that attempt.

## Executed move: unfolding lower bounds on exact observed rectangles

For residual owner block $E_{g,h}$ and any CP-rank-$r$ correction $X$,

$$
\|E_{g,h}-X\|_F^2
\geq \max_j\sum_{i>r}\sigma_i\!\left(M_j(E_{g,h})\right)^2,
$$

because every unfolding of a CP-rank-$r$ tensor has matrix rank at most $r$, and
Eckart--Young gives the best rank-$r$ matrix error. This is a certificate: optimizing
a different CP factorization cannot beat the lower bound.

The missing-data repair used no imputation. The validity mask is exactly broadcast
over phase and source. For each target owner $h$, the analysis retained all targets in
$h$ and only documents valid for every one of those targets. The primary `m16` target
rectangle contains 6 targets and 46 complete documents. Every source owner is therefore
compared on identical target/document support.

### Primary numerical result: sources into `m16` targets

| source owner | residual energy/cell | 95%-energy CP-rank lower bound | normalized rank-16 lower-bound tail |
|---|---:|---:|---:|
| a8 | 0.002855 | 18 | 0.05689 |
| a16 | 0.000135 | 25 | 0.13153 |
| **m16** | **0.148770** | **25** | **0.14514** |
| a3 | 0.002153 | 22 | 0.10503 |
| m14 | 0.004412 | 17 | 0.05266 |
| m13 | 0.015375 | 9 | 0.01292 |

The registered energy ratio is 9.6762, while the registered normalized rank-tail ratio
is 1.1035. This satisfies the prospective **amplitude/weighting** branch and falsifies
the **asymmetric-rank** branch.

The raw-versus-residual comparison reveals why pooled MSE was misleading. Shared rank
32 removes 68.41% of raw `m16 -> m16` energy, but removes essentially none of
`a16 -> m16` (−0.32%) or `a3 -> m16` (0.38%). Across all 36 owner pairs, median
block-relative energy recovery is only **5.57%**, although pooled recovery is 65.17%.
The median recovery by source owner is 68.08% for `m16`, 9.33% for `a8`, 6.05% for
`m13`, 4.42% for `m14`, 0.30% for `a3`, and 0.22% for `a16`.

Thus the 32-dimensional code is mostly a compact program for the high-amplitude `m16`
family, not a uniform 49-by-49 causal-interface compiler. This is more informative
than merely saying `m16` has the largest absolute error.

The mathematical basis is classical tensor rank and block-term decomposition:
[Kruskal (1977)](https://doi.org/10.1016/0024-3795(77)90069-6) and
[De Lathauwer (2008)](https://doi.org/10.1137/070690729). The planted CPU gate recovered
the zero post-rank-2 tail of a rank-2 four-way tensor and a positive rank-2 lower bound
for a rank-5 tensor. One test initially required exact float equality under axis
permutation and failed at about $10^{-16}$; the preserved repair uses a tight numerical
tolerance. Four tests pass.

## Ranked top three genuinely new mathematical moves

### 1. Block-relative causal tensor approximation, with absolute error retained

**Exact bilin18 object.** The 36 source-owner/target-owner response blocks of
$R_{pstd}$ and their shared/private tensor programs.

**Operational definition.** Keep $(P,C)$ as separate simplicity prices and keep pooled
MSE, but add the meaningful-interface coordinate

$$
L_{\mathrm{rel},\infty}
=\max_{g,h:\,\|R_{g,h}\|^2/n_{g,h}>\tau}
\frac{\|R_{g,h}-\widehat R_{g,h}\|_F^2}{\|R_{g,h}\|_F^2}.
$$

$\tau$ must come from split-half measurement noise, not from inspecting model fit.
Optimization can use a fixed epigraph or smooth maximum while the reported frontier
retains pooled absolute error, worst relative error, $P$, and $C$ separately. This is
still a multilinear tensor program; there is no data-dependent router.

**Assumptions that may fail.** Small blocks may be dominated by measurement noise;
relative error without the noise floor would overweight meaningless effects. A
balanced training fit may also sacrifice important pooled CE.

**Prediction beyond reconstruction.** If this is the right simplicity notion, a
frontier survivor should transport on held-out documents across many owner pairs,
rather than reproduce the current 65%-pooled/5.6%-median-block disparity. Selective
removal should have smaller collateral effects because small but real interfaces are
no longer ignored.

**Cheapest falsifier.** First score the already frozen candidates on validation using
both pooled and noise-thresholded block-relative error. If block-relative transport is
unstable or uncorrelated with later finite edits, do not refit. If it is stable, run
one prospectively weighted rank-32 fit and require held-out worst-block improvement at
matched $(P,C)$ and bounded pooled regression.

### 2. Robust CP identifiability and seed-atom stability

**Exact bilin18 object.** Each shared-rank-32 predicted tensor, whose fourth factor is
the 229-by-32 document-code matrix, and the three independently optimized seeds.

**Theorem/definition.** For a four-way rank-$R$ CP representation, the generalized
Kruskal condition

$$
k_A+k_B+k_C+k_H\geq 2R+3
$$

is sufficient for essential uniqueness up to permutation and reciprocal scalings.
Robust variants replace exact k-rank by quantitative conditioning
([Bhaskara et al., 2014](https://proceedings.mlr.press/v35/bhaskara14a.html)).

**Assumptions that may fail.** Exact uniqueness applies to each fitted prediction, not
automatically to the unknown response tensor. The phase factor has only two rows, CP
approximation can be ill-conditioned, and approximate decompositions can move greatly
under small tensor changes even when algebraic uniqueness holds.

**Prediction beyond reconstruction.** Well-conditioned identifiable atoms should align
across seeds after one permutation/sign gauge and should predict the same fresh
intervention and edit direction. Poorly aligned atoms are not semantic components even
if their summed tensor has low MSE.

**Cheapest falsifier.** CPU-check factor ranks and condition numbers, solve a global
seed-to-seed atom assignment using product cosine similarity over phase/source/target/
document factors, and compare per-atom response edits. Reject atom-level semantics if
matching is weak or unstable.

### 3. Empirical balanced causal quotient across MLP/RMSNorm/residual interfaces

**Exact bilin18 object.** At an early residual interface (starting with MLP0/MLP1), let
$x$ be the local state, $W_c$ its covariance under natural/token/context perturbations,
and $W_o=J^\top FJ$ the pullback of downstream circuit/logit functionals through the
actual suffix, with $F$ an output Fisher or explicit terminal-circuit weighting.

**Theorem/definition.** Linear balanced truncation orders state directions by Hankel
singular values, jointly measuring reachability and observability; empirical Gramians
extend the computation to nonlinear systems
([Lall, Marsden & Glavaški, 2002](https://doi.org/10.1016/S0098-1354(02)00120-5),
[Himpe & Ohlberger, 2013](https://arxiv.org/abs/1301.6879)). The proposed quotient keeps
directions large in both $W_c$ and $W_o$, not directions with large activation variance
alone. Its causal validity is then judged by an intervention-commutation error, in the
sense of graded causal abstraction
([Geiger et al., 2025](https://www.jmlr.org/papers/v26/23-0058.html)).

**Assumptions that may fail.** The classical truncation bound is for stable linear
systems, not transformers with RMSNorm, attention, and finite edits. Local Fisher
geometry can miss large nonlinear interactions and context changes.

**Prediction beyond reconstruction.** Discarded balanced directions should have small
finite downstream CE/circuit effects, independently reduced MLPs should compose, and
the quotient should predict which removal operations are selective rather than merely
which local activations reconstruct well.

**Cheapest falsifier.** On cached FIT activations, fit the empirical Gramian basis on
half the documents and predict finite suffix/circuit damage of held-out perturbations
on the other half. If predicted versus actual damage is not calibrated, do not build a
whole-model balanced compiler.

## Pruned mathematics for now

- **More owner-private CP rank:** directly pruned by the 1.10 normalized tail ratio.
- **Plain HOSVD, activation SAE, or weight dictionary learning:** optimizes geometry or
  pooled reconstruction and would repeat the exact failure exposed here.
- **MDL/prequential coding:** useful only after held-out prediction and a decoder/code
  protocol exist; otherwise a shorter bad causal program wins.
- **Information bottleneck:** estimating mutual information in these continuous states
  adds another unstable estimator and does not by itself guarantee intervention
  commutation. The balanced causal quotient gives a cheaper operational test.
- **Hankel/automata minimal realization:** promising for genuinely sequential terminal
  circuits, but the current response library has no registered concatenation algebra,
  so a Hankel matrix would be an arbitrary reshaping.
- **Sparse program synthesis/DAG search:** no validated primitive library or utility
  metric yet; search would optimize naming rather than causal transport.
- **Polynomial invariant theory/gauge quotients:** still important for certifying a
  validation survivor, but premature before seed atoms and block-relative transport
  survive.

## Artifacts and preserved failures

- Preregistration: `CAUSAL_RESPONSE_RESIDUAL_UNFOLDING_CERTIFICATE_PREREGISTRATION.md`
- Missing-support amendment:
  `CAUSAL_RESPONSE_RESIDUAL_UNFOLDING_CERTIFICATE_AMENDMENT_1.md`
- Machine receipt: `causal_response_residual_unfolding_certificate_receipt.json`
- Source/test: `causal_response_residual_unfolding_certificate.py` and
  `test_causal_response_residual_unfolding_certificate.py`
- Preserved failures: exact axis-permutation float equality was too strict; the original
  all-cells-valid production precondition failed at 70.52% support. Neither failure was
  converted into a scientific negative or silently imputed.
