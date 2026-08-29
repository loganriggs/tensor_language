# Hourly strategic review — 2026-08-29 01:05 UTC

## Bottom line

The project has complete structural ownership of the network, but it has not yet
compiled most of the model into named, causally validated, simpler components.  The
honest global numbers do not move this hour:

| Ledger | Explained or certified | Largest remaining gap |
|---|---:|---:|
| Top-level structural inventory | 36/36 attention and MLP sites | No missing formula, but formulas are not explanations |
| Whole-program storage certified removable | 5.3481% | 94.6519% not certified removable |
| Named behavioral estimate | 32.1% +/- 6.4% | Different currency from causal recovery; cannot be added to it |
| Strict named causal CE recovery | 10.923% | 4.72714 nats, or 89.077%, remains unnamed |
| Final coupled extraction/removal/OOD actions | 0/68 | The semantic decision/replay producer is still absent |

Here **CE** means cross-entropy loss: the average negative log-probability assigned to
the true next token.  The strict causal percentage asks how much of a fixed measured
loss increase is recovered by named interventions.  The storage percentage asks how
many stored values can be removed while passing a separate whole-program consequence
test.  These denominators answer different questions and must not be combined.

The strongest new mechanistic result remains the failed MLP3--8-only de-alias test.
Late attention cancels most of the large compatibility effect between the early MLP
prefix and late MLPs.  Therefore attention plus its following bilinear MLP is the
smallest currently justified compiler unit.  Treating the late MLP suffix as an
independent replaceable component is empirically wrong.

## What changed in this hour

The exact grouped-block algebra from the mathematical review was turned into a
preregistered coefficient-level screen at block 3.  For attention output projection

\[
C,
\]

MLP product factors

\[
L,R,
\]

and MLP down projection

\[
D,
\]

each product gate has an exact scale gauge

\[
L_j\mapsto s_jL_j,
\qquad
R_j\mapsto s_j^{-1}R_j.
\]

This changes the parameter coordinates but not the gate product.  The positive scale

\[
s_j=\sqrt{\frac{\lVert R_j\rVert_2}{\lVert L_j\rVert_2}}
\]

minimizes the sum of the two squared row norms.  This is the precise, limited version
of norm minimization before HOSVD: it removes arbitrary scale imbalance, but it cannot
manufacture low tensor rank.

After balancing, each gate was weighted by the norm of its downstream `Down` column,
and the attention-to-product-factor interface

\[
A=
\begin{bmatrix}
\operatorname{diag}(w)\widetilde L C\\
\operatorname{diag}(w)\widetilde R C
\end{bmatrix},
\qquad
w_j=\lVert D_{:,j}\rVert_2,
\]

was decomposed exactly on CPU.  This is a downstream-weighted coefficient screen.  It
does not use activations, final logits, or cancellations between different gates, so
it cannot by itself establish behavioral simplicity.

### Result

| Quantity | Block 3 result |
|---|---:|
| Balanced/native weighted factor norm | 0.999758 |
| Minimum / median / maximum gauge scale | 0.8899 / 1.0000 / 1.0769 |
| Stable rank | 158.53 |
| Rank for 90% energy | 508 |
| Rank for 95% energy | 630 |
| Rank for 99% energy | 839 |
| Rank for 99.9% energy | 1005 |
| Rank-256 relative Frobenius error | 0.5708 |
| Rank-512 relative Frobenius error | 0.3123 |

The preregistered promising threshold was 95% energy at rank at most 256.  It failed:
block 3 needs rank 630.  Norm balancing reduces the weighted factor norm by only about
0.024%, because the trained model is already nearly balanced in this gauge.

The stable rank of 158.53 does **not** mean that 159 directions reconstruct the map.
Stable rank is total squared singular-value energy divided by the largest squared
singular value.  A large leading direction can make stable rank look small even when
hundreds of smaller directions collectively carry substantial energy.  The 95%-energy
rank and explicit approximation errors are the relevant quantities here; they expose
a long spectral tail.

This prunes raw coefficient HOSVD and norm minimization as the next large experiment.
It does not prune activation-weighted or final-consequence-weighted factorization:
natural text may occupy a much smaller part of this dense coefficient interface, and
the downstream network may ignore much of its output.

Artifacts:

- `GROUPED_BLOCK_COEFFICIENT_SCREEN_V1_PREREGISTRATION.md`
- `grouped_block_coefficient_screen.py`
- `grouped_block_coefficient_screen_results.json` (SHA-256
  `84159aa8c2a3a7fd3adb421b51798dd25259c65091d6450bfa3cea22267457f1`)
- `test_grouped_block_coefficient_screen.py`

Nine grouped-algebra and coefficient-screen tests pass.  The authoritative CPU run
took 0.38 seconds after checkpoint load, so data loading is not a bottleneck for this
weights-only question.

## Current largest gaps and confusing observations

1. **The coupled block interface is exact algebraically but not yet compressed.**  The
   four polarized terms replay the native MLP exactly, including RMSNorm, but evaluating
   all four native banks would cost more than the original MLP.  We need a shared
   activation/consequence-weighted factorization, not merely an identity.
2. **Global CE simplicity and conditional causal complexity coexist.**  On the complete
   five-action cube, about 98% of CE variance lies in degree-one and degree-two Walsh
   terms.  Yet the small higher-degree tail contains the early-prefix x attention x MLP
   compatibility effect.  A global 95%-energy compressor would erase the circuit under
   study.  Simplicity therefore has to be conditioned on the consequence or edit being
   preserved.
3. **Top-1 is a noisier judge than CE here.**  Its higher-degree action energy is about
   12.7--13.0%, versus 2.0--2.4% for CE, and several frontier differences are only 4--24
   tokens out of about 36,800.  CE is the primary optimization measure; top-1 remains a
   secondary extraction-style measure with uncertainty attached.
4. **The 68-action harness is physically assembled but scientifically unfinished.**  A
   reviewed producer still has to derive all objective and transport gates, copy and
   frequency comparisons, eight gauge replays, an SVD replay, and a
   difference-in-differences replay.  Comparator directions and point-versus-bootstrap
   semantics must be frozen before final rows are opened.
5. **The context-free frontier is promising but not a whole-model explanation.**  The
   independent large-coverage rank-256 build replicated across a disjoint fit draw and
   beat the published full-rank table builds in CE at lower modeled cost.  The currently
   uncommitted iso-cost job measured ranks 256/320/352/361, then crashed in its reporting
   tail because it indexed a nonexistent `full` entry.  The expensive numerical rows
   survived, but no new claim should be promoted until the owner's report is repaired
   and audited.

## Candidate pruning

The following ideas are not next actions:

- **Raw HOSVD/CP of the block-3 coefficients:** directly falsified as a cheap rank-256
  description by the new screen.
- **Further scalar gauge norm minimization:** the native gate factors are already
  balanced; the exact available norm gain is negligible.
- **Independent late-MLP compression:** contradicted by the MLP-only suffix experiment;
  attention cancellation is part of the interface.
- **A global 95%-energy action compressor:** would discard the conditional three-way
  circuit even though it predicts average CE well.
- **Generic SAE on weight columns:** may name directions but has no reason to preserve
  the coupled response, OOD transport, or selective removal.  It is admissible only
  inside a joint downstream-weighted objective.
- **Global Lipschitz certification now:** products of worst-case block norms will be too
  loose.  Local certificates become useful only after a passing replacement exists.

## Updated top five

### 1. Collect and factor the block-3 typed response tensor

Collect the exact four polarized outputs—residual-residual, residual-attention,
attention-residual, and attention-attention—together with their RMS scalars and
vector-valued downstream effects on frozen documents.  Fit shared factors using
activation frequency and downstream consequence, then test untouched documents and
held-out term masks.  This has the highest expected information gain because it asks
whether the empirically used part of the dense coefficient map is small.  It is
causal, composable at an exact residual port, falsifiable, and moderate GPU cost.

The CPU coefficient screen executed this hour was the stage-zero gate for this move.
Its failure determines that the fit must be activation/consequence weighted rather
than a raw weight decomposition.

### 2. Close the 68-action final semantic reducer

Freeze the comparator for every objective gate and use document-bootstrap upper bounds
for copy/frequency non-worsening, then implement and independently replay every required
diagnostic.  This is the largest direct extraction/removal/OOD hole—currently 0/68—and
would let simplicity definitions compete on useful consequences rather than local MSE.
It is high information gain but more engineering-heavy than priority 1.

### 3. Prospectively test the sparse action spectrum at an adjacent cut

Freeze the eight stable CE Walsh terms learned at the current cut, make no coefficient
changes, and predict a new physical action cube at the neighboring cut.  Success would
turn a retrospective spectrum into a composable local transition law; failure would
show that the low-degree structure is cut-specific.  GPU cost is moderate and the test
is sharply falsifiable.

### 4. Fit a joint downstream-weighted MLP0/MLP1/MLP2 dictionary

Use shared producer-consumer response coordinates and charge real stored values and
products.  Compare against PCA/rank baselines at matched CE, OOD, extraction, and
selective-removal consequences.  This is where sparse dictionaries, hierarchical
features, or a DAG can help; fitting any one MLP or its raw weights independently is
excluded.  It is important but partly redundant until priorities 1--2 provide the
correct consequence interface.

### 5. Add local incremental-quadratic composition certificates

For a replacement that passes empirical tests, estimate local gain bounds across
RMSNorm and residual interfaces and propagate them only over the measured finite
horizon.  This can certify that an edit does not amplify elsewhere.  It is cheap CPU
math once local Jacobian/sector bounds exist, but it cannot generate the replacement,
so it is fifth.

## Resource and coordination state

At review time the shared GPU is occupied by the bilinear-quotient runner
(`bilin18_canary2.py`, about 10.3 GB).  No competing GPU work was launched.  CPU work
was used for the preregistered block-3 screen and tests.  The runner's modified logs,
canary artifacts, gate check, and uncommitted iso-cost script belong to the concurrent
workstream and were not staged or edited here.

No global explanation, storage, causal, or 68-action ledger receives credit from this
hour.  The useful progress is a fast, exact negative result: raw coefficient rank and
scale canonicalization are not the missing simplicity principle at block 3, so the
next experiment is narrowed to consequence-weighted grouped structure.

