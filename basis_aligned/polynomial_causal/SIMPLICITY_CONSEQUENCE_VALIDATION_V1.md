# Operational validation of simplicity measures for bilin18

Status: prospective project-level criterion. This document authorizes no data role,
model forward, selection, or scientific promotion.

## Principle

A simplicity measure is not validated by assigning a small number to a program. It
is validated only for a named consequence that the score predicts at matched causal
fidelity. “Simpler” without a promised consequence is bookkeeping.

For programs $P$ on a common intervention and support, let $D(P)$ be held-out causal
distortion and let $C_j(P)$ be a candidate simplicity measure. The basic comparison
uses programs with matched validation distortion and asks prospectively whether the
ordering induced by $C_j$ predicts a capability on untouched data:

$$
C_j(P)<C_j(Q),\qquad D_{\rm val}(P)\simeq D_{\rm val}(Q)
\quad\Longrightarrow\quad
Y_j(P)>Y_j(Q).
$$

The consequence $Y_j$ must be named before observing it. A measure may validate for
one consequence and fail for another.

## Measures, promises, and falsifiers

| simplicity measure | legitimate promise | operational validation | what it does not establish |
|---|---|---|---|
| standalone structural parameters / serialized bytes | smaller storage and independently executable replacement | zero native calls; include every producer, basis, index, decoder and shared dependency; measure artifact bytes | semantic meaning, edit locality, or OOD by itself |
| runtime operations, memory, sequential depth | cheaper execution | benchmark latency, memory and energy on fixed batches | causal understanding |
| gauge-quotiented dimension | representation-independent structural size and identifiability | exact function-preserving gauge scrambles leave the score unchanged; sample requirement tracks intrinsic rather than raw factor dimension | generalization without norm/precision assumptions |
| quantized or prequential MDL bits | fewer data-dependent bits and better statistical efficiency | frozen coding rule; held-out prequential codelength and fit-size/doubling curves | editability or compositionality |
| sparse typed program graph | localized dependencies and predictable composition | cut/keep and interchange interventions; unclaimed edges inert; joint effects predicted before measurement | compact storage if node functions remain large |
| low tensor rank / polynomial degree with bounded norms | tractable algebra and certifiable response bounds | exact reconstruction identities; held-out Taylor/remainder or Lipschitz bounds contain observed interventions | semantics or causal sufficiency by rank alone |
| causal interface dimension | manipulable sufficient state | same-forward transport, interchange interventions, OOD teacher KL, and selective edits reproduce native responses | cheap production unless its producer is also priced |
| natural-language or symbolic semantic description | human/model-level prediction of behavior | a blinded simulator predicts activations and intervention outcomes above shuffled/difficulty controls | executable compression unless compiled |

The repository's current `params/nat` score is therefore called **structural
efficiency**, not literal MDL. It is a valid same-grammar resource comparison when all
factors are counted, but it does not by itself certify interpretability.

## Consequence suite for every proposed frontier point

Every candidate promoted beyond discovery must carry the following scorecard:

1. **Independence.** Native calls to every claimed replaced component are exactly
   zero. The artifact contains or transitively binds every required producer.
2. **Common causal distortion.** Suffix KL and final CE are measured on identical
   rows, supports, interventions, and denominators. Local MSE is auxiliary.
3. **Statistical consequence.** Report fit-size and data-doubling stability, then an
   untouched document split and a second corpus or code distribution. A compression
   claim that merely memorizes fit rows fails here.
4. **Gauge consequence.** Refactorization or internal basis rotation cannot change
   complexity, predictions, selection, or intervention results.
5. **Program consequence.** Before running the joint arm, predict the sign and size
   range of composition from the typed graph. Score the prediction error; do not
   retrofit an interaction story.
6. **Edit consequence.** Intervene on one declared variable. The proposed program
   must predict both the target response and collateral response outside declared
   descendants. Locality is measured, not inferred from sparsity.
7. **Compute consequence.** Benchmark latency, memory, and operation counts against
   the native components and competing frontier points.

A point may remain on a storage/fidelity frontier while failing editability; it must
then be labeled a storage compression only. “Simpler tensor program” is reserved for
points that validate at least independence, causal distortion, gauge invariance,
program composition, and one of statistical or edit consequences.

## Validating the definitions themselves

Once a common bank of candidate programs exists, the simplicity definitions become
competing predictors. Across sites, ranks, grammars, and seeds:

1. condition on validation causal distortion;
2. use each $C_j$ to rank the candidates without seeing final consequences;
3. measure its held-out rank correlation and paired prediction accuracy for the
   promised $Y_j$;
4. compare against raw parameter count, random rankings, and shuffled site labels;
5. cross-validate by holding out entire sites or grammar families, not merely rows.

If gauge-quotiented dimension predicts data requirements better than raw parameter
count, or typed-graph sparsity predicts edit collateral better than MDL bytes, that is
evidence the corresponding definition captures a real kind of simplicity. If it does
not, the definition is pruned regardless of mathematical elegance.

## Immediate use

The middle-feature curve currently validates only a fixed-grammar structural-efficiency
claim: after factor-complete pricing, every arm remains Pareto-nondominated, while both
total and successive marginal params per recovered nat worsen with $k$. The corrected
held-out replication is complete: the $k=512$ gain over $k=0$ is $+3.675$ percentage
points on `skip7000` and $+3.811$ points on `skip11000`, with paired-bootstrap 95%
intervals $[3.514,3.841]$ and $[3.671,3.957]$. This validates portability across a
second FineWeb document split of the *incremental native-feature return*. It does not
validate OOD generalization, executable compression, editability, or semantic simplicity.

The early-MLP suffix L/R/T experiment is the first planned consequence-tested
frontier slice. L and R have matched program size but different objectives; T adds a
priced typed edge $p_0A$. Their suffix, transport, OOD, composition, gauge, and edit
tests directly ask whether the extra structure buys the capabilities it claims.

## Learning and then optimizing a simplicity rule

Solved circuits may supervise a small predictor from candidate measures to useful
consequences. This is an admissible bootstrap only under a nested family-level split:

1. **teaching circuit families** fit the measure-to-consequence predictor;
2. **development circuit families** select its inputs, hyperparameters, search
   procedure, and acceptance thresholds;
3. **sealed confirmation circuit families** remain unopened until the rule is frozen
   and has been used to construct a new candidate program.

Each circuit family also retains disjoint program-fit and consequence-test data.
At least one intervention type and one composition partner are held out by kind, not
only by row. A failure on sealed families is recorded; changing the rule creates a new
generation that requires a new sealed set.

For consequence $j$, compare candidate pairs at matched validation causal distortion:

$$
A_j = \Pr\!\left[
 \operatorname{sign}(S_j(P)-S_j(Q))
 = \operatorname{sign}(Y_j(P)-Y_j(Q))
 \;\middle|\;
 |D_{\rm val}(P)-D_{\rm val}(Q)|\le\epsilon
\right].
$$

Prospective pairwise accuracy $A_j$, held-family rank correlation, and calibration are
the evidence. A better optimized surrogate score is not evidence by itself. Separate
predictors are retained for OOD transport, extraction, selective editing, and
composition until untouched families justify a scalar combination.

Once frozen, $S_j$ may guide program search as a constraint or regularizer. The search
may use only program-training data and the frozen predictor. The resulting candidate
is evaluated against ordinary rank, complete bytes, sparse-edge, and randomized-search
baselines at matched causal distortion and literal price.

Because optimization actively searches for loopholes in $S_j$, candidates outside the
support of its teaching/development examples require either rejection or a prespecified
uncertainty penalty. Adversarial candidates must include equal-price gauge rotations,
shuffled circuit labels, low-local-error but causally wrong programs, inactive sparse
edges, duplicated modules presented as reuse, hidden native calls, and collateral moved
outside the measured target set. Reuse is credited only when identical parameters and
intervention semantics serve multiple held-out consumers.

The first cheap screen may use a chronological split of the existing ledger, with whole
module and grammar families excluded from training. Because those historical experiments
were adaptively selected, this is only a filter. Any positive claim requires a newly
registered prospective family split and sealed consequence labels.
