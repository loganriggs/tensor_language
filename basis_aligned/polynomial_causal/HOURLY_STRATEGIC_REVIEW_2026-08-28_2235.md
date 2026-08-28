# Hourly strategic review: use CE to choose the next tensor cross

**Time:** 2026-08-28 22:35 UTC

**New CPU action:** document-bootstrap stability audit for maximum-volume crosses

**Artifacts:** `cut_cross_bootstrap_stability.py`,
`cut_cross_bootstrap_stability_results.json`

## Bottom line

The next tensor experiment should be designed from cross-entropy, not top-1
accuracy. On the completed layer-5 mask grid, the rank-four CE maximum-volume pivot
is selected in 1,999 of 2,000 document bootstraps. The rank-four top-1 pivot is
selected in only 860 of 2,000. CE therefore supplies a stable experimental-design
rule; top-1 remains a useful outcome but is too discontinuous to choose the masks.

This advances the tensor-cross direction from “the full matrix looks compressible”
to a statistically testable plan. It still earns no model-explanation credit because
the source grid is fully revealed. The next credit-bearing step must freeze these
selection rules and predict untouched outcomes on a fresh mask family.

## Honest fraction explained

There is no single valid percentage because the project ledgers use different
denominators.

| Question | Current answer | What remains |
|---|---:|---:|
| Does every top-level module have a structural surrogate? | 36/36 | Meaning, minimality, and faithful composition |
| How much complete-program storage is consequence-certified removable? | 5.3481% | 94.6519% lacks that certificate |
| How much behavior has an older human-readable account? | 32.1% plus or minus 6.4% | Most measured behavior is unnamed |
| How much strict named causal CE headroom is recovered? | 10.923% | 4.72714 nats, or 89.077%, remains unnamed |
| How many final early-MLP/suffix action rows are scored? | 0/68 | The final causal-response tensor is absent |

The strictest answer to “have we reverse engineered the whole model?” remains no.
We can execute structural substitutes everywhere, but we do not yet have a smaller
program that predicts their interactions, OOD behavior, and selective edits across
the complete model.

## What changed since the last review

### The per-token class boundary is now clean

The full-rank length-one program matches the model-defined per-token/length-one
ceiling on covered positions to numerical precision on all three roles. The old
roughly 0.55-nat discrepancy was a rank-64 build, not a full-rank program. Thus the
remaining roughly 2.74 nats from the per-token ceiling to the live model are chiefly
the price of contextual computation, especially attention. Better fallback rows or
more per-token rank cannot close that class boundary.

The active `rank_to_ceiling.py` run was testing the rank ladder directly. Its third
attempt reproduced the full-rank ceiling and then failed at 22:34 UTC while building
rank 256: CUDA had only 248 MB free and a new 396 MB allocation failed. The source
tries to release the previous bank, but the local `hks` list still contains hook
closures whose `full_rows` cells retain all 36 bank tensors. Assigning `bank = None`
therefore does not free the roughly 8.3 GB bank before `build()` allocates another.
The precise repair for the owning agent is to delete `hks` and `bank` after each
rank's scoring loop, then empty the cache before building the next rank. I did not
alter its untracked source, queue, or result namespace.

### Composition failures are now the central evidence

- Nine of twelve early sites prefer the fitted-context table in isolation, but the
  resulting mixed whole program is worse than either uniform program.
- The fitted-table advantage across compiled depth oscillates
  `+1.6, -19.9, +2.1, -1.4` percentage points. It is neither monotone in depth nor
  reducible to a scalar rule.
- MLP2 is harmful alone but beneficial after MLP0 and MLP1 restoration in the exact
  factorial experiment.
- On the layer-5 mask grid, rank at most two predicts large total costs but fails on
  the anchored interaction itself, especially for CE.

These are different views of one gap: we do not yet have the small state passed
between components that makes their substitutions compose.

### The joint early-MLP PCA experiment is prepared but still NO-GO

Concurrent untracked sources propose a useful cube: fixed rank-64 output projections
for MLP0/MLP1, followed by exact native MLP2 restoration. A prior conceptual audit
found the causal question coherent but the execution lifecycle incomplete. It is an
oracle-interface test because projected corrections still call native MLPs; even a
pass would not be an executable compressed program. Its exact blockers are committed
source closure, full parent/row provenance replay, hash-after-load checks,
transactional publication, post-forward closure, and runtime intervention tests.
Those files are owned by the concurrent agent and were not modified here.

### The 68-action final interface is physically assembled but semantically blocked

The model/row side, response reductions, 18 consumer observations, action plans, and
one-shot final callback exist. The missing component is a reviewed producer of
`FinalDecisionReplayEvidence`. It must compute rather than invent:

- all fifteen objective gate booleans;
- the six observational transport gate booleans;
- eight full scored-row gauge replay difference vectors;
- the SVD replay difference;
- the difference-in-differences replay difference; and
- a scalar numerical payload with explicit copy and nine-bin frequency comparisons.

The preregistration says copy worsening and every nonempty frequency-bin worsening
must be at most 0.01. What remains ambiguous is whether the final producer uses only
pooled point differences or requires document-bootstrap upper bounds for those two
gates, and exactly which program arm is the comparator in every objective gate.
Silently choosing either convention after seeing final rows would invalidate the
final role. This is a genuine semantic blocker, not a missing tensor plumbing stub.

## CPU action executed: bootstrap stability of tensor-cross pivots

For each of 2,000 document bootstrap resamples, I reconstructed the complete 7 by 7
anchored interaction matrix and exhaustively selected the largest-determinant square
cross at ranks two, three, and four. The audit also measured pivot condition numbers
and the error from freezing the original point-estimate pivot.

### Results

| target | rank | point pivot selected | distinct winners | median / 95% cross NRE | 95% pivot condition |
|---|---:|---:|---:|---:|---:|
| CE | 2 | 100.00% | 1 | 0.4741 / 0.4904 | 1.65 |
| CE | 3 | 82.05% | 2 | 0.3074 / 0.3291 | 3.66 |
| CE | 4 | 99.95% | 2 | 0.1610 / 0.1737 | 8.11 |
| top-1 | 2 | 43.35% | 5 | 0.2913 / 0.3621 | 3.18 |
| top-1 | 3 | 55.95% | 7 | 0.1436 / 0.1837 | 9.57 |
| top-1 | 4 | 43.00% | 14 | 0.0918 / 0.1437 | 28.86 |

“Cross NRE” is the Frobenius norm of the reconstruction error divided by the norm
of the interaction matrix. “Pivot condition” measures amplification from inverting
the selected intersection; smaller is safer.

The result is unusually decisive:

- CE rank four has a stable discrete pivot even though its winner/runner-up volume
  margin is not enormous. Sampling noise almost never changes the winner.
- Top-1 has a slightly smaller retrospective error but a highly unstable pivot and
  much worse rank-four conditioning. This is consistent with argmax accuracy being
  discontinuous under small logit changes.
- Rank two is stable for CE but leaves almost half the CE interaction norm. It is a
  stable wrong model, so it should not be retried.

Three focused tests pass: exact rank-two recovery has one stable pivot and numerical
zero error; a controlled changing-pivot example reports the literal selection
frequency; malformed tensors fail closed.

## Candidate actions and pruning

### 1. CE-led prospective tensor-cross assay on a fresh physical mask family

**Expected information gain:** very high. It tests whether a compact interaction
state predicts genuinely untouched compiled programs.

**Causal relevance:** medium to high. CE is the design currency, but top-1 and causal
response coordinates must be reserved outcomes.

**Whole-model composability:** high if the same cross transfers to another cut;
otherwise the hypothesis is falsified locally.

**Falsifiability:** excellent. Freeze the CE rank-three/rank-four pivots and additive,
ALS, and hereditary-Möbius baselines before measuring reserved cells.

**GPU cost:** moderate and bounded, comparable to the completed 64-mask wave.

**Why first:** it directly addresses the observed composition failure, exploits the
network's tensor ordering, and now has a stable selection rule. The bootstrap audit
executed in this review is the required go/no-go precursor. It passes for CE rank
four and rejects top-1-led selection.

The fresh mask family should emphasize MLP0/1/2 prefixes and contextual suffixes,
because that is where exact nonadditivity and the 2.74-nat class boundary meet. The
measurement must remain separate from the already revealed layer-5 grid.

### 2. Close the 68-action semantic producer, then use vector-valued cross outcomes

**Expected information gain:** very high. It exposes CE, code/logit response,
consumer, copy, and frequency consequences for the same interventions.

**Causal relevance:** highest of the candidates.

**Whole-model composability:** high at the early-MLP-to-suffix interface, though not
yet all 36 sites.

**Falsifiability:** excellent once comparator and interval semantics are frozen.

**GPU cost:** high but already engineered; the marginal cost is the final execution.

**Blocker:** the semantic choices above must be resolved and implemented before the
final role opens. The safe independent work is to specify the reducer against
synthetic reductions; the final action itself remains NO-GO.

### 3. Repair and run the joint MLP0/MLP1 projection plus exact-MLP2 cube

**Expected information gain:** high for whether the discovered 64-dimensional
interfaces retain causal value jointly.

**Causal relevance:** high, with a strict oracle-only interpretation.

**Whole-model composability:** directly tested for MLP0/1/2, but native MLP calls mean
it is not yet a compiled program.

**Falsifiability:** good; same-run exact denominators and heldout row bootstrap are
already defined.

**GPU cost:** moderate to high. Do not launch until the seven audit blockers close.

**Redundancy:** it extends rather than repeats singleton PCA and exact factorial
results because the projected joint arms have not been measured.

### 4. Fit downstream-Hessian-weighted low-rank corrections jointly

For token \(t\), the locally CE-optimal constant obeys

\[
q_t^*=\mathbb E[H\mid t]^\dagger
      \mathbb E[H z-g\mid t],
\]

where \(z\) is the native write, \(g\) the downstream CE gradient, and \(H\) a
downstream curvature approximation. This prices errors by what later computation can
distinguish rather than Euclidean output distance.

**Expected information gain:** high if fitted jointly across MLP0/1/2; low if done
site by site. **Causal relevance:** medium. **Composability:** explicit in a joint CE
objective. **GPU cost:** moderate. **Falsifier:** it must beat equal-cost PCA and
rank-table baselines on untouched CE and retain MLP2's conditional benefit. Defer
until the current rank-to-ceiling curve and joint PCA cube settle, to avoid redundant
rank sweeps.

### 5. Turn a passing pair of adjacent crosses into a minimal action realization

If two physical cuts have stable low-rank consequence blocks, shifted block-Hankel
factorization can test whether one small latent state and layer-indexed transition
maps predict unseen action strings. This would yield the most directly executable
small program.

**Expected information gain:** very high conditional on the crosses. **Causal
relevance and composability:** very high. **Falsifiability:** transfer the frozen
state basis across adjacent cuts. **GPU cost:** low to moderate after the response
bank exists. It is fifth now only because running it before two passing crosses
would repeat the earlier underidentified Hankel mistake.

## Explicitly pruned this hour

- **Another rank-one/rank-two interaction fit:** CE rank two is bootstrap-stable but
  leaves about 47% normalized error and already failed prospective completion.
- **Generic low-rank plus sparse exceptions:** rank-two residual energy is diffuse,
  not concentrated in a few cells.
- **Independent fitted tables or a scalar depth rule:** directly fail whole-program
  composition.
- **Standalone weight SAE/PCA/HOSVD:** no untouched consequence prediction and
  redundant with completed local studies.
- **Norm minimization as a rank reducer:** useful for gauge conditioning only; it
  cannot change the contracted tensor's HOSVD spectrum.
- **Generic token-prefix Hankel expansion:** previous splices were severely OOD and
  did not reveal a compact predictive state.
- **More local MSE optimization:** MLP4 is the direct counterexample—closer local
  output reconstruction produced worse downstream behavior.

## Coordination and immediate continuation

- `rank_to_ceiling.py` remains owned by the concurrent lane. Its third OOM and the
  hook-closure retention diagnosis are preserved above and on `AGENT_BOARD`.
- The untracked joint early-MLP PCA sources remain owned by their concurrent agent
  and are still NO-GO under the recorded audit.
- This review touched neither set.
- The new CPU stability code and result are isolated in
  `basis_aligned/polynomial_causal/`.

Immediate continuation after this review: freeze the fresh CE-led mask registry,
rank-three/rank-four cross predictors, untouched cells, document-bootstrap gates,
and matched additive/Möbius baselines before any new model outcome is observed. Do
not use top-1 to select the pivot; score it only after the registry is frozen.

No whole-model ledger value moves from this discovery-only action.
