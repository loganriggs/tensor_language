# Hourly strategic review — 2026-08-30 01:45 UTC

## Bottom line

The Rayleigh experiment now has real held-out measurements, and every straightforward
point-estimate reading is strongly negative.  The exact MLP2 program was replayed
correctly, but the proposed downstream quadratic geometry neither had stable scale nor
predicted which unseen document/program errors were costly.  We therefore must not
train the proposed consequence-weighted MLP2.

There is an important protocol qualification: independent review found that the frozen
documents require a shared bootstrap/max band, but the exact cross-cell aggregation and
bootstrap realization were never completely frozen.  The post-outcome score receipt is
therefore an exactly reproducible **exploratory negative, not an authoritative
preregistered PASS/FAIL ruling**.  This design defect is preserved rather than silently
repaired after seeing HELDOUT.

The strict whole-model ledger does not move:

- certified removable native storage: `29,196,288 / 545,904,054 = 5.348245316%`;
- named causal CE: `0.57968 / 5.30682 = 10.923302467%`;
- unexplained CE: `4.72714` nat, or `89.076697533%`;
- complete extraction/removal/OOD actions: `0 / 68`.

Those percentages are deliberately conservative.  The compiled lookup program is
smaller and can beat the deployed program on average, but that is not a certificate
that we understand the corresponding native computation.

## What was computed this hour

### The object being tested

For each of three rank-512 MLP2 replacements and each of two upstream states, the
collector formed the MLP2 write error

$$
e = \text{replacement output} - \text{native MLP2 output}.
$$

The upstream states were native MLP0 and compressed MLP0-C512.  The collector inserted
scaled versions $\alpha e$ into the otherwise native suffix and measured how the final
logits and attention layers 5 and 6 changed.  It used 32 fresh HELDOUT documents that
were unavailable when the predictor was fit.

The hypothesis was that a downstream quadratic

$$
q(e) = e^T G e
$$

would predict the finite CE interaction.  Here $G$ is not a stored matrix: its action
was estimated by symmetric perturbations through the real suffix.  Intuitively,
$q(e)$ should be large when the suffix is sensitive to the error direction and small
when downstream computation is nearly blind to it.

### Terms used in the gates

- A **symmetric finite difference** compares outputs at $+\alpha e$ and $-\alpha e$.
  Dividing their difference by $2\alpha$ estimates the suffix derivative in direction
  $e$ while cancelling leading even-order error.
- The **categorical Fisher quadratic** is the local output-distribution sensitivity
  of the changed logits.  It predicts the teacher KL as
  $\tfrac12\alpha^2 q_{\rm logit}$ when the perturbation is sufficiently local.
- **Teacher KL** measures how much the model's complete predicted token distribution
  changes, not only whether the top token changes.
- **Spearman correlation** asks whether two quantities rank the documents similarly.
  A value of 1 is identical rank order, 0 is no monotone ordering, and -1 is reversed.
- The **finite interaction** is the change in an MLP2 replacement's CE effect when
  MLP0 is changed from native to C512.  It measures composition, not MLP2 in isolation.
- **MSE gain** is the fractional reduction in squared prediction error relative to a
  baseline.  Positive is better; zero means no predictive improvement.

### Held-out point estimates under the explicit saved reduction

| gate | required | observed | result |
|---|---:|---:|---|
| $1/16$ versus $1/8$ scale agreement | disagreement $\leq 20\%$ and Spearman $\geq 0.8$ | `45.27%`, Spearman `0.9186` | fail |
| Fisher prediction of teacher KL | ratio in `[0.8, 1.25]`, Spearman $\geq 0.6$ | ratio `1.8268`, Spearman `0.8996` | fail |
| full frozen predictor versus local MSE | at least `25%` MSE gain and Spearman $\geq 0.5$ | `0.063%` gain, Spearman `0.1722` | fail |
| full predictor versus final logits only | at least `10%` MSE gain | `-0.679%` | fail |
| three mean finite interactions | correct signs and error $\leq 0.0025$ nat | all signs correct; errors `0.00130`, `0.00185`, `0.00148` | pass |
| randomized controls | fail their predictive gates | both fail | pass |

The endpoint implementation passed exactly: injected $\alpha=1$ writes and direct
physical replacements had bit-identical final logits and attention-5/6 writes.  Thus
this is not a hook or replacement bug.

The mean interaction gate passes because all three observed means are positive and
fairly close together.  That does **not** rescue the hypothesis.  A nearly constant
prediction can estimate three aggregate means while failing to predict document-level
variation.  The decisive predictor Spearman is only `0.1722`, and its MSE is essentially
the same as local error.

Exploratory receipt: `mlp2_error_rayleigh_v4_heldout_score_receipt.json`.  Independent
audit status is NO-GO for treating it as the authoritative preregistered decision.

Timing was `61.41 s` inside the collector and `94.04 s` wall time including model load
and artifact checks.  CPU scoring took about `2.5 s`.  The independent pre-open source
audit ran `138/138` tests in `9.50 s` and did not access HELDOUT.

## What the negative result means mathematically

The safe operational conclusion is to reject this feature bank and finite-difference
scale as a training objective for MLP2 errors.  In particular, separate scalar norms of
attention-5 and attention-6 responses did not add predictive value beyond final logits
under the saved reduction.  This is not the missing formal bootstrap/max-band ruling.

It does **not** show that consumer-relative geometry is impossible.  Three important
pieces were discarded by this pilot:

1. attention response **sign and direction** were reduced to one nonnegative norm;
2. nonlinear curvature between $\alpha=1/8$ and the physical endpoint $\alpha=1$ was
   omitted;
3. the final-logit Fisher is a teacher-KL curvature, not the full true-token CE Hessian,
   whose suffix second derivative can matter.

So the next mathematical version, if pursued, must use a richer signed consumer bank or
explicit finite interactions.  Repeating the same scalar Rayleigh features with more
documents would be poor ROI: the held-out margins are much too large to attribute to
sampling noise alone.

## Other new result: the compiled fallback failure persists at higher coverage

The separate high-coverage run used 16,110 covered inputs.  Cutting the fallback map
rank still worsened uncovered-input CE on every role by `0.001024--0.001791` nat while
having exactly zero covered effect.  Raising late table ranks improved covered CE by
`0.006691--0.009247` nat, but its uncovered effect varied: it helped skip7000 by
`0.001498` and hurt skip11000/skip1200 by `0.001869/0.003655`.

Thus the earlier problem was not just too little coverage.  Table capacity and fallback
map capacity are separate contracts.  Pooled CE still favors the converged program by
`0.005566--0.007970` nat because covered rows dominate, while two uncovered roles get
worse.  This strengthens the rule that simplicity must be evaluated conditionally at
each interface.

This high-coverage artifact is currently shared, uncommitted work and is not used to
move the strict native ledger.

## Largest remaining gaps

1. We lack a validated residual-state equivalence relation: reconstruction error is
   measurable, but downstream indistinguishability is not yet predictable.
2. We do not know whether independently simpler MLP0, MLP1, and MLP2 programs compose,
   or which layer repairs which upstream approximation.
3. MLP1 has sparse descriptive atoms but no cheap router; selecting a few atoms after
   computing all 4,608 native gates is not an executable simplification.
4. Compiled fallback behavior is not protected on minority/uncovered inputs or fresh
   OOD distributions.
5. No named circuit yet passes extraction, selective removal, and OOD transport; the
   terminal ledger remains `0/68`.

## Candidate pruning

- **Prune the current Rayleigh objective as a training branch.**  Three of four primary
  point gates fail by large margins on untouched documents, and the scorer audit is
  NO-GO.  No consequence-weighted MLP2 fit is licensed.
- **Prune another scalar attention-norm feature sweep.**  It discards exactly the signed
  response information that could distinguish consumer pathways.
- **Defer raw HOSVD/norm balancing.**  exact polarization slices are smoothly full rank
  and existing rank tails do not identify the compiled layer-10 transition.
- **Defer whole-model SAE or semantic clustering.**  Neither gives a composition or
  executable-cost certificate by itself.
- **Do not optimize pooled compiled CE alone.**  The covered/uncovered reversal is now
  replicated at larger coverage.

## Top five actions after pruning

### 1. C512 × best-MLP1 × CONTINUE512 factorial

Run every combination of the three replacements on the same rows and compute their
Möbius interaction.  This has the best ratio of causal relevance to GPU cost because
the intervention machinery and component programs already exist.  It directly says
whether MLP1 transports the C512 error and whether MLP2 compensates for it.  It is
composable, falsifiable, and does not duplicate the failed Rayleigh predictor.

### 2. Signed consumer bank, then consumer-common blocks

Retain response direction by projecting attention-5/6 and final-logit changes onto a
small, frozen signed basis rather than replacing each field by its norm.  If these
features predict unseen finite interactions, form the common commutant of the resulting
consumer pullback forms.  Its projectors would define gauge-invariant state blocks
respected by multiple consumers.  This is higher mathematical upside than action 1 but
requires a new audited design and is therefore slower.

### 3. MLP1 sparse-router oracle bound

Allow an oracle to select the best `k=8,16,32` existing atoms per held-out position,
then charge both dictionary storage and executed products.  This cheaply answers
whether a sparse-per-datapoint DAG/router could possibly beat the current program
before training one.  Failure prunes an entire family; success licenses flat versus
tree/DAG routing tests under prequential MDL.

### 4. Repair the fallback contract and validate on fresh clusters

Restore fallback-map rank or condition it on coverage, while reporting covered and
uncovered CE separately.  This is immediately useful for the executable compiled
program and falsifies whether the pooled gain transports to new documents.  It ranks
below native-layer composition because it explains less of the original model.

### 5. Expand downstream circuit endpoints

Add several late, behaviorally distinct consumers—such as capitalization, induction,
and token-copy endpoints—to the early-layer assay.  Early features should be defined by
which of these consumers they causally affect.  This can turn an underdetermined MLP0/1
decomposition into a jointly sparse writer/reader factorization, but only after each
endpoint has a clean intervention and specificity test.

## Action executed

The exact-source HELDOUT audit returned GO at commit `a1f8e887` with audit SHA
`958197eebabe05e1db325567740c6d71e086cef20ac3bd74372a134543f7a3a6`.
The one-shot 32-document HELDOUT collection then completed, with ledger SHA
`da39efdca9c862878b146417e08ca51c1130da129239b17271286c12c43f053a`.
The frozen predictor was applied without refitting and produced the negative point
estimates above.

Independent scoring audit then returned NO-GO for an authoritative decision.  The
collector and frozen-predictor chains are valid, endpoint replay and the 688-forward
call census are exact, and the score receipt reproduces.  But the reporter omitted the
registered shared document bootstrap/max band; its average-over-six-cells and flattened
96-row reductions were not fully frozen; and the post-outcome scorer lacked a pre-run
source authority and receipt-last transaction.  Those defects cannot be repaired
retrospectively without changing the epistemic status of the experiment.

This is genuine held-out evidence plus a genuine analysis-protocol failure, not merely
infrastructure.  The margins are enough to prevent spending a much larger GPU budget on
the unsupported objective, but not to claim a formal preregistered rejection.  The next
safe experiment should be action 1; action 2 should be preregistered in parallel only
after specifying signed projections, cross-cell reductions, bootstrap bands, and finite
held-out consequence gates completely.
