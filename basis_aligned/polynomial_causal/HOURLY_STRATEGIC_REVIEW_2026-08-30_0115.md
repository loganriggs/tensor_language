# Hourly strategic review — 2026-08-30 01:15 UTC

## Bottom line

The strict native-model explanation has not moved: we have certified `5.348245316%` of
native stored values as removable and named `10.923302467%` of the model's causal CE.
The unexplained balance is `4.72714` nat, or `89.076697533%`, and none of the 68 desired
terminal extraction/removal/OOD actions is certified.

There are two useful developments this hour:

1. The heterogeneous lookup program is genuinely smaller and better on pooled CE, but
   its improvement is not uniform.  It improves covered inputs and worsens every
   uncovered-input role.  The new attribution experiment shows that the shallow map-rank
   cut causes the uncovered loss; the extra late-MLP table rank supplies the covered
   gain.  Thus pooled CE hid a real interface/OOD-like failure.
2. Rayleigh v2 ended in a preserved pre-open infrastructure failure, not a scientific
   negative.  A clean v3 recollection path is now implemented and tested.  It removes
   the downstream scorer from the collector's source closure and binds all immutable v2
   artifacts and absences.  It awaits an independent source audit before model access.

## What the newest computations mean

### Covered and uncovered inputs

A **covered input** has a stored table row in the compiled lookup program.  An
**uncovered input** must be reconstructed by a fallback map.  The build has two distinct
mechanisms, so averaging their CE can make one mechanism's gain hide the other's loss.

Relative to the previous build, the converged program changes uncovered-input CE by:

| role | map-rank cut only | late-table rank only | both changes |
|---|---:|---:|---:|
| skip7000 | +0.001067 | -0.000316 | +0.000734 |
| skip11000 | +0.000950 | -0.000130 | +0.000790 |
| skip1200 | +0.000868 | +0.003837 | +0.004693 |

Positive means worse.  The map cut has exactly zero effect on covered inputs, as it
should, while the late-table expansion improves covered CE by `0.003329--0.004970` nat.
The two changes therefore optimize different interfaces.  The pooled `0.003064`-nat
gain is a weighted net, not a uniformly better program.

This matters to our definition of simplicity.  A smaller program that wins on an
average but systematically damages the fallback interface is not yet a composable or
OOD-robust simplification.  The missing constraint is conditional fidelity at each
interface, not another global MSE or CE average.

### Rayleigh consequence geometry

For an MLP2 approximation error $e$, the Rayleigh experiment asks whether a quadratic
downstream metric predicts its finite CE damage.  Schematically,

$$
R(e)=e^TGe,
$$

where $G$ is induced by the downstream suffix.  If the same frozen predictor ranks
unseen errors on held-out documents, then two locally different MLP2 writes can be
considered equivalent when consumers cannot distinguish them.  That would provide a
principled consumer-relative simplicity measure and seed the consumer-common block
decomposition from the mathematical review.

V2 did not answer this question.  Its scorer stopped before opening DESIGN because the
collector authority had mistakenly frozen the future scorer source.  V3 recollects the
same registered measurements under a corrected source boundary.  No feature, response,
control, row, ridge value, null, or scientific threshold changed.

## Largest remaining gaps

1. **No validated equivalence relation at residual interfaces.**  We can compress local
   writes, but cannot yet certify which differences are invisible to all relevant
   consumers.
2. **Composition remains weak.**  The best two-background MLP2 fit explains only about
   `13.2%` of the MLP0-C512 × MLP2 interaction; independent local improvements therefore
   need not compose.
3. **Uncovered/OOD behavior is not protected.**  The compiled build's newest failure is
   direct evidence that pooled optimization can sacrifice a minority interface.
4. **Sparse descriptions are not yet sparse executors.**  MLP1's weight-action
   dictionary is strong at $k=8,32,64$, but the implementation still computes all 4,608
   native gates before selecting atoms.
5. **No terminal editable circuit.**  The four-head copy bundle is causally important,
   but its tested mean-replacement bank exceeded the collateral limit.  We still lack a
   circuit that can be extracted or removed selectively and transported OOD.

## Pruned candidate directions

- Another raw HOSVD, tensor-rank sweep, or norm balancing is low priority: all 18 native
  MLP polarization slices are full rank 1,152 and their rank-768 tails are smooth across
  the compiled layer-10 knee.
- Another context-free MLP0/2 polynomial extrapolation duplicates degree-1--3 failures
  at the physical suffix interface.
- Another pooled compiled-program parameter sweep is actively dangerous until covered
  and uncovered constraints are reported separately.
- A whole-model SAE or post-hoc semantic hierarchy is premature: it can be sparse while
  retaining the full native execution cost and has no composition certificate.
- Hankel/minimal-realization work remains deferred until the consumer/intervention panel
  is rich enough; the earlier narrow tangent panel was full-rank and split-unstable.

## Top five actions, ranked

### 1. Complete Rayleigh v3 DESIGN → frozen predictor → HELDOUT

Highest information gain and causal relevance.  It tests a consumer-relative metric on
native MLP2 errors, has a binary held-out falsifier, and can directly enable a
consequence-weighted physical replacement.  Cost is about four GPU minutes per collected
role plus cheap CPU scoring.  V3 collector implementation and `76/76` focused tests are
complete; the independent audit is running.  No model access is permitted before GO.

### 2. Test consumer-common commutant blocks on real pullback metrics

For several validated consumers, form $G_c=E[J_c^TW_cJ_c]$ and solve
$XG_c=G_cX$ jointly.  Nontrivial common projectors predict state blocks respected by all
consumers and low cross-block edit interactions.  This directly targets joint
MLP/reader simplicity and gauge invariance.  It should follow Rayleigh success; on
Rayleigh failure, first broaden the signed consumer bank.

### 3. Run the C512 × best-MLP1 × CONTINUE512 factorial

This localizes whether MLP0 compression error is transported through MLP1 or repaired
only by MLP2.  The Möbius interaction

$$
\Delta_{ABC}=L_{ABC}-L_{AB0}-L_{A0C}-L_{0BC}+L_{A00}+L_{0B0}+L_{00C}-L_{000}
$$

is a direct composition statistic.  It is more useful than fitting MLP1 or MLP2 alone
because it identifies the missing interface.

### 4. Compute the MLP1 sparse-router oracle bound

Before training a tree or DAG, choose the best $k=8,16,32$ frozen interaction atoms per
held-out position and charge the complete stored dictionary and executed products.  If
even the oracle cannot beat the current program at equal CE/interface fidelity, routing
is pruned cheaply.  If it succeeds, compare flat/tree/DAG routing using prequential MDL
and unseen-edit prediction rather than atom labels.

### 5. Repair and OOD-test the uncovered fallback contract

Restore or condition shallow fallback map rank, then evaluate covered and uncovered CE
as separate constraints on fresh document clusters.  This is lower priority for native
reverse engineering, but high priority for claiming that the compiled program is a
faithful simplification.  A pooled improvement alone no longer qualifies.

## Action executed this hour

Rayleigh v3 recovery is implemented, tested, committed, and pushed as `b2cc884d`.
The amendment freezes every v2 artifact hash and absence, uses fresh v3 namespaces, and
recollects rather than promoting the old tensor ledger.  The collector closure no longer
contains its downstream scorer; the scorer closure still contains the collector and all
predictor code.  Focused collector/scorer/metric suites pass `76/76` in `5.95 s`.

This is verified infrastructure, not a scientific outcome.  The next permitted action
is the audited v3 DESIGN collection if and only if the independent audit returns exact
source-bound GO.  Meanwhile the separate compiled-program attribution completed in
`92.6 s` and numerically assigned the uncovered loss to the map cut.

## Execution update — 01:18 UTC

The independent collector audit returned GO: 45-file collector closure, 47-file scorer
closure, `138/138` tests in `9.15 s`, audit SHA
`633ffad0e780686134833bd4fefb6b7919c29780c325bf5c1dac32f7e74e2299`, and zero
outcome access.  V3 DESIGN recollection then completed successfully in `46.505 s`:

- authority SHA `0a1f0e3da1ccfd90716e6a7cc5a6edda65e3e2b265d9f9043cf8f3ae4f523283`;
- ledger SHA `d2bd1aac9aca8f8ee0bf4f01f683716beabf5603d75be7970b4813e1518f1172`;
- receipt SHA `36365b7ab2096b68b68b2c06b65a5cc331d8ce38b15f630539be08f243d02a18`.

All model-response collection controls passed and the receipt was published last.
HELDOUT remains locked.  The DESIGN tensor is now sealed pending a separate scorer
audit; no predictor has been selected and this is still not a scientific result.

The 92.6-second attribution also passed all registered predictions.  At uncovered
inputs, the shallow map cut alone worsens CE by `0.000868--0.001067` nat across the
three roles.  The late-table rank change helps slightly on two roles but hurts
skip1200 by `0.003837`; together they reproduce the full uncovered deficit.  Restoring
map rank is therefore the direct compiled-program repair, but it does not by itself
solve fresh-document OOD validation or native-model interpretation.
