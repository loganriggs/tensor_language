# Three-hour mathematical review — 2026-09-02 19:08 UTC

## Goal and present boundary

The goal remains an executable tensor program whose pieces generalize out of distribution, can be extracted, can be
removed without harming unrelated computations, and recur compositionally across examples/modules. Rank, storage,
quantization, and reconstruction are not circuit criteria.

Rung501 leaves one calibrated cross-head action, `L5H5 score -> L8H4`, and no new directed edge. Rung502 attempted to
expand the resulting MLP9 response into the exact quadratic products among20 named residual sources. The quadratic
algebra and calls are exact, but the first receipt is invalid: the early-absent parent used the wrong reference, and
the explicit BF16/RMS numerical source contributes10.9--13.3% of the small response versus the frozen2% ceiling.

## The mathematical object and the real obstruction

For a fixed normalized input `z=sum_s z_s`, MLP9 is exactly

`q(z)-bias = sum_{s,t} Down[(Left z_s)*(Right z_t)]`.

This source-pair tensor is much better posed than a rank reduction: it says which actual earlier writes are multiplied
and gives a literal output vector for every pair. Internal Left/Right rescalings and permutations do not change the
complete pair output.

The obstruction is one level earlier. In real arithmetic, the residual recurrence is linear and RMS normalization is
a common scalar on a fixed example, so the named writes give a canonical additive `z_s`. In deployed BF16 arithmetic,

`round(a+b) != round(a)+round(b)`.

Repeated residual mixing therefore leaves a small implementation-dependent complement. The whole MLP9 write changes
by only about10% under the score action, so a numerically tiny state complement can produce10% of that difference.
Assigning the complement to `E`, distributing it across sources, or calling it `NUMERICAL` changes individual pair
attributions while leaving the deployed function unchanged. This is a genuine source-attribution gauge, not evidence
for a semantic numerical circuit.

## Repair selected before rerun

Rung502b must preserve the first receipt and change only the invalid instrument:

1. Add the missing early-absent `late_native` trajectory. The true price is eight rather than seven forwards per
   batch:1,000 total.
2. Capture the exact deployed residual immediately before MLP9 normalization.
3. Define the19 explicit attention/MLP write sources with the registered residual-mixing coefficients.
4. Define the base `E` source as the exact remaining deployed residual. It therefore honestly means
   “embedding/skip path plus deployed recurrence roundoff,” rather than pretending the latter vanished.
5. Compare two exact normalized allocations:
   - `E-absorbed`: RMS roundoff is included in the base `E` source;
   - `proportional`: raw-recurrence and RMS complements are distributed across all20 sources in proportion to their
     squared norm at that token.
6. Each allocation must sum exactly and reconstruct MLP9. A semantic shortlist may pass only if the complete selected
   pair names and their held-out signs agree across both allocations. All old C/D thresholds stay fixed.

This does not claim either allocation is metaphysically correct. Agreement says the identified computation is robust
to the small deployment arithmetic gauge; disagreement says the source-pair explanation is not identified at this
precision and grain.

## Alternative mathematical routes considered

### 1. Float32 analytical tensor program

Run the same frozen writes through an all-float32 residual/MLP9 calculation. This removes finite-precision
non-additivity and gives the clean real-arithmetic tensor network. It is useful as an explanatory control, but it is
not the deployed model. It cannot replace a BF16 causal intervention unless the response and downstream effects agree.

### 2. Shapley allocation of residual rounding

Treat the20 sources as players in the BF16 residual-plus-normalization computation and average marginal source
contributions over orders. This is symmetric, but exact evaluation costs `2^20` states per example; sampled Shapley
would reintroduce approximation and variance before we know the coarse atlas is useful. The two-gauge falsifier is
currently more informative per GPU minute.

### 3. Direct finite source factorial

Remove selected earlier writes upstream in all four states and compute their Möbius interaction on copy CE. This is
fully causal but changes intervening attention, residual mixing, and normalization, so it does not isolate the MLP9
multiplication. It is the necessary confirmation after a local pair is selected, not a replacement for locating it.

### 4. Circuit-by-source-pair tensor factorization

Once finite effects exist, form a tensor with axes `(circuit, source1, source2, example/background)` and seek a sparse
CP/Tucker/block decomposition. Such a factorization is useful only if its factors predict held-out finite effects and
support selective removal/reuse. Tensor rank or reconstruction alone cannot choose the circuit basis; downstream
interchange/removal supplies identifiability.

### 5. Mixed Hessian screen through the whole suffix

Use Hessian-vector products to find pairs whose interaction is created after MLP9 rather than inside it. This covers
the multiple-mediators problem but is only a search heuristic. The exact MLP9 tensor should be tested first because it
is algebraically available and already has a calibrated positive/negative observation.

## Decision

The highest-information next move is the exact deployed two-gauge rung502b repair. A pass licenses the already frozen
finite group-removal test. A gauge disagreement is itself a decisive result: source-pair semantics at BF16 deployment
precision are not identifiable from this additive decomposition, and the program should move to finite source
factorials or a float32 explanatory control rather than tune rank or attribution thresholds.
