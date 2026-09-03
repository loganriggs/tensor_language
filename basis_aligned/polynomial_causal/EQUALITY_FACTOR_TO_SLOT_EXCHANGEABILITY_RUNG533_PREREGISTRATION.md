# Rung 533 preregistration: can either source factor fill either target factor slot?

**Registered:** 2026-09-03 12:46 UTC

**Owner:** Codex

**Status:** CPU contract implemented; GPU implementation not yet authorized

## Decision this experiment resolves

Rung 532 showed that all four observed source-factor to target-slot substitutions met its basic downstream-transfer
bars on the 62-circuit census, while the two tested key-permuted controls failed. The registered cross-branch
identification claims nevertheless failed because their same-branch controls worked equally well.

This suggests two incompatible explanations:

1. **Branch-exchangeable equality family:** either factor from source head `L8H3` carries a key-dependent equality
   quantity that can fill either score-factor slot of target head `L8H4` after a fixed scalar rescaling.
2. **Incomplete-control accident:** one or both same-branch substitutions looked useful on the rung-532 corpus but
   would fail a matched key-permutation comparison or fail on a different text distribution.

Rung 533 distinguishes them. It is a circuit-grouping and held-out-prediction test, not a rank or compression test.

## Exact four-way intervention

Write the source factors as `(a,b)` and target factors as `(c,d)`. The target equality score contribution uses
the elementwise product `c*d`, followed by the target head's unchanged value/output path. Freeze all scales from
rung 531; do not fit them again:

```text
a -> c : -1.268044102615207     b -> c :  1.227983240318439
a -> d : -0.8533769036200292    b -> d :  0.6995515454196305
```

For source factor `s`, target slot `t`, its native companion `u`, and frozen scale `lambda`, evaluate:

```text
substitution:          (lambda*s) * u
matched key control:   (lambda*reverse_keys(s)) * u
```

`reverse_keys` reverses the valid causal key prefix separately for each query position. It preserves the factor's
scale and value distribution while breaking which earlier token is paired with each query. Every one of the four
mappings gets its own matched control. Also run `native=c*d`, `absent=0`, and the frozen donor-product control
`-1.0785167862928777*(a*b)`.

Run all arms with the donor equality term present and absent. This checks whether the conclusion depends on the
source and target contributions coexisting downstream.

## Data and measurements

Use the already hash-frozen but cross-corpus roles:

- 192 `final_natural` documents;
- 192 repository-disjoint `ood_code` documents.

These roles were opened by earlier, different equality experiments, so this is prospective for the new four-way
interventions but is not described as a pristine program-wide holdout. Split each role into documents `0:96` and
`96:192`. No rung-532 census row is reused.

For each arm and background, compute cross-entropy loss per token and aggregate it into:

- each document's copy-positive positions, producing a held-out downstream-effect vector across documents;
- the global copy-positive effect;
- matched copy-negative positions; and
- all remaining positions.

For an arm, the per-document effect is `CE_absent - CE_arm`; compare it with `CE_absent - CE_native` using cosine
and relative error. A document enters the vector only when it has at least one copy-positive position. No scale is
fit to the effect vector.

## Frozen bars

A substitution passes one context when:

- effect cosine is at least `0.85`;
- relative error is at most `0.60`;
- global copy-positive recovery is in `[0.65, 1.40]`;
- its cosine exceeds its own key-permuted control by at least `0.15`;
- absolute mean CE change from native is at most `0.01` nat on matched negatives and at most `0.01` nat on all
  remaining positions.

A mapping passes only if all eight contexts pass: two corpora times two halves times two donor backgrounds.

## Registered predictions

### A — valid physical instrument

Native and analytical replay logits agree exactly; target-product reconstruction is exact in deployed precision;
every intended edit has nonzero RMS; all document/task supports and model-forward counts reconcile; row and source
hashes match.

### B — product-level positive control

The frozen donor-product control meets the cosine, relative-error, positive-recovery, and off-target bars in every
context. This verifies that each corpus can recognize the already established product-level action.

### C — both source factors can fill target slot `c`

Both `a -> c` and `b -> c` pass all eight contexts against their separately matched key controls.

### D — both source factors can fill target slot `d`

Both `a -> d` and `b -> d` pass all eight contexts against their separately matched key controls.

### E — branch-exchangeable downstream family

A through D pass. This identifies a four-way operational equivalence for this source/target pair: later model
computation treats either source factor as an adequate input to either target score slot on natural text and code.
It does not claim that the raw factor matrices are equal, and it does not yet generalize beyond this pair.

### F — donor-background stability

For each mapping, its effect vectors with the donor term present and absent have cosine at least `0.90` in every
corpus half. This tests whether the mapping remains recognizable when the otherwise redundant source action is
removed.

## Opposing outcomes

- **Fixed pairing:** A and B pass, only the two cross-branch mappings pass. Keep the original swapped pairing.
- **Target-slot asymmetry:** exactly one of C or D passes. Treat the slots as distinct downstream variables.
- **Product only:** A and B pass, but neither C nor D passes. Close individual-factor portability and keep only the
  complete bilinear product as the circuit unit.
- **Invalid:** A or B fails. Repair or reject the measurement; do not interpret factor identity.

No threshold will be changed after outcomes are opened. A positive result remains an identification result for one
head pair, not adoption of a smaller executable model; broader head grouping, joint installation, and literal price
would still be required.
