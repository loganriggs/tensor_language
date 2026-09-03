# Rung 534 preregistration: shared equality signal plus target-specific context correction

**Registered:** 2026-09-03 13:06 UTC

**Owner:** Codex

**Status:** CPU algebra and parent localization complete; GPU implementation not yet authorized

## Why this changes the object

Rung 533 cannot identify a fully interchangeable four-factor family under its frozen rules because the complete
source-product control passes only 4/8 cross-corpus contexts. The failure is structured, not a dead intervention.
In the two `ood_code`, donor-absent halves, the source product reproduces the target's copy-positive effect with
cosines `0.9898/0.9915`, relative errors `0.218/0.174`, and effect recoveries `0.875/0.870`, but differs from the
native target by `0.0249/0.0233` nat on matched copy-negative tokens. Every key-permuted factor control fails.

This matches the proposed possibility that two heads compute the same useful copy/equality relation but reject
noise differently. Rung 534 therefore stops asking whether whole factors are interchangeable. It splits the
gauge-invariant score product into a cross-head shared equality signal and a target-specific residual, then asks
whether the residual is an independently acting circuit component or only an interaction with the shared signal.
This targets cross-head grouping, within-head splitting, OOD prediction, selective intervention, and composition.
It is not a rank or compression experiment.

## Exact score decomposition

Let source head `L8H3` have complete equality-score product `P_s=a*b`, and target head `L8H4` have product `P_t=c*d`.
Freeze the rung-531 least-squares product scale `gamma=-1.0785167862928777`. Define

```text
shared score       S = gamma * P_s
private correction R = P_t - S
native target      P_t = S + R
```

This is at the complete product level, so branch rescalings `a -> q*a, b -> b/q` do not change either `P_s` or `S`.
The scale is fixed from rows `0:500`, before rung-533 outcomes. The CPU contract verifies `P_t=S+R` to floating-point
error below `4e-16` on an independent synthetic tensor.

This differs from rung 464. Rung 464 exchanged 19 later attention/MLP residual writes induced by two complete
matcher implementations. Rung 534 splits the layer-8 score product itself, before the target head's unchanged
value/output path.

## Physical arms

At the target equality edge, contract each score pattern through the target head's unchanged support and
value/output tensor. Run:

- `native`: `P_t`;
- `absent`: `0`;
- `shared`: `S`;
- `private`: `R`;
- `shared_key_control`: causal-prefix key reversal of `S`;
- `private_key_control`: causal-prefix key reversal of `R`;
- `private_sign_control`: `-R`.

Run each arm with the source equality term present and removed. Native replay is computed separately. All scalar
scales, signs, factor sources, target payloads, and key transformations are fixed before model outcomes.

## Data and causal measurements

Use the same separately frozen 192 `final_natural` and 192 repository-disjoint `ood_code` documents as rung 533,
with fixed halves `0:96` and `96:192`. This is a result-conditioned mechanistic follow-up, not a fresh corpus claim;
the new `private`, `private_key_control`, and `private_sign_control` outcomes have never been opened.

For cell `c` and arm `X`, define the per-document effect vector

```text
E_X[c] = CE_absent[c] - CE_X[c].
```

The observed marginal effect of adding the private correction to the shared score is

```text
K_R|S[c] = E_native[c] - E_shared[c]
         = CE_shared[c] - CE_native[c].
```

If `R` is an autonomous circuit component, its standalone effect `E_private` should predict `K_R|S`. If it is only
meaningful through nonlinear interaction with `S`, those vectors will differ. Compare them without a fitted scale
using cosine and relative error, separately for copy-positive, matched copy-negative, and all remaining positions.
Only documents with at least one selected token enter a cell vector.

## Registered predictions

### A — exact live instrument

Native and analytical replay logits agree exactly; factor and deployed target-product reconstruction errors are at
most `1e-10`; `P_t=S+R` has maximum error at most `1e-7` on every batch; every removal/replacement has nonzero RMS;
all cell/document supports and exactly `1,440` model forwards reconcile; row, source, and checkpoint hashes match.

### B — the known shared-signal premise reproduces

In both `ood_code`, donor-absent halves, `shared` reproduces the native copy-positive per-document effect with cosine
at least `0.85`, relative error at most `0.60`, and aggregate recovery in `[0.65,1.40]`, while its absolute
matched-negative CE difference from native is at least `0.02` nat. Its copy-positive cosine beats
`shared_key_control` by at least `0.15`. This is an inherited positive-control premise, not a new family claim.

### C — the private correction acts autonomously where it is needed

In both `ood_code`, donor-absent halves, `E_private` predicts `K_R|S` on both copy-positive and matched-negative
document vectors: cosine at least `0.80` and relative error at most `0.60`, with no fitted scale.

### D — the private correction is key-specific rather than merely norm- or sign-matched

In each C comparison, `private` exceeds both `private_key_control` and `private_sign_control` by at least `0.15`
cosine. This must hold in both code halves and both cells.

### E — the shared/private split transfers to natural text

C and D also hold in both `final_natural`, donor-absent halves. This is the cross-corpus identification gate for an
autonomous private correction.

### F — the correction's autonomy survives the redundant donor

C and D hold in both roles and halves when the source equality term remains present. This determines whether the
private component is a reusable correction or depends on removing the source head first.

## Opposing outcomes

- **Shared signal plus autonomous private correction:** A--E pass. We may split the target score into a cross-head
  shared equality component and a target-specific noise/context correction; F says whether that split survives
  redundant implementation.
- **Interaction-only correction (strong null):** A and B pass, but C fails in both code halves for either the
  copy-positive or matched-negative cell. Then `R` is only meaningful when composed with `S`; do not call it an
  independent circuit or retry with a lower rank/bar.
- **Relation-unspecific residual:** C passes but D fails. The standalone response is not enough to identify the
  target-specific score relation.
- **Invalid:** A or B fails. Preserve the receipt and do not interpret the private arms.

A pass identifies a computational split for one source/target pair. It does not yet save parameters or adopt a
replacement. Broader cross-head reuse, joint installation, selective removals, and literal price remain separate
adoption requirements.

## Literal price

Seven analytical arms in two backgrounds plus one direct-native identity forward give 15 forwards per four-document
batch. Across 384 documents this is `15 * 96 = 1,440` model forwards, zero backward passes, and zero fitted vector
parameters. A one-batch, 15-forward managed smoke must expose only identity, decomposition, edit, support, call, and
checkpoint diagnostics before the full run may open any private-correction loss.
