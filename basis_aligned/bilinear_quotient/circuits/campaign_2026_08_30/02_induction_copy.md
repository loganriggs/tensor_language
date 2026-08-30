# Induction and repeated-bigram copying

## CURRENT tier: 4

The matcher, identity payload, relays, and product algebra are known; not every required
source has been recursively replayed to embeddings in one terminal program.

## Behavior and tensor program

Endpoints are synthetic `AB ... A -> B` change in `log p(B)`/CE and natural CE where
the current/next bigram occurred earlier.  Matched negatives repeat `A` with a different
earlier follower; first mentions and noninductable repeats are controls.

Fixed tensor form: double-bilinear QK matchers and rank-128 identity OV payloads in
`L5H5/L7H3/L8H3/L8H4`, with L4H7/L6H3 relays.  The **router is the source-match QK
score itself**.  Extraction executes native/reconstructed scores; selective removal
gates only the matched source edge.  An unconditional mean router is forbidden.

## Evidence

- [`induction_mechanism.py`](../../induction_mechanism.py),
  [`circuit_induction.py`](../../circuit_induction.py), and
  [`induction_injection_family.py`](../../induction_injection_family.py).
- [`terminal_copy_selection_v1_attempt2_result.json`](../../../polynomial_causal/terminal_copy_selection_v1_attempt2_result.json):
  four-head target effect `+0.44869993` nat but collateral margin `-0.01440914`, rejecting
  unconditional mean replacement.

## Terminal gates

Extraction retains/rebuilds the four heads and relays in an attention-null background.
Removal is source-match gated.  Collateral includes first mentions, wrong followers,
copied non-targets, entity/capitalization, successor, and global CE.  OOD holds out
token identities, lag bands, natural/synthetic roles, and domain; default gates apply.

Shared-owner caveat: the copy heads overlap copied-entity and capitalization service.

**Next experiment:** recursively replay matcher/payload/relay sources to token and
position primitives, then run source-match-gated terminal extraction/removal.

## 2026-08-30 equality-tensor update

The source-match-gated experiment completed and passed all SELECT gates. The four-head
equality-fetch tensor caused `+0.51225` nat target damage when removed, with specificity
`+0.55251` and off-target damage only `+0.006264`. Extraction from the four-head-deleted
background recovered `0.97397` (95% `[0.94789,0.99479]`); a fixed cyclic-vocabulary
derangement recovered approximately zero. Analytical replay was bit-exact and candidate
native-call counts were zero at all three affected layers.

This resolves the prior collateral failure: unconditional whole-head replacement was
the wrong intervention because those heads have other services. The fixed
equality-and-successor contraction isolates induction without introducing a router.
Current status is **mechanistic Tier 4; extraction/selective removal SELECT pass; OOD
pending**. It is frozen for natural FINAL and code OOD.

Details: [`INDUCTION_EQUALITY_TENSOR_DISCOVERY_FINDINGS.md`](../../../polynomial_causal/INDUCTION_EQUALITY_TENSOR_DISCOVERY_FINDINGS.md).

## FINAL/OOD execution status

The fresh evaluation rows and the executable owner both passed independent,
outcome-blind audits.  The first authorized execution nevertheless stopped before
scoring any row: its integrity checker attempted to reinterpret a scalar bf16 state
tensor as bytes without flattening it first.  PyTorch rejects that operation.  This is
an implementation failure, not a negative scientific result, and is preserved in
[`induction_equality_tensor_final_ood_v2_failure.json`](../../../polynomial_causal/induction_equality_tensor_final_ood_v2_failure.json).

An implementation-only retry changes exactly that byte-hashing operation and uses a
fresh authority/output namespace.  It does **not** change rows, tensor program, arms,
metrics, bootstrap, thresholds, or decision rules.  The amendment is
[`INDUCTION_EQUALITY_TENSOR_FINAL_OOD_V2_RETRY1_AMENDMENT.md`](../../../polynomial_causal/INDUCTION_EQUALITY_TENSOR_FINAL_OOD_V2_RETRY1_AMENDMENT.md).
Until that retry independently audits and runs, natural FINAL and code OOD remain
pending and no terminal promotion is claimed.
