# Successor fixed-pointer module-response screen v1

## Claim and novelty

This is a **screen**, not an identification or compression result. The frozen successor-pointer factorial showed that a late prefix swap changes the native continuation baseline while the final token pointer, answer, token multiset, and position remain fixed. Legacy successor component patching changed the final element itself and therefore mixed pointer and prefix effects. This experiment asks which, if any, single complete native module response at the final query carries the prefix-conditioned backward-alternative suppression.

The 30 frozen rows in `SUCCESSOR_POINTER_PREFIX_INTERACTION_V1_ROWS.json` are reused without selection. Within each family/final-index group, coherent, early-swap, and late-swap prompts align one-to-one. No outcome is used to change rows, sites, controls, or bars.

## Intervention and price

Capture all 18 attention outputs (all nine pre-`c_proj` head responses) and all 18 MLP outputs for coherent, early-swap, and late-swap batches. For each of 36 complete modules, transplant only the coherent final-query response into the aligned late prompt and, separately, the aligned early-control prompt. Every other response stays native to the recipient prompt. The final token pointer is never edited.

The exact self arm replaces all late responses by their own cached values. The full ceiling replaces all 36 late final-query responses by their coherent counterparts. Price: 77 full forwards over 10 sequences each, 770 sequence evaluations, zero fits, gradients, parameter updates, ranks, gains, thresholds, or quantization.

## Metrics and frozen bars

For each row and direction, margin means recipient answer logit minus the corresponding backward or forward donor-answer logit. The late backward target is coherent margin minus native late margin. A candidate rescue is patched-late margin minus native late margin. Per family, report target projection `dot(rescue,target)/dot(target,target)`, cosine, and residual RMS. Controls are the candidate's late forward-margin change and coherent-to-early backward-margin change.

- **A — instrument:** all-late self replay maximum logit error <= `1e-5`; all-coherent-query ceiling maximum logit error versus coherent <= `1e-4`; executed forward and sequence counts equal 77 and 770.
- **B — live target/capability:** native answer accuracy is >= `0.75` in every family-condition cell, and mean late backward target is >= `1.0` logit unit in each family.
- **C — singleton response:** at least one complete module has projection >= `0.50` and cosine >= `0.70` in each family, while its late-forward and early-backward control RMS values are each <= `0.50` times that family's late-backward target RMS.
- **D — distributed or unresolved null:** A and B pass and no singleton satisfies C.

Candidates are ranked by the minimum family projection, then minimum family cosine, then module order. A C pass localizes a response carrier only on these rows and requires a fresh confirmation and, for attention, a head split. A D pass rejects a single complete-module carrier under this grammar; it does not reject distributed or typed response circuits. If A or B fails, the result is invalid. No failed arm may trigger a site subset, gain, rank, row, direction, or bar rescue.
