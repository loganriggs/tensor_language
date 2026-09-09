# Contextual answer-history versus binding retrieval

Registered2026-09-09 14:16 UTC before teacher outcomes. Small-model continuation
under the reconstruction handoff. Radius-one first-layer transport is closed.

Observation used for design: an independent512-document token-only audit of the
generator at seed4909 gives~21% repeated(entity,hop) queries. With chance~1/24 on
novel queries this could explain the small model's~25% higher-hop accuracy. This is
a hypothesis, not a measured mechanism. No native outputs from these new documents
have been opened. Archived hop reports discuss pointer/probe interpretations but
contain no repeated-query or answer-history controls found by the targeted search.

Use the same bound400640-parameter attn-mlp-attn-rms-seed0 checkpoint. Populations:
128 IID cycle documents seed4909;128 unique-query-key documents seed4910 (48 keys
sampled without replacement from96(entity,hop) pairs, OOD repetition structure);
128 arbitrary-permutation-function documents seed4911 (short-cycle topology OOD).
Answers always recomputed from the function; no capability filtering. The IID
population uses the same seed as the token-only design audit; no teacher outcome selection.

At every query H_k position define disjoint native final-attention source groups:
H = all earlier answer-token positions whose query(entity,hop) equals the current
query; B = the unique initial binding-value position whose key is the current
entity; O = all other causal sources. R is the final residual-skip contribution.
For repeated queries with enough previous same-hop/different-entity answers, C
selects as many most-recent such controls as H. Record eligibility independently
of correctness; matched H/C comparisons use only these declared eligible rows.
The count/distance distribution is reported: controls need not match exact age.

Execute original final-attention forwards with no removal, remove H, remove B,
remove H+B, and remove C. Keep all upstream computation and live normalization.
Fold the final output matrix into W_U W_O; a standalone partial executor then
computes tokens -> native prefix -> live Q/K/V -> full29-way residual + source
contributions. Its first-layer/MLP prefix and contextual routers remain explicitly
native and priced; no stored native activations. Physically remove the final W_O
from the copied program, storing the folded readout instead. This exact compiler
fold is a baseline saving, not itself a newly discovered semantic operation.

pred_a_exact_extraction: every original/compiled full logit vector on all239 input
positions and all5 arms matches at atol=rtol=1e-9; planted CPU checks verify the
mask partition, forward causality and fold. Report full/query teacher KL and all
centered intervention-vector errors, not only answer margins.

pred_b_repetition_explains_floor: on IID and short-cycle OOD, higher hops2/3 have
accuracy>=.80 when repeated, <=.15 when novel, and gap>=.50. On unique-query OOD,
higher-hop accuracy<=.15. These are fixed mechanistic prediction bars; no claim
that poor native behavior is a success for the model.

pred_c_selective_sources: on matched-eligible repeated higher-hop queries in IID
and short-cycle OOD, mean gold-probability loss from H removal>=.25, and C removal
<=.10. On novel hop1 queries B removal gold-probability loss>=.25 on both these
populations. Report C and B effects, KL and all hop groups even if a clause fails.

pred_d_composition: compiled single/joint effect vectors and native joint logits
z_H+z_B-z_0 agree with z_HB at1e-9 across all populations. Since the final readout
is affine and H/B are disjoint, this is a mathematical instrument prediction,
not empirical evidence that upstream circuits compose independently.

Null: repetition does not explain the floor or history sources are not causally
selective. Then retain only the exact full-readout decomposition and look for the
observed alternative; no post-hoc population filtering or claim of chained lookup.
A positive identifies an answer-history-dependent contribution under this parser;
it does not establish that the native router literally computes discrete equality.
Next operation discovery must explain its contextual matching strengths/payload,
including native errors, with a smaller implementation and new held-out evidence.

GPU via managed runner. Batch4, context239, all29 outputs,384 docs x5 arms per
executor,1800-second watchdog, every new analysis tensor<256MiB. Full-fold parameter
count400640 -16384 +3712 =387968, retaining the original3712-element unembedding
for the residual branch. All rotary/mask buffers and parser operations count.
This supporting small-model study does not establish any new circuit in bilin18.
