# Narrative tense fresh A1+A2 native capability V1

This is a native-only capability gate. It freezes 16 lexically fresh groups before model access: groups 0–7 are FIT and groups 8–15 are an unopened lexical HOLDOUT. The 48 one-token content words and all prompt endpoints are disjoint from every earlier narrative authority checked by the builder.

The fixed A1 target uses a direct `Yesterday/Today ... served/serves` clause followed by an explicit `At that/this time` rec cue. The fixed A2 target uses a distinct reported-record construction with `showed/show` and a repeated `that/this period` cue. P and C preserve tense while changing the subject in the matching A1 and A2 construction. The answer vocabulary is the single-token pair ` was` and ` is`.

FIT contains four examples in every family × direction × side cell. Every FIT cell must be at least .875 accurate, which requires 4/4 at this count. HOLDOUT remains unopened if FIT fails. If FIT passes, the same bar applies independently to all HOLDOUT cells. No row removal, template fallback, threshold change, attention access, or causal intervention is allowed.

Registered outcomes:

- `pred_a_fit_all_cells_pass`: every frozen FIT cell passes.
- `pred_b_holdout_all_cells_pass`: after a FIT pass, every untouched HOLDOUT cell passes.
- `pred_c_license_issued`: a hash-bound native capability license is issued only after both phases pass.

This gate establishes dataset capability only. It does not establish a carrier, source factor, routing mechanism, or lexical universality. A causal run must validate the exact authority, result, registered-cell, and license hashes before model loading.

Maximum price: two model forwards, 128 endpoint evaluations, zero backwards, zero updates, and zero causal interventions.
