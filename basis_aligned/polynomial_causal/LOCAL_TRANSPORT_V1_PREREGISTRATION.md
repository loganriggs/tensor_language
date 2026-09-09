# First-layer local transport: fixed dependency test

Registered2026-09-09 14:04 UTC after the equality/gain null, before transport outcomes.
Controlling objective remains the reconstruction handoff and its final success
criterion. This is an empirical test of a specific source dependency, not a claim
that locality itself explains all retained arbitrary weights.

Hypothesis: first attention of the same trained attn-mlp-attn-rms-seed0 checkpoint
needs only current and immediately preceding tokens. Compile its entire coupled
read-route-write as two shifts: for d in {0,1}, evaluate native normalized Q1/Q2 at
t, K1/K2/V at t-d, multiply the two dot products /32^2, multiply V, sum d, apply
native W_O and residual lerp. Keep absolute cached rotary factors and live norms.
All four heads share the two source-index operations. No n-by-n attention matrix
is produced by the extracted first layer. All400640 parameters remain charged.
The later MLP, attention and output head are fully recomputed in a copied model.

Same four-property standards as equality v1: full29-way teacher distributions,
explicit token inputs/native background, matched native edge removals, and joint
execution/interaction. Fresh populations16 documents each: cycles seed3909,
renaming seed3910 (metamorphic, not OOD), arbitrary permutation functions seed3911
(short-cycle topology OOD). All239 positions and48 answer positions reported and
gated separately, no selection. Native model accuracy is descriptive only.

Arms: none; remove all first-layer self edges (d=0); remove all first-layer previous
edges (d=1); remove both. In the native model all other causal edges remain live.
In the compiled model no other source edge exists. Branch removals affect all four
heads and correspond to the same native pattern edit; neither private selective
effects nor additivity are assumed. The actual interaction vector is predicted.

Instrument pred_a: on a CPU synthetic model, compiled two-shift contraction matches
a dense causal pattern restricted to d<=1, under all arms, atol=rtol=1e-9. Repeat
that full-logit equality on every trained-model population/arm. Earlier outputs
must not change under edits to later input tokens in the synthetic fixture. Negative
control: removing the previous branch must have a nonzero synthetic effect.
pred_b: each arm/population all/query KL mean<=1e-3, p99<=1e-2.
pred_c: each single/joint/interaction centered-logit relative error<=.01, or absolute
error RMS<=1e-8 if target RMS<1e-6. pred_d_live_branches: native removals of both
branches separately have RMS>1e-6 on every population. These are necessary tests;
unchanged400640 opaque parameters prevent calling the entire model explained.

Failure closes this fixed radius-one transport candidate; no radius sweep, fitted
mask, threshold relaxation, or post-hoc document/head selection. A passing candidate
licenses joint decomposition of its two remaining token-to-write operations, with
full consumer/parameter accounting. It does not satisfy the full bilin18 goal.
Watchdog1800s, batch4, context239, each analysis tensor<256MiB. Run GPU only via
bqrunner. Report complete model constants and the first-layer edge count reduction
T(T+1)/2 ->2T-1, keeping later layers in the baseline. No training or provisioning.
