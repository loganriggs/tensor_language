# Absolute-position audit of the fixed equality-router hypothesis

Registered 2026-09-09 13:57 UTC after v1, before this audit. Preserve v1 as
instrument_invalid: its real-arithmetic lag reduction failed its frozen full-logit
control (worst absolute error0.00102). Its table KL was tiny (~1e-11) whereas the
invariant candidate KL was ~13.6, but the original gate does not license a scientific
verdict. This audit repairs positions; it does not relax that gate or optimize a new
candidate from observed outcomes.

Same checkpoint, entity group,37 orbit means, four heads, all three16-document
populations, full239-position outputs and48 queries, and four removal arms as v1.
These populations are now reused diagnostic data, not a new held-out confirmation.
Use exact native cached rotary factors for each absolute query/source position when
forming K[h,t,s,a,b]. Compute one query-position slab at a time (<7MiB); retain only
its37 orbit means. The whole diagnostic coefficient table is <68MiB. No tensor over
256MiB, no fit or head/document selection, no changes to normalizers or checkpoint.

For each slab, gather the unreduced entries actually exercised by all48 documents;
compare with the original first-layer pattern at atol=rtol=1e-9. Independently replay
the original full forward and manually contracted first-layer forward under every
removal, also at1e-9. This is the repaired instrument gate pred_a_instrument.
pred_b_distribution and pred_c_interventions retain every v1 threshold, including
full and query KL mean1e-3/p99 1e-2 and centered effect/interaction relL2 .01 with
RMS floor1e-6 /absolute error1e-8. pred_d_same_fixed_hypothesis: absolute-position
and v1 lag-projected candidate logits agree at atol=rtol=1e-4; report failure honestly.
Unlike v1, pred_d is only a repair diagnostic, never promotion or discovery credit.

If pred_a passes and either b or c fails, the naive invariant first-layer router is
rejected at the fixed thresholds; interpret it only as a failure of that boundary.
If b and c pass, the prior symmetry conclusion was positional contamination and must
be corrected. An absolute-position table has4x239x239x37=8453908 constants versus
65536 removed Q/K weights: it is an explicitly expensive diagnostic and cannot itself
meet the structural simplicity criterion. Its purpose is to adjudicate v1, not create
a more opaque "circuit". GPU only through managed runner, watchdog1800seconds.
