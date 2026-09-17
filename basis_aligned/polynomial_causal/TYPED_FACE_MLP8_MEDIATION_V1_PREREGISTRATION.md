# Split the native head8 write from its MLP8 response

Same frozen native8 half-write and same opened prospective40sequences. Four
arms: native, native8 midpoint, block9 reentry with skip delta8 only, block9
reentry with local MLP8 response only. Full native8 is the true joint, not a
linear sum of scores.160bodyforwards plus40local MLP8 evaluations;300seconds.
The full retained product and destination mask remain fixed.

Capture raw post-attention8 input g and native MLP8 output on baseline. Compute
exact RMS-aware deltaMLP8 using native Left/Right/Down weights, explicitly
retaining normalization, cross, quadratic terms, and bias cancellation.
Only intervention application differs between arms. All later native paths
are recomputed. Reference artifacts are native8 scope arms[0,2].

pred_a: replay references<=1e-5 maximum absolute; local folded deltaMLP8
matches direct FP32 native evaluation <=1e-4 relative; transported joint
block9 input equals native8 block9 input <=1e-4 relative change norm;
finite, zero response outside destinations,160bodyforwards.
pred_b: both single target effects >=.05jointRMS and>=1e-5 in every family.
pred_c: skip-only effect predicts joint <=.35relativeL2 everyfamily.
pred_d: MLP8-only effect predicts joint <=.35relativeL2 everyfamily.
pred_e: joint predicted from singles <=.10relativeL2 and interaction <=.25
smaller single effect everyfamily. Report allfour unrelated readers but make
no new selectivity/null or fresh-row claim from this four-arm screen.

Opposing predictions: if skipping MLP8 retains the amplified effect, other
attention consumers/routing changes dominate the scope difference. If MLP8
alone suffices, its exact local response is the next extraction boundary.
If neither suffices or interaction is large, preserve the coupled expression.
Small interactions alone do not certify composition without random-split nulls.
Do not alter gates after scores. Both single-sufficiency predictions may fail.
