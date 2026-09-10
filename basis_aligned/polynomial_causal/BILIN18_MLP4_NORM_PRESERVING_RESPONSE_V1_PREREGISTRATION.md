# Keep nonlinear normalization; simplify only multilinear factor responses

Registered10 September2026 before native execution. Whole receiver tangent
misses37–48% of the finite value-path effect. That tangent linearizes RMS
as well as bilinear products. This NEW candidate preserves actual changed
normalization and tests first-order responses only in the resulting factors.
No scalar repair, step reduction, fitted coefficients or midpoint trajectory.

Same72 opened pairs, all valid MLP4 source outputs swapped, both directions.
Capture receiving-native attention q,k,v,q2,k2 and MLP Left/Right factors
at layers5..8. During the source change, all receiving residual recurrence,
RMS, projections, head RMS and RoPE recompute on CURRENT inputs.

For each MLP use native base output plus
Down[L_base*(R_current-R_base)+(L_current-L_base)*R_base]. Its bias remains
in the base output. For attention use native base head read plus the
first-order change in the FIVE current normalized factors q,k,v,q2,k2,
with base scores, exact causal mask and native output projection. The
formula is implemented in polynomial_factor_response.py. Preserve actual
receiving first-value payload in returns. Products between changes are
omitted; nonlinear normalization is explicitly retained, never frozen.

This combined candidate applies to BOTH module types at layers5..8.
No posthoc module subset or retention of selected higher-order terms.
Capture its induced localV9, then use the usual receiving-native H1/H4
value-reader assay, keeping Q/K/background native and recomputing suffix.

Perpanel:2 native factor captures,2 full MLP4swap captures,2 candidate
captures,2 identity-product-response captures without source change,
2 fullvalue reader forwards,2 candidate reader forwards=12. Total48forwards/
864seq. Candidate identity outputs must replay native full logits.

A: parent/source/rows/backend/checkpoint hashes; CPU factor controls;
native factor-read oracle/identity full-logit maxabs<=1e-3 ANDrelFrob<=1e-5;
fullvalue parent margin/effectnorm replay atsamebars/1e-5relative; exact
counts, current factor capture completeness, source masks, unselectedvalue
preservation, allfinite and hook/method restoration. No newmodelbackend.
B: candidate value-reader effect relativeerror<=.10 versusfullvalue in BOTH
centered-logit andmargin frames, EVERY panel/direction. Zero reference
norm<=1e-8 uses absoluteerror<=1e-8. C: both effects nonzero>1e-8. A miss
invalidates; B miss remains a null with no degree/dose/source/rank rescue.

This tests whether nonlinear normalization plus context-dependent affine
factor responses suffices for an EXISTING partialpath. It is not a new
text OOD result, a fixed context-free linear operator, or whole-task
sufficiency. Prior has full-logit carrier miss remains. All545902902 native
parameters, receiving base factors and source generation remain charged;
actualweight saving0, no fit/adoption.600-second watchdog, managed enqueue.
