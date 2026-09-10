# Contextual value versus mean routing, in both recipient contexts

Use all16 original/fronted pairs from THIRD_NOUN_CAUSAL_ROLE_ALIGNMENT_V1. Preserve
all prior structural/gender and strict producer-only/reader-only failures. The
tested object is the common mixed attention write, not a whole semantic circuit.

From native layer9 normalized inputs u and actual post-head-RMS/RoPE Q,K,Q2,K2,
compile C(v,p)=(1-lambda)W_O sum_s P0_p(q,s) Voh_v(s), with Voh=W_V Qoh u.
Value source and mean-routing source each take original/fronted contexts. Native
lambda=-.65625; all9heads retained. P0 is the four-corner conditional mean of the
actual product-attention kernel. Native source head width128, no softmax.

Common queries are to/action; mixed values can occur only at the final3positions:
first jointly informed token/to/action. Both layouts have the causal mask
[[1,1,0],[1,1,1]]. Source alignment is by information stage, not identical noun
role. Normalized state production and all positional computations remain explicit.
Evaluate W_V then W_O in their factored form; no rank selection or materialized
opaque tensor is counted as structural savings.

Capture each layout's baseline and local-value edit to obtain observed C_r. For
recipient r, background B_r=A_r-C_r; true replacement installs B_r+C(v,p).
Evaluate all2value×2routing×2recipient effects relative to the removed-C background.
First validate native factors, then homogeneous producers v=p in both recipients
against the previous interchange. Only then evaluate the hybrid v!=p producers.

Perpair:8forwards baseline/local captures,4forwards removed-C backgrounds,
16forwards eight combinations. Total448forwards7168sequenceinstances,0fits,
0offline decoder,600s cap. Same texts, no new OOD claim.

A instrument: bound files, finite/hookcleanup, exact448/7168counts; inherited
value commutation<=1e-4, early input support<=1e-6, QK/firstV unchanged. Own native
compiled C versus observed common C relativeFrobenius<=1e-4 for each layout.
Native, removed, and all four homogeneous-producer recipient outputs replay
the completed common-interchange result with BOTH maxabs<=1e-3 and relative
Frobenius<=1e-5. CPU factorized-output contraction agrees with independent
three-index contraction<=1e-12. Verify factor corners, reader tokens and causal
shape; incoming output patch matches native bitwise and firstV tuple preserved.
Mechanical failure prevents hybrid interpretation. Scientific zero denominators
fail the relevant gate, not A.

For each recipient separately, arrange E[value,route] after Qoh, and score both
the correct answer margin and centered three-reader vector. EVERY pair, recipient,
direction and both readout objects must pass a rule:
- B value-following: E10≈E11 relative to E11 and E01≈E00 relative to E00,
  eacherror<=.10. Tests invariance to routing source at fixed value source.
- C routing-following: E10≈E00 relative to E00 and E01≈E11 relative to E11,
  eacherror<=.10. Tests invariance to value source at fixed routing source.
- D small value-routing interaction: ||E11-E10-E01+E00|| divided by the larger
  diagonal norm<=.10. Zero denominators fail. Save the complete effect cube and
exact symmetric value/routing change accounting without fitting any scalar.

If neither B/C passes, retain the joint value-routing mechanism; do not promote
the smaller error. Passing a factor rule specifies this bounded producer under
the tested backgrounds, not independently generated inputs or selective full
behavior. All545902902 native parameters, counterfactual production and native
suffix remain; savings0. No changed head, source role, gain, thresholds or rows.
