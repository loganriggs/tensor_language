# Fixed midpoint edit: downstream response census

Keep the completed prospective-panel midpoint intervention unchanged, on its now
opened 40 sequences/240 endpoint rows. Native plus midpoint,80bodyforwards/300s.
No new intervention strength, support, token, normalization or source omission.

Capture actual block inputs/outputs and MLP writes at layers9–17. Recover each
attention write as block_output - mixed_input - MLP_write in FP64 from native
FP32 tensors, so edited attention9 includes the existing injected odd-value
change. This includes local addition roundoff; report global closure. Transport
all changes with gamma_l=product(lambda[j,0],j=l+1..17).

At final state x0,x1 and rho_i=sqrt(mean(x_i²)+FP32eps), normalized difference
is sum(gamma*module_delta)/rho1 + x0*(1/rho1-1/rho0). Contract each term with
native unembedding readers. Allocate softcap exactly with per-token secants
(cap(u1)-cap(u0))/(u1-u0); zero-difference limit is cap derivative. This is a
chosen finite-change attribution, not a unique causal decomposition. Norm term
and all nine attention/nine MLP terms stay explicit, including signed cancellation.

pred_a: anchors versus saved native/midpoint scores <=1e-5 maxabs; transported
residual sum relative error<=1e-4; total attributed-margin error<=1e-4maxabs AND
<=1e-3relativeF; all values finite and80forwards. Failure invalidates census.
pred_b (direct-readout sufficiency): attention9 response alone predicts actual
target effect within .35 relativeL2 in EVERY construction family.
pred_c (fixed early suffix): attention9+MLP9+MLP10+MLP11+output-normalization
response sum predicts actual target effect within .35 in EVERY family.
pred_d (family stability): signed term-effect vectors (mean British-minus-American
paired contributions) have cosine>=.9 between line-break and every other family.
No dropping failed family or reselecting top terms to change these gates.

Report every module's norm ratio and signed aligned fraction, plus four unrelated
readers; top rankings are descriptive. Null: an apparently direct/simple response
is actually distributed/canceling or changes across frames. A passing response
set nominates a later independent edit; it cannot establish a causal suffix,
selectivity, extraction, compression or composition by itself.
