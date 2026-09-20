# Exact conditional input-space capacity — 2026-09-20 18:22 UTC

Question: how much coefficient energy is available to ANY quartic using the fitted program's frozen input directions? This distinguishes input dictionary limitations from root-core limitations without claiming global optimality.

Teacher: pure MLP16 -> MLP17 -> unembedding quartic with exact QR output frame; other residual terms, attention and normalization omitted. Freeze three32-column spaces from NATIVE_INPUT_MODE_V1.pt: leading input modes, CP8 readers, learned shared-bank readers. Reorthogonalize in FP64. Restrict first-layer readers to each basis. Enumerate all C(35,4)=52,360 unordered input tuples, weighting permutation multiplicities. Exact projected coefficient energy in floating-point arithmetic; teacher total norm still estimated. No coefficients stored or fitted.

Opposing outcomes: low student utilization motivates changing its root representation; high utilization motivates changing its input dictionary. This is an unrestricted quartic ceiling conditional on each frozen subspace, not an attainable CP/bank result or bound for other input directions.

Predictions: pred_a_precision first128 coefficients FP32/64 discrepancy<1e-4 and basis orthogonality<1e-5; pred_b_sampling all exact energies within6 reported SE of independent1024-probe estimates; pred_c_utilization learned-bank gain/projected energy in(.6,1.001]. Report all outcomes. Circuit target: feature-dictionary sufficiency; no causal identity claim.

CPU check: exhaustive d3 ordered-vs-weighted norm agrees exactly in FP64; multiplicities sum to32^4. Work: three52,360-entry streams,batch256,zero model/text forward passes. No large tensor artifact. An initial authoring call used the wrong working directory and failed before creating the plan/runner or queueing; corrected before enqueue.
