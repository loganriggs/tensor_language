# Exact source attention fold and spectral response baseline

20 September 2026. These results concern fixed-context source-amplitude response programs, not a complete extracted circuit.

## Exact attention fold

For x[b,t] = h[b,t] + K[b,t]a[b], compile both QK score numerators as quadratic forms in five shared source amplitudes. Keep input RMS, four Q/K head RMS denominators, rounded RoPE, causal masking, first-layer cached values and position-dependent downstream readers explicit. Contract readers through the output projection and values. Execution needs the compiled context but no native attention weights. The compiler does not materialize a fifth-degree tensor.

The initial native run crashed because the cached values were float32 in a float64 contraction. Its original runner and log are preserved. The corrected compiler explicitly promotes cached values; the CPU mixed-precision test also promotes them before multiplying, matching the independent native float64 reference. Output/gradient/Hessian toy errors are at most3.56e-15, and removing the live cache causes error1.009.

Corrected native receipt: `../bilinear_quotient/circuits/followups/native_attention_source_core_v1r1_result.json`. Across112 local attention cases (seven layers,96 opened texts, two sites), output error4.89e-15; Hessian absolute error7.64e-17, relative3.74e-15; archived native Hessian replay exactly zero. All registered counts match. All native backgrounds, five source sensitivities, four downstream readers and caches remain dependencies and their producers remain charged.

The preregistered one-shot speed gate **fails**: compilation1.201s plus folded Hessians7.584s versus native Hessians8.580s, ratio1.02397. Both methods receive forward warmups, CUDA-synchronized timing and alternating order. This single run does not establish a statistically significant slowdown; it does fail the stated strict speed gate. Folded execution alone is cheaper here, but ignoring compilation would change the comparison. No end-to-end speedup is established. Each compiled batch context stores1,109,640 values; per-context storage and shared native weight storage are different amortization regimes.

## Conventional spectral baseline

`audit_five_source_spectral.py` independently truncates each context/output analytic symmetric5x5 Hessian by absolute eigenvalue, retaining signed eigenvalues and exact native gradients. It compares the same outputs, five source ports and finite edits as the dense response program. All contexts are opened. It fits no native outcome labels, but its per-context eigenspaces are not a shared learned dictionary.

| Hessian rank per output | Literal values/context, four outputs plus gradients | Worst number error,15 singles/pairs | Worst modal error/number budget | Gate |
|---|---:|---:|---:|---|
|1|44|18.99%|4.59%|Fail|
|2|68|5.16%|1.57%|Pass|
|3|92|4.70%|1.35%|Pass|
|4|116|4.70%|1.39%|Pass|
|5|140|4.70%|1.39%|Pass|
|Dense symmetric|80|4.70%|1.39%|Pass|

Factor price is20 gradient values +4 outputs * rank *(5 vector entries +1 eigenvalue). No discount for orthogonality constraints or unstored theoretical degrees of freedom. Higher ranks are computational baselines, not storage improvements. Rank-two also passes prior unit-B/full-null/half-null edits: number8.84%, modal2.74%. Rank-one fails number11.67% there. Full-rank coefficient recovery8.33e-17 and prediction replay2.23e-16 rule out an elementary eigensolver/sign/axis defect as the cause of rank-one failure.

Receipt: `FIVE_SOURCE_SPECTRAL_BASELINE_CPU_RESULT.json`. Dense baseline storage80; rank-two68 is15% lower conditional coefficient storage, with native producer costs unchanged. No prospective OOD claim, stable feature identification, selective-removal improvement or native HT comparison follows. The remaining task is to discover reusable weight-derived features and validate their finite interventions prospectively. This result supplies a stronger conventional baseline for that task.
