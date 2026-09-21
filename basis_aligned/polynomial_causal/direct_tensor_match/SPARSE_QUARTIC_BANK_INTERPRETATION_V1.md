**A broader shared hierarchy beats the flat CP baseline on two diagnostics, but remains inaccurate.**

Two random starts,100Muon steps,144quadratic features with4products each and512fixed selected feature pairs. Both starts pass the registered integrity, learning and comparative gates. The gate is a comparison to a poor baseline, not a10%fidelity test.

|Start|Sampled coefficient error|Gaussian function error|Opened text scalar error|
|---|---:|---:|---:|
|1101|96.916%|91.501%|65.657%|
|1102|97.215%|91.555%|63.821%|
|CP400 best across starts|97.776%|98.878%|39.027%|

The shared program uses1088variable products,1353728float coefficients and1024indices, versus CP's1536products/2385920coefficients. It improves coefficient and Gaussian metrics at lower final program cost, while text reconstruction is worse. Exact fitting was more expensive:883.28seconds and13.73GBpeak versus CP's259.65seconds and1.50GB. Training compute and iteration counts are not matched. No hierarchy-wide superiority or optimizer convergence is established.

Normal-equation residuals<1.2e-15 and physical FP32 export errors<1e-7 pass. The regularized explained score reaches.006280/.006415. Millions-fold improvement over nearzero random initial capture is not a millions-fold error reduction. Tensor query errors remain sampled, not exact normalized full Frobenius certificates.

The pair support contains every quadratic square plus368seeded off-diagonal pairs. These connections were fixed before fitting, not discovered through sparsity optimization. Learned input directions are free; all selected products are shared across output readouts. There is still no general graph-edit search in this native run.

The result supports testing reusable quadratic features as one structural hypothesis. It does not support feature meanings, faithful extraction, selective removal, unseen-text prediction or composition. Further fitting, adaptive pair selection and less aggressive budgets remain distinct possibilities; the bad absolute fit cannot choose between them alone.

[Native receipt](SPARSE_QUARTIC_BANK_NATIVE_V1.json) · [Registered plan](SPARSE_QUARTIC_BANK_NATIVE_PLAN_V1.md) · [Dense controls](SPARSE_QUARTIC_BANK_CONTROLS_V1.json) · [Wider planted recovery](SPARSE_QUARTIC_BANK_RECOVERY_WIDE_V1.json).
