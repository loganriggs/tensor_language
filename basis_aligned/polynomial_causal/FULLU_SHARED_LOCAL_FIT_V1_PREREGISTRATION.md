# Full-vocabulary shared/private graph: multistart fit

13 September 2026. This extends the completed native timing screen; it does not revise its measured result.

**Object and objective:** the complete U-folded MLP17 quadratic coefficient tensor, using exact centered Hilbert coordinates X. Retain its vocabulary mean separately. Minimize squared coefficient error without text, activation statistics, token labels or a task-selected vocabulary. The final denominator includes the exactly retained mean energy.

**Two fixed capacities:** shared width 64 / 32 private groups / private width 8; and shared width 128 / 64 groups / private width 16. Prices are 16,167,936 and 34,489,344 bytes including int32 assignments. Keep all native L/R/Down and products. The matched global comparators use ranks 78 and 167, with exact optimal coefficient errors computed from the same centered tensor and exact mean.

**Optimization:** ten independent private-bank initializations per capacity, seeds 9518–9527, with the same spectral global initialization. Three exact alternating sweeps screen every start; refine the three lowest-loss starts for up to 240 additional sweeps each. Each capacity gets 1800 seconds; total process limit 3900 seconds. A limit is an unconverged result, not failure of the structural hypothesis. Exact smaller-Gram subspace solves and joint code/assignment projection retain the objective from the timing experiment.

Use the independently controlled shared-projection encoder. Before fitting, compare it against the original encoder on an actual full-vocabulary initialization. A failed replay stops interpretation. The old bound timing source and helpers remain unchanged.

**Convergence:** normalized-objective maximum intrinsic bank gradient at most 1e-6, with zero changed assignments over two final consecutive sweeps. Joint code solves occur every sweep. Preserve all iteration histories, gradients, assignment changes and time limits. Local convergence does not certify global optimality, correct group discovery, or a stable factorization. The executed planted counterexample is why this distinction is mandatory.

Registered predictions:

- `pred_a`: native original/reused encoder function disagreement at most 1e-8; saved FP32 compiled function disagreement at most 1e-5 in the full coefficient metric.
- `pred_b`: at least two promoted starts at each capacity meet the local convergence criterion.
- `pred_c`: each capacity improves squared full coefficient error by at least 10% relative to the exact global baseline at matched byte budget.

The null is weak or unconverged grouped fitting. Preserve failures of these bars even if one capacity or individual token subset looks useful afterward. Function cosine across promoted starts is descriptive, not proof of aligned semantic groups.

The price is local to this folded numerator. Original U may still be required for the residual/bias routes; retaining it there would erase a claimed whole-model U-storage saving. Applying the compressed reader to those routes instead requires separate fidelity evidence. Do not infer net whole-model savings from the local byte comparison.

**Compiled output and accounting:** save only the best final program at each capacity, with FP32 native residual readers, FP32 global/private token coefficients and mean, and int32 group IDs. Transform Hilbert-coordinate banks into native residual readers by a triangular solve; the metric inverse is not a runtime adapter. Reconstruct the saved program in the coefficient metric, including mean rounding, and check its error before recording an instrument pass. Program tensor payload matches the stated price; serialization headers are small additional file bytes. No giant coefficient tensor or vocabulary activation cache is saved.

**Next interpretation:** inspect convergence across starts, matched-cost gain, per-token residuals, and output-edge concentration before choosing a behavioral validation. Meaningful frozen candidates must still preserve the relevant normalization/readout interface and prove selective manipulation, OOD prediction and composition. A good output graph can still contain dense quadratic functions; it is not automatically a simpler multiplication circuit.
