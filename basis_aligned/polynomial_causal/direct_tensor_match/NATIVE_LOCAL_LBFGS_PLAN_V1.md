# Native local quartic residual: separate L-BFGS comparison

22 September 2026, 05:09 UTC. Registered before execution.

Recover missing computations on the pure MLP16-to-MLP17 quartic path, viewed through the existing 16 output readers. Keep CP seed1001 fixed and add eight quartic atoms to each output4–15. Inputs are the full1152-dimensional normalized MLP16 state. Residual, attention, bias and other cross terms remain outside this selected polynomial. Outputs0–3 are protected. This is a component-recovery screen, not a semantic circuit or full-model replacement.

Use exactly the factors, frozen input hashes, output weights, ridge1e-6 variable-projection objective and starts25001/25002 in LOCAL_QUARTIC_RESIDUAL_NATIVE_INPUTS_V1.json. Only the optimizer changes. Fit no text labels. The Gaussian mean/covariance are calibration-derived; this is data-informed weight matching, not a wholly data-free metric.

L-BFGS: lr1, strong Wolfe, history50, max250 iterations, hard500 objective/gradient evaluations, tolerance_grad1e-12, tolerance_change1e-14. Divide each arm's objective by max(abs(initial objective),1e-10), matching the native baseline. Select the lowest finite fitting objective, including valid line-search trial points. Record both first251-evaluation and final checkpoints; export only final. The original Adam/Muon budget is251 objective evaluations and250 backward calls, so compute, memory and wall time are not identical. A constant target-energy term is omitted in all optimizers; percentage objective gain is not percentage reconstruction-error reduction.

Predictions: (a) every export relative drift<1e-4, profiled normal residual<1e-8, finite gradients, protected outputs exact; (b) both starts improve the initial profiled objective by at least10%; (c) both final starts reduce old-panel equal-small-output value RMS AND matched-response RMS by at least15% versus frozen parent. Null: better numerical optimization does not repair native component fidelity. Compare first251 and final with the separately registered Adam/Muon arms when all receipts exist. No post-hoc winner-only reporting.

Price: +96 quartic atoms, +288 variable products, +442464 stored coefficients. Total1824 products and2828384 coefficients including common writer. L-BFGS history alone is roughly354MB at float64; report actual peak GPU memory. This added capacity is identical across optimizers, not a simplification by itself. A passing candidate must subsequently survive larger-panel, native manipulation, OOD, sharing and simplification tests.

Five distinct planted wide quartic families motivated this comparison: scaled L-BFGS8/10 below5% error,7/10 within251 evaluations, versus tested wide Adam2/10. Those controls establish neither native performance nor optimality. Variable projection analytically eliminates linear coefficients but leaves a nonconvex direction problem: [O’Leary and Rust](https://www.cs.umd.edu/~oleary/software/varpro/varpro.pdf).

Execution uses the managed queue only. No existing queued scripts or frozen helpers are modified. Model-free dry run exercises the actual closure helper on a profiled quadratic and a small quartic with96 atoms/512 parent features/12 outputs, including packed export shapes. Existing native objective preflight covers actual1152-width contractions; this comparison introduces no new contraction formula.
