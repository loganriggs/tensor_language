# Learning new output-local quartic products: instrument and optimizer evidence

22 September 2026, 02:32 UTC. This is preparation for a native residual fit, not a successful circuit result. The new factors are trained against the difference between the true selected quartic weights and a frozen CP parent. Eight new quartic products are assigned to each of output coordinates4–15; dominant coordinates0–3 remain unchanged at the polynomial interface.

The loss uses exact Gaussian Gram matrices and native cross contractions, subtracting the frozen parent's cross contractions. Each output's eight readout coefficients are solved exactly up to ridge regularization. Differentiating the resulting profiled objective trains the product directions. Separate output destinations prevent an improvement in one small output from directly rewriting the dominant coordinates. The representation still permits nonunique products and has no semantic guarantee.

Ten independent loss/gradient checks cover five native two-bilinear-layer toy structures and both pure Gaussian and mixed Gaussian/coefficient objectives. Gaussian reference values come from quadrature; coefficient references enumerate symmetric input slots. The largest gradient discrepancy is6.8e-14. Five planted targets with known two-atom-per-output residuals have capacity-witness errors below1.1e-7. These distinguish a representational witness from successful discovery.

The first20fits used100steps, equal rate0.03 and two starts. Adam did better on every family, but residual errors remained several percent or higher. A subsequent fixed40-fit sweep used500steps and rates0.01/0.1 for each optimizer:

| Optimizer | Rate | Median relative error across10cases | Worst error | Cases below1% |
| --- | ---: | ---: | ---: | ---: |
| Adam | .01 | 3.31% | 15.99% | 0/10 |
| Adam | .1 | 0.69% | 3.95% | 6/10 |
| Muon | .01 | 0.83% | 4.99% | 6/10 |
| Muon | .1 | 2.61% | 5.01% | 1/10 |

The short pilot did not establish an optimizer winner. With more steps and a different rate, Muon is competitive. Native hyperparameters are frozen at the best median rate for each optimizer: Adam0.1 and Muon0.01. These are small planted problems, not a guarantee that either rate is optimal in1152dimensions. Native comparison retains both optimizers and two starts, with equal250update budgets and training-objective checkpoint selection.

An actual-size native CPU gradient test used12outputs,96new atoms,512parent atoms and1152input dimensions. Forward/backward took4.39seconds; process peakRSS was3.45GB. Native finite differences at two step sizes agreed to1.77e-4 and1.59e-5 relative error; normal-equation residual was3.4e-16. This is a CPU instrument profile, not a GPU performance estimate. The managed runner's model-free dry run also exercises all grouping/readout axes and1152-dimensional export assembly.

The upcoming model adds288variable products and442,464coefficients. It is a capacity-discovery experiment, not a claim of a smaller final circuit. If the learned additions improve both values and finite responses, they become candidates for the second-stage sharing and pruning search, then native intervention tests. Output preservation at the numerator does not imply unrelated final logits remain unchanged after normalization.

[Native preregistration](LOCAL_QUARTIC_RESIDUAL_NATIVE_PLAN_V1.md) · [Frozen inputs and rates](LOCAL_QUARTIC_RESIDUAL_NATIVE_INPUTS_V1.json) · [Gradient controls](LOCAL_QUARTIC_RESIDUAL_CONTROLS_V1.json) · [Capacity witnesses](LOCAL_QUARTIC_RESIDUAL_WITNESSES_V1.json) · [Initial optimizer pilot](LOCAL_QUARTIC_RESIDUAL_TOYS_V1.json) · [Longer optimizer sweep](LOCAL_QUARTIC_RESIDUAL_TOY_SWEEP_V2.json) · [Actual native CPU profile](LOCAL_QUARTIC_RESIDUAL_NATIVE_PROFILE_V1.json).
