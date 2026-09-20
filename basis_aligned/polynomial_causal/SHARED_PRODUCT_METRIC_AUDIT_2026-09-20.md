# Shared upstream products: coefficient versus functional sparsity

Keeping the full input span and pruning shared MLP16 products does not yet preserve the extracted parent component. However, a trace-aware functional metric materially improves on coefficient-Frobenius pruning. Both are restricted native-product baselines, not searches over new quadratic features. All reported native panels are opened.

## Object and metrics

Each atom is T_k=C[:,k] outer sym(L_k outer R_k). Products are computed once and reused by every downstream reader. C=D16 for the whole write; C=diag(sqrt(abs(parent coefficients))) readers^T D16 for the parent dictionary. This latter weighting describes upstream quadratic-reader fidelity, not exact fidelity of the final quartic composition.

If K_kl=<T_k,T_l> and t_k=L_k dot R_k, the isotropic Gaussian functional Gram is F=2K+(C^T C) elementwise (t t^T). For uniform fixed-radius sphere inputs, the same Gram has a common positive scalar factor, so the pruning order is identical. Actual normalized native inputs need not be isotropic. The trace term charges quadratic feature means, which coefficient Frobenius error can underweight.

Relative errors are L2 prediction error / target L2 norm. Maximum error includes scalar predictions and intervention-change predictions at full MLP16-write scales0/.5/1/1.5. Native targets recompute the actual suffix. Candidate programs explicitly execute attention17 and normalization; h16/x0/v1 remain upstream ports. The 10% error target and20% storage saving must both hold.

## Results

| Metric / support | Products | Full storage fraction | Max calibration error | Max validation error |
| --- | ---: | ---: | ---: | ---: |
| global_products | 256 | 0.379 | 2.559 | 2.818 |
| global_products | 1024 | 0.489 | 2.554 | 2.795 |
| global_products | 2304 | 0.671 | 2.546 | 2.776 |
| parent_products | 256 | 0.379 | 2.392 | 2.643 |
| parent_products | 1024 | 0.489 | 2.293 | 2.480 |
| parent_products | 2304 | 0.671 | 2.287 | 2.465 |
| random_products | 256 | 0.379 | 2.304 | 2.578 |
| random_products | 1024 | 0.489 | 1.665 | 1.855 |
| random_products | 2304 | 0.671 | 0.764 | 0.847 |
| global_sphere | 256 | 0.379 | 2.663 | 2.964 |
| global_sphere | 1024 | 0.489 | 2.575 | 2.828 |
| global_sphere | 2304 | 0.671 | 2.514 | 2.724 |
| parent_sphere | 256 | 0.379 | 0.758 | 0.868 |
| parent_sphere | 1024 | 0.489 | 0.736 | 0.814 |
| parent_sphere | 2304 | 0.671 | 0.666 | 0.722 |

Both runs select the dense fallback. At2304 products, parent trace-aware pruning outperforms the matched single random support, but still has72% maximum validation error. This is not success against the10% gate. Random-support variability has not been measured, so one seed does not establish statistical superiority.

## Red-team evidence

- Full-support replay passes in both experiments, including changed summation order. The three existing atom-pruning tests pass.
- An independent81-point product Gaussian quadrature evaluates all pairwise atom functions directly. It agrees with the analytic functional Gram within7.03e-16 relative error. Omitting trace gives25% error in that planted control, so the check detects the relevant mistake.
- Absolute squared numerators and target norms are recorded, including intervention norms; these failures are not relative errors divided by a near-zero target.
- The metric correction changes empirical rankings. v632 is evidence against that specific coefficient-metric greedy selector, not against every native subset, scalar refit, or sparse new-feature program.
- No fitting or model updates: each experiment uses16native forwards and208local replays. The same panels and random seed allow paired comparisons, but are not fresh confirmation.

## Consequence for discovery

Coefficient sparsity is not enough: select interactions under a declared polynomial/functional metric and separately evaluate native causal effects. The improved baseline still cannot reach a small accurate program by deleting existing channels. New quadratic features, cancellation-aware combinations, and cross-branch reuse remain the next substantive search space. Do not repeat native-support budget sweeps or relabel this result as a lower bound on DAG simplicity. Semantic selectivity and useful reuse remain unproven for the parent component.

[v632 receipt](../bilinear_quotient/circuits/followups/recent_sparse_products_v632_result.json) · [v633 receipt](../bilinear_quotient/circuits/followups/recent_sparse_products_v633_result.json) · [quadrature control](ISOTROPIC_ATOM_METRIC_CONTROL.json)
