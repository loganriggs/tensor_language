**One proposed square repairs the registered worst-read constraint; random capacity does not**

The fixed-bank coefficient fit had a numerical lower bound above1.5, against an acceptance target1. We differentiated its weighted active constraints with respect to the full quadratic form. Existing-direction gradients are near zero (maximum1.32e-20), while the largest absolute matrix eigenvalue is0.0118143. Its eigenvector supplies a new squared linear feature. The seeded random direction has slope1.97e-5. This is a proposal mechanism; its success requires actual refitting and evaluation.

Both arms append one square at initial coefficient zero, exactly preserving the previous graph. The directed arm reaches a global maximum normalized squared ratio **0.982203** after two separation rounds in18.33seconds. All eight original fidelity requirements and the added component-three/second-read worst-input requirement pass. The random arm runs eight rounds in30.87seconds and finds no better global candidate than the original1.53396. We retain incumbents using exact separation, so bad finite-cut trials cannot be reported as improvements.

The directed graph has1121products,1,122,723source multiplications and1,133,134stored float coefficients. This is1155more multiplications and1154more coefficients than the earlier graph, a separately priced extension rather than a result at unchanged capacity. Both experimental arms have identical capacity. Source arithmetic is15.62%below the original1,330,560comparison program; this is not a whole-model speedup.

The saved graph replays its solved form within3.86e-15. Original-interface component errors are1.727%,2.174%,8.345%; covariance-shaped coefficient error7.632% and original-coordinate error57.731%. FP32 component drift is below8.2e-6. Generated-interface errors on the original448states are3.10%,2.83%,8.91%.

**Stronger resource-matched baselines**

Width324in every pair exceeds candidate storage by86floats, despite fitting its arithmetic budget. We therefore compare every placement of widths323/324/324, in both isotropic and covariance-shaped geometries, with the common writer stored once. These six baselines each use1,121,505source multiplications and1,132,066floats, within both candidate budgets.

The candidate passes generated-interface fitting-state value comparisons against all six. Its targeted worst-read error is at most1.09212times the strongest relevant baseline, below1.10. Other reads are unchanged, so their earlier worst-input failures against the isotropic baseline are **not** repaired by this result. Adaptive fitting effort also remains unequal between graph and algebraic baselines.

This is evidence that a constraint-guided new feature can achieve something unavailable to coefficient changes in the old dictionary, with a same-capacity random control. It is not yet transfer evidence, semantic identification or a full folded circuit. The next managed screen freezes this graph and the stronger baselines on the two already-opened panels, evaluating both declared scalar interfaces. A subsequent fresh claim would require untouched data and actual behavioral checks.

[Directed fit](ROBUST_DIRECTION_DIRECTED_V1.json) · [Random control](ROBUST_DIRECTION_RANDOM_V1.json) · [Export and six-baseline audit](ROBUST_DIRECTION_EXPORT_AUDIT_V1.json) · [Frozen transfer plan](ROBUST_TRANSFER_PLAN_V1.json).
