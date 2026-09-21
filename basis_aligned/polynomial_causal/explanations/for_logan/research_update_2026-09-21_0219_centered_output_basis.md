# Choosing output directions from the centered operator helps, but capacity still matters

21 September 2026, 02:19 UTC.

Selecting output directions from the original centered bilinear operator improves conditional coverage at the same width. This confirms that the output objective matters. It does not yet produce a better executable circuit: the measurements below use exact native values projected into candidate output spaces.

## Three choices of output space

We compare256 output directions chosen from:

- **Paired-output variation:** the original basis, selected from calibration outputs on native paired inputs.
- **Centered operator:** the exact trained bilinear weights, weighted by separate centered calibration covariances for the two input slots.
- **Isotropic operator:** the same trained weights, with identity input metrics.

We also test512 centered-operator directions as a capacity control. No diagnostic-panel outputs are used to construct these spaces.

The folded operator is

$$
F(n,m)=C[(Ln)\odot(Rm)+(Rn)\odot(Lm)].
$$

For independent centered input slots with covariance matrices $M_n,M_m$, its channel second moment is exactly

$$
\begin{aligned}
H={}&(LM_nL^\top)\odot(RM_mR^\top)
+(RM_nR^\top)\odot(LM_mL^\top)\\
&+(LM_nR^\top)\odot(RM_mL^\top)
+(RM_nL^\top)\odot(LM_mR^\top).
\end{aligned}
$$

With output metric square root $S$, we take leading eigenvectors of $SCHC^\top S$. This contracts the joint function; it is not an independent compression of its factor matrices. A Cartesian-product toy verifies the covariance formula to relative error $2.65\times10^{-16}$.

Independent centered marginal inputs define the fitting metric. They are not the same distribution as the native same-token source/context interventions used for evaluation.

## Context-only projection errors

Each entry gives **linear vocabulary-centered error / native final-logit effect error**, on the same reused diagnostic documents and donors.

| Output basis | FineWeb | Related code |
|---|---:|---:|
| Paired256 | 44.48% / 43.00% | 31.33% / 30.35% |
| Centered256 | 40.24% / 39.33% | 27.39% / 26.97% |
| Centered512 | 29.16% / 28.59% | 19.66% / 19.45% |
| Isotropic256 | 55.78% / 69.29% | 50.96% / 62.99% |

Centered256 improves both domains at matched width. Its source-only linear projection errors also improve: FineWeb22.95% to20.57%, code17.89% to15.89%. All registered covariance/instrument, context-improvement and source-preservation checks pass.

The isotropic basis is much worse for these interventions, despite being optimized for its own coefficient metric. This supports covariance-informed weighting for this task; it does not make isotropic coefficient error an invalid metric for other questions. Unlike the preceding comparison, the isotropic basis's native final nonlinearities increase relative error substantially. The effect of those nonlinearities depends on the candidate.

## How much room remains for product approximation?

An output projection is an oracle lower bound in the linear metric. A real decomposition must also compute the retained coordinates. Orthogonality gives

$$
E_{\mathrm{total}}^2=E_{\mathrm{omitted}}^2+E_{\mathrm{inside}}^2.
$$

For a30% total-error planning target, Centered512's FineWeb floor of29.16% leaves only **7.07% of full-target norm** for inside-space computation error, or7.39% relative to the projected target. At256 directions, the measured FineWeb floor already exceeds30%, so no refit restricted to that space can meet this target.

These are fixed-space bounds for this linear metric on this panel. They are not impossibility results for every256-dimensional space, for nonlinear final effects, or for discovering smaller task-specific circuits. “Possible in the span” is not proof that a cheap product decomposition achieves it.

The next factorization must therefore budget output omission and computation error jointly. A promising follow-up is a matched-product comparison that allocates capacity differently across output width and per-output interactions, scored on the conditional operator as well as aggregate behavior. Simply reusing the old narrow output basis would retain a known bottleneck.

No semantic feature identity, selective task intervention, cross-module composition or broad external OOD claim follows from this oracle screen. Those remain requirements of the full goal.

Evidence: `MIDPOINT_CENTERED_OUTPUT_BASIS_V1.json`, its raw native records, `MIDPOINT_CENTERED_OUTPUT_BASES_V1.pt`, and the successor CPU calculation `MIDPOINT_CENTERED_OUTPUT_ERROR_BUDGET_V1.json` under `direct_tensor_match`.
