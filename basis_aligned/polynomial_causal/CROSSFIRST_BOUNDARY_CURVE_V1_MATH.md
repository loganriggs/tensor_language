# Locating composition error with additive-state boundaries

13 September2026. This follows the native MLP9 precision discriminator. The path, child and remainder remain fixed; no data fitting or new factorization is used.

## What was measured

Write the native and child/parent/remainder removal states after block l as N_l,C_l,P_l,R_l. The additive state is A_l=C_l+R_l-N_l, and the accumulated mixed state is M_l=P_l-A_l. Each separate intervention resets the parent-removal trajectory to A_l at one boundary and executes the remaining native model. This removes accumulated interaction while preserving the two individual state changes.

All six replay comparisons pass exactly: four native arms, boundary9 versus prior measured-native MLP9 correction, and boundary17 versus the earlier final additive-state readout, separately regional and FineWeb. 2080forwards took21.55seconds.

The early sufficiency hypothesis fails: no boundary9–12 leaves at most half the original final interaction norm in all four regional groups. The monotonicity hypothesis also fails: moving the reset later sometimes increases the remaining interaction. Those are valid negative findings, not failed replay instruments.

| Regional group | After9: remaining/original interaction | After11 | After16 | After17 |
|---|---:|---:|---:|---:|
| Old near | 89.7% | 67.3% | 78.8% | 25.5% |
| Near-message | 88.5% | 57.5% | 56.3% | 18.1% |
| Near-person | 84.6% | 42.8% | 37.4% | 12.3% |
| Distant | 86.2% | 51.0% | 48.0% | 8.1% |

This points to the last block as a useful next decomposition target. It does not follow from the earlier activation-norm census alone. FineWeb curves differ: final resetting can worsen remaining interaction, consistent with internal/readout cancellation. No universal favorable effect of deeper resetting is claimed.

[Native result](CROSSFIRST_BOUNDARY_CURVE_V1_RESULT.json), [preregistration](CROSSFIRST_BOUNDARY_CURVE_V1_PREREGISTRATION.md).

## A more precise interpretation of adjacent boundaries

The preregistration correctly warned against treating adjacent cumulative-reset differences as an additive attribution on the native parent trajectory. However, they do have an exact local-generation interpretation under an explicitly different background convention.

Let F_l map a full sequence state through block l, retaining the same shared initial state x0 and first-layer value v1. These shared inputs are unchanged by our head9 interventions. Let S_l be the suffix after that block, including the final readout.

Define the locally generated mixed response around the additive input:

$$
G_l=F_l(C_{l-1}+R_{l-1}-N_{l-1})
-F_l(C_{l-1})-F_l(R_{l-1})+F_l(N_{l-1}).
$$

Since the individual states follow their original trajectories,

$$
G_l=F_l(A_{l-1})-A_l.
$$

The accumulated mixed response splits exactly into local generation and propagation of previously existing mixed input:

$$
M_l=G_l+T_l,
\qquad
T_l=F_l(P_{l-1})-F_l(A_{l-1}).
$$

Our adjacent boundary experiments already measure

$$
Y_{l-1}-Y_l
=S_l(F_l(A_{l-1}))-S_l(A_l)
=S_l(A_l+G_l)-S_l(A_l).
$$

Thus the adjacent contrast is the actual causal effect of adding G_l at the **additive output background A_l**. It is not its effect at P_l, and it is not a unique attribution of final nonadditivity to layer l. Different layers use different backgrounds, and their state-level contributions need not compose as independent semantic circuits.

At the final block17, this comparison is especially direct because S17 is just the known final normalization/unembedding/softcap/metric. The local-generation contrast has 33.9–59.7% of the original regional interaction norm, with cosine0.892–0.990 relative to it. Its aligned projection is31.7–53.3%. These are norms and projections of outcome vectors across prompts, not percentages of variance explained or fractions of a globally additive causal decomposition.

The strongest alternative to the naive last-layer story is nonlinear propagation or cancellation rather than a single isolated local circuit. The identity above executes a discriminator: it identifies a finite local mixed-generation effect in the additive-input convention, while retaining the separate propagated term and forbidding an attribution on the original parent background. A synthetic signed bilinear/RMS residual-chain control verifies the partition to2.96e-16 and the nonlinear suffix effect identity to4.99e-15. Saved native adjacent outcomes provide the reported physical contrast without new forwards.

[Executable algebra and scorer](crossfirst_adjacent_boundary_identity_v1.py), [control and native outcome audit](CROSSFIRST_ADJACENT_BOUNDARY_IDENTITY_V1_RESULT.json).

## Next decomposition decision

Split F17 into its residual, attention17 and MLP17 computations under this same additive-input convention. The residual term is affine and cannot itself generate a mixed term. Attention changes the input reaching the final bilinear MLP, so merely subtracting native attention and MLP mixed-output norms is insufficient. Evaluate the additive input explicitly, retain its normalization, and distinguish attention-generated mixed input from the MLP's response to it and the MLP's own two-input cross term. The existing MLP17/unembedding dossiers and earlier final-layer response methods must be reused before implementing that test.

This is still conditional circuit analysis on reused panels. It improves the definition of nonlinear composition, not autonomous extraction, fresh/OOD prediction, or a claim that all final-layer structure is now understood.
