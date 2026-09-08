# Temporal/is-was residual-identity weight-route reuse analysis — 2026-09-08 10:42 UTC

## Decision this note prepares

The queued residual-skip x module-response factorial asks whether the missing effect of the
selected is-was writers `L7H7 + L9H4` is carried mainly by the raw block-10 residual state, by
the downstream `A10--M17` module responses, or by their interaction. This note freezes the exact
weight route to use only if prediction C passes: `R1M0` must carry a majority of the writer effect
on both FIT and HOLDOUT. It does not inspect that outcome and does not authorize this route if C
fails.

This route advances computational specification, extraction, and selective manipulation. It is
not a low-rank, DAS, reconstruction, or compression experiment.

## Exact controlled recurrence

Let `x_l` be the raw residual entering block `l`, let `x_0` be the unchanged token embedding
skip, and let `a_l` and `m_l` be the complete attention and MLP outputs. In the deployed model,
the relevant recurrence has the form

```text
u_l = lambda_l[0] x_l + lambda_l[1] x_0
x_(l+1) = u_l + a_l + m_l
```

where the module outputs are evaluated at their live intermediate states in an ordinary run.
The factorial's `R1M0` arm instead installs the writer-run `x_10` while clamping every complete
`a_l,m_l`, for `l=10,...,17`, to its native-run tensor. Because the input text and therefore
`x_0` are unchanged, subtraction of native from `R1M0` eliminates the embedding skip and all
module terms:

```text
Delta x_(l+1) = lambda_l[0] Delta x_l,
Delta x_18 = g_10 Delta x_10,
g_10 = product_{l=10}^{17} lambda_l[0].
```

This is an exact scalar weight translation in the physical residual gauge, not a tangent
approximation. If `R1M0` wins, a direct intervention that installs

```text
x_18^direct = x_18^native + g_10 (x_10^writer - x_10^native)
```

at the final residual boundary must match the dynamic `R1M0` arm before final RMSNorm and
unembedding, up to deployed precision. Native `x_18` self-patching and dynamic-versus-direct
logit equality are mandatory instrument controls.

## Valid precedent and excluded authority

The closest valid precedent is
`temporal_iswas_v15_entry12_rank2_direct_residual_final_head_route_v1_result.json`. It passes all
five registered predictions, including the exact recurrent-product law from entry 12, direct-route
sufficiency, and P/C selectivity, at its exact 24 differentiable-forward price. It used the same
physical product of downstream `lambda_l[0]` values to show that the residual state surviving a
complete downstream-write reset is the direct final-head route. The present test is not a new
algebraic mechanism: it prospectively applies that established identity to a different, causally
selected writer boundary at block 10 and requires equality to the newly queued `R1M0` arm.

The valid
`temporal_auxiliary_will_had_h3_rank8_weight_direct_residual_route_v2_result.json` receipt already
tests the same kind of controlled identity transport from a block-11 attention-head output. It
passes all four registered predictions, uses the product of `lambda_l[0]` for layers 12--17,
matches the corresponding dynamic all-module clamp within `2.861e-6` maximum logit error, and
retains `0.992--1.030` of the rank-8 parent behavior. Its final-residual state maximum absolute
difference is `0.01171875`; therefore the successor must record both absolute and scale-relative
state error rather than treating a raw BF16-sized absolute discrepancy as behavioral failure.
That receipt validates the dynamic-versus-direct control pattern. Our route is simpler because
the boundary object is already a residual delta, so no attention `c_proj` contraction is needed.

The later
`temporal_iswas_v15_entry12_rank2_normalized_weight_reader_atlas_v1_result.json` is explicitly
excluded as scientific authority. It is terminal `invalid`: it executed 10 differentiable
transformer forwards against a frozen maximum of 6. Its tangent and reader rankings may suggest
diagnostics, but none may be used to select, validate, or claim a reader in this successor.

## Final readout and claim boundary

After direct residual transport, the deployed readout is still nonlinear:

```text
n(x) = RMSNorm(x)
z_j(x) = 30 tanh(w_j^T n(x) / 30).
```

The exact direct intervention therefore tests the whole physical route through RMSNorm,
unembedding, and soft cap without linearizing them. A later exact secant can decompose the
observed token-pair margin, while the RMSNorm Jacobian can be retained only as a descriptive
local reader diagnostic. Static weight alignment or a tangent cosine cannot identify a reader
by itself.

## Prospective successor if the identity branch passes

1. Reuse the factorial's immutable authorities and exact original FIT/HOLDOUT rows.
2. Capture native and selected-writer `x_10`, native `x_18`, and the frozen block lambdas.
3. Execute native, dynamic `R1M0`, native final-residual self-patch, and direct `g_10 Delta x_10`
   arms. Require dynamic/direct logit agreement and report scale-relative residual-state error.
4. Freeze the direct route on original FIT, then test the same formula on original HOLDOUT and
   fresh OOD without refitting or choosing coordinates.
5. Only after exact route equality, remove or swap the transported `Delta x_10` and require the
   intended is-was change with low temporal-command collateral.

Kill this route if factorial C fails, if the direct installation does not reproduce dynamic
`R1M0`, or if removal/swap is not selective. In those cases use the module-factor or explicit
state-by-response interaction branch selected by the factorial rather than tuning this identity
formula.
