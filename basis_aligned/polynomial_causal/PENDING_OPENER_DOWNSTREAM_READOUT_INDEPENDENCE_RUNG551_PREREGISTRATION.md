# Rung 551 preregistration: is the later response distinct from the closer readout?

**Frozen:** 2026-09-03 16:18 UTC, before R549 started or produced an outcome

## Question

If R549 selects and validates a later attention-head or MLP response, does that response constrain the pending-opener
interchange beyond directly writing the correct closing delimiter?

R549's cross-construction classification and answer-preserving controls establish a reproducible causal response.
They do not alone establish that the response differs from another copy of the closer-token output direction. R551
is a pre-outcome interpretation guard, not a second candidate-selection pass.

## Computation

For the R549 FIT-selected candidate, reconstruct the six pooled FIT transition templates $T_t$ from the saved causal
response vectors. Let the three raw closer-token contrasts be

$$
u_{ab}=W_U[a]-W_U[b],
$$

for parenthesis, square bracket, and quote closers. For an MLP write, these vectors already live in the
1,152-dimensional residual-write coordinates. For an attention-head output $y_h$ followed by output projection
$W_O$, pull each contrast into that head's 128-dimensional coordinates:

$$
\tilde u_{ab}=W_{O,h}^{\mathsf T}u_{ab}.
$$

The three pairwise contrasts span at most two dimensions. Compute an orthonormal basis $Q$ for their span and report

$$
r_t=\frac{\lVert QQ^{\mathsf T}T_t\rVert_2}{\lVert T_t\rVert_2}
$$

for all six transitions. $r_t^2$ is the fraction of template energy contained in the direct closer-readout span.

## Frozen decision

The selected R549 response qualifies as an independent second target for a later multi-output DAS only if:

1. R549's FIT-selected candidate passes all frozen SELECT validation bars; and
2. the median of the six $r_t$ values is at most 0.50.

Thus at most 25% of median template energy may lie in the direct closer-readout span. R551 independently recomputes
R549's reported median maximum absolute cosine with the three pairwise contrasts, but that pairwise statistic is a
cross-check rather than the decision rule.

If R549 has no validated candidate, or if the median span fraction exceeds 0.50, the strong null is that this atlas
did not provide a nonredundant downstream target. The response remains valid descriptive evidence; it cannot be used
to make an endpoint-trained DAS look multi-output. R551 loads checkpoint weights without a model forward, uses no
outcome split, and does not open FINAL_TEST/OOD.
