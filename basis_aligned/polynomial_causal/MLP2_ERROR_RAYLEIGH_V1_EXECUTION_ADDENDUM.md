# MLP2 error-Rayleigh v1 — execution addendum

**Frozen before response access:** 2026-08-29 23:48 UTC

This addendum resolves implementation details left open by
`MLP2_ERROR_RAYLEIGH_VALIDITY_PILOT_PREREGISTRATION.md`. It does not change its
scientific gates.

## Role lifecycle

The same source-closed collector supports `DESIGN` and `HELDOUT`, but they are
separate create-only transactions. `DESIGN` may run after an independent audit of
the exact collector sources. `HELDOUT` must refuse to start unless a receipt-last
predictor artifact exists that binds the DESIGN ledger, feature normalizers, ridge
grid, selected ridge values, and all three fitted predictors. Thus no HELDOUT model
response can influence feature construction or model selection.

## Physical error and endpoint replay

For each background $b$ and program $P$, the collector first obtains the native MLP2
write $f_2(z_b)$ and program write $P(z_b)$ on every position. The actual error is

$$E_{P,b}=P(z_b)-f_2(z_b).$$

Actual perturbations use endpoint-preserving linear interpolation between these two
stored writes. This is algebraically $f_2(z_b)+\alpha E_{P,b}$, while ensuring that
the implementation returns the exact program endpoint at $\alpha=1$. A separately
executed physical-program forward must match that endpoint bit-for-bit in capped
logits and complete attention-5/6 writes.

## Negative controls

Controls are constructed separately for each program/background over the complete
32-document role.

- `DERANGED` circularly moves each whole-document error field to the next recipient
  and rescales it to the recipient error norm. No document receives its own error.
- `COV_RANDOM` subtracts the document mean error field, forms deterministic Gaussian
  mixtures of the other 31 document fields with a zero diagonal, and rescales each
  mixture to the recipient error norm. This samples from the empirical document
  covariance span without retaining recipient matching.

The frozen seed is `2026082951`, separated deterministically by program, background,
and role. Controls are evaluated at both signs of `1/16` and `1/8`. The actual error
also receives the finite `alpha=1` physical replay.

Taylor/Fisher validity is direction-agnostic: a random direction may legitimately
pass the small-amplitude tangent and Fisher/KL gates. The preregistration's control-
failure requirement therefore applies to **prediction of the true finite composition
target**, not to local calculus. The true-error DESIGN predictor is applied unchanged
to deranged/random background-contrast features; each null family must fail either
the held-out predictor gate or the three-program finite-interaction gate. No claim is
made that a valid derivative ceases to be a valid derivative when its direction is
randomized.

## Stored sufficient statistics

Raw vocabulary logits are reduced in memory and are not published. The ledger stores,
per source document, program, background, and control:

- local mean-squared error;
- CE directional derivative;
- categorical-Fisher logit quadratic;
- separate attention-5 and attention-6 normalized response energies;
- teacher KL and CE change at each sign of each amplitude.

For actual errors it additionally stores direct-program and injected-alpha-1 CE
changes plus exact replay diagnostics. Capped logits use positions `64:256`;
attention responses use the complete `0:256` write field. Reduction without retaining
the raw fields does not weaken any frozen gate and avoids a multi-terabyte artifact.

The source document remains the inference unit. No token is treated as an independent
replicate.

## Predictor unit and target

There is one regression row per `(source document, program)`, hence 96 rows but only
32 independent DESIGN document clusters. For program $P$, the target is the document-
level finite interaction

$$
i_{d,P}=
\big[CE_{d,C}(\alpha=1)-CE_{d,C}(0)\big]
-\big[CE_{d,N}(\alpha=1)-CE_{d,N}(0)\big],
$$

where $C$ is the C512 background and $N$ is native MLP0. Predictor inputs are the
corresponding C512-minus-native contrasts. The three frozen predictor families are:

1. intercept plus local-MSE contrast;
2. intercept plus CE-linear and $q_{\rm logit}$ contrasts;
3. family 2 plus separate $q_5$ and $q_6$ contrasts.

Program identity is not a feature. Feature means and nonzero standard deviations are
fit on DESIGN only. Ridge values are
`[0, 1e-6, 1e-4, 1e-2, 1, 100]`; selection uses leave-one-document-out clustered
cross-validation, minimizes mean squared error averaged over the three program rows
inside each held-out document, and breaks exact ties in favor of the larger penalty.
The selected penalty, normalizers, and coefficients are frozen receipt-last before
HELDOUT can open. Bootstrap resampling is always by the 32 source documents with all
three program rows kept together.
