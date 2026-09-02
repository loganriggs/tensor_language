# What we currently understand about MLP2

*Consolidated 2026-09-02. The proposed common-causal-measurement run is now paused in favor of using the 62 known
behaviors to discover components inside attention0.*

## Short version

MLP2 is not well approximated by selecting a subset of its 4,608 native product channels. At 512 channels, every
tested native-channel selection was worse than deleting MLP2, showing that the channels rely on coordinated
cancellation. A learned 512-product bilinear program is much better: it adds about `.05` CE rather than `.16` for
deletion, at 1,770,624 stored float32 values, but it still misses the fidelity threshold for replacing MLP2.

Five executable programs at exactly that same rank and parameter count already exist: `DOWN512`, `FULL512`,
`RANDOM512`, `CONTINUE512`, and `ROBUST512`. They differ in which factors were trained, initialization, and whether
training included states produced by an MLP0 approximation. Their ordinary CE/KL behavior is known, but their effect
on the current common tests—preserving attention16 mean ablation and combining with the MLP16 replacement—has not
been measured. This makes them a plausible new comparison group, provided we freeze genuinely new documents and the
candidate metadata before generating those outcomes.

## Native computation and price

MLP2 receives `x ∈ R^1152` and computes

`y = Down[(Left x) elementwise-times (Right x)] + bias`,

with `Left, Right: R^1152 -> R^4608` and `Down: R^4608 -> R^1152`.

It stores 15,926,400 numbers and computes 4,608 scalar products per token. The 4,608-dimensional middle vector is the
product-feature vector; it should not be confused with an input rank, output rank, or whole-layer tensor rank.

## Dense coefficient-tensor factorization is closed

The implicit folded-tensor study computed the exact singular spectra of the MLP2 bilinear coefficient tensor without
materializing the `1152 × 1152 × 1152` tensor. All three mode ranks are numerically 1,152. Retaining 90% of ordinary
coefficient Frobenius energy requires roughly 826 output and 922 input directions. Dense Tucker programs at those
ranks cost hundreds of millions of values and hundreds of thousands of products—larger and slower than native MLP2.

This closes ordinary coefficient-Frobenius HOSVD/Tucker for MLP2. It does not close activation-weighted,
downstream-weighted, CP, or learned small-product programs.

Primary finding: [`MLP2_IMPLICIT_FOLDED_TENSOR_V1_FINDINGS.md`](../MLP2_IMPLICIT_FOLDED_TENSOR_V1_FINDINGS.md).

## Selecting 512 native product channels is also closed

The CMR study physically retained 512 of the 4,608 native products and their corresponding Left, Right, and Down
coefficients, with the omitted average folded into the bias. It compared local-output, RMS, activation-mass,
downstream-suffix, deranged, and random selection rules at equal price.

On 192 held-out documents:

| program | extra CE |
|---|---:|
| delete MLP2 | `.16235` |
| local selection | `.26528` |
| RMS selection | `.26549` |
| downstream-suffix selection | `.28920` |
| mass selection | `.30701` |
| random selection | `.36629` |

Every 512-channel selection was worse than deleting MLP2. The joint distortion of omitted channels was 1.835 times
the sum of their singleton distortions, and small signed edits did not predict the finite deletion. Native product
channels therefore are not independent circuit atoms at this scale; more selection rules or a native-channel K sweep
would duplicate a closed strategy.

The suffix-selected program mildly improved a copy-positive subset while damaging other positions. That remains a
possible copy-circuit extraction clue, not a faithful MLP2 replacement.

Primary finding: [`MLP2_CMR_V1_VALIDATION_FINDINGS.md`](../MLP2_CMR_V1_VALIDATION_FINDINGS.md).

## Learned 512-product coordinates work much better

The free-factor refit uses the same functional form as native MLP2 but with 512 learned products:

`y_hat = bias_hat + Down_hat[(Left_hat x) elementwise-times (Right_hat x)]`.

Each program has:

- `Left_hat, Right_hat: 1152 -> 512`;
- `Down_hat: 512 -> 1152`;
- one 1,152-number bias;
- 1,770,624 stored float32 values = 7,082,496 bytes;
- 512 products and three dense matrix multiplications per token; and
- zero native-MLP2 calls.

The original held-out comparison was:

| program | what was trained | extra CE | teacher KL | logit NRMSE | native top-1 agreement |
|---|---|---:|---:|---:|---:|
| `ZERO` | delete MLP2 | `.16222` | `.16894` | `.24257` | `78.35%` |
| `LOCAL512` | keep 512 native channels | `.25419` | `.26045` | `.31036` | `74.31%` |
| `DOWN512` | native selected Left/Right; refit Down+bias | `.10027` | `.10669` | `.20016` | `83.23%` |
| `FULL512` | learn Left, Right, Down, bias | `.05147` | `.05619` | `.14538` | `87.52%` |
| `RANDOM512` | same full fit from random native support | `.05739` | `.06220` | `.15270` | `86.74%` |

Thus mixed learned coordinates are far better than selected native coordinates at the same executable price.
`RANDOM512` being close to `FULL512` is evidence against treating the initial native channels as privileged semantic
atoms. The absolute replacement gate still failed (`CE, KL <= .02`, logit NRMSE `<= .10`, agreement `>=90%`), and the
local MSE training had not converged. The useful lesson is about the coordinate system, not an adopted compressor.

Primary finding: [`MLP2_RANK512_REFIT_V2_RECOVERY_FINDINGS.md`](../MLP2_RANK512_REFIT_V2_RECOVERY_FINDINGS.md).

## Interaction with an MLP0 approximation

Installing the earlier MLP0-C512 approximation increased `FULL512`'s marginal CE cost by about `.0086` nat on fresh
documents. This established a real positive interaction: fitting MLP2 only on native pre-MLP2 states does not make it
fully robust to a changed upstream trajectory.

Two equal-price continuations started from the same `FULL512` bytes:

- `CONTINUE512` received additional native-trajectory training;
- `ROBUST512` used a balanced objective on native states and states produced by MLP0-C512.

On a later 192-document evaluation, standalone CE was `.05127` for `FULL512`, `.04225` for `CONTINUE512`, and `.04220`
for `ROBUST512`. Joint with MLP0-C512, it was `.06199`, `.05210`, and `.05179`. Extra training helped, but the robust
objective did not beat the matched continuation control by the registered amount and did not halve the interaction:
absolute interaction remained `.00744`, above the `.005` target. Paired trajectory exposure under this optimizer was
therefore rejected as the explanation/repair.

Primary registration and result:
[`MLP2_TRAJECTORY_ROBUST_R512_V1_PREREGISTRATION.md`](../MLP2_TRAJECTORY_ROBUST_R512_V1_PREREGISTRATION.md) and
`mlp2_trajectory_robust_r512_v3_physical_eval_result.json`.

## What is closed and what remains open

Closed without a changed mathematical object:

- dense coefficient-Frobenius Tucker/HOSVD;
- selecting or ranking 512 native products as a faithful replacement;
- interpreting native product channels as independent circuit atoms;
- repeating the same capped activation-MSE rank-512 fit and calling nonconvergence a rank result; and
- the claim that paired MLP0/native trajectory exposure specifically fixes composition brittleness.

Still open:

- training a small-product program under a downstream-response, Fisher, suffix-logit, or common causal objective;
- coordinated block-term/CP programs rather than native-channel selection;
- copy-specific extraction from the suffix-selected signal with explicit collateral tests;
- why learned mixed products are useful, including whether their equivalence is determined by downstream readers;
- common attention16-ablation and MLP16-interaction measurements for the five existing equal-price programs; and
- OOD, selective removal, and circuit semantics for a successful program.

## Possible later comparison, not the current priority

The proposed comparison is not another MLP2 compression sweep. It would hold architecture, rank, product count, and
stored parameter count exactly fixed while changing the fitting method:

1. `DOWN512`: only output mixing and bias refit;
2. `FULL512`: every factor fit from native-selected initialization;
3. `RANDOM512`: every factor fit from random-support initialization;
4. `CONTINUE512`: `FULL512` with matched additional native training;
5. `ROBUST512`: `FULL512` with balanced native/MLP0-C512 trajectory training.

This could be a useful stress test for a learned approximation-robustness rule because bytes and rank are constant. Any predicted
difference must come from structural type, objective, conditioning, local fidelity, or interface robustness rather
than capacity alone.

It does not determine how to split a representation into components and does not use the 62 known behaviors to merge
features that have the same downstream effect. For that reason, the five-program consequence run is paused. The
current priority is the attention0 response-tensor plan described in the 00:24 explanation. This dossier is retained
to prevent duplicated MLP2 work and to make the comparison executable later if it becomes useful as a validation.

Before execution, a new bank entry must hash-pin all five tensors, verify identical shapes/prices and no native MLP2
calls, and record the already-known fit objectives without loading the new outcomes. The evaluation documents must be
disjoint from every program's fitting/selection/evaluation rows. The two common measurements and their thresholds must
be fixed before model execution. Failure should close this five-program comparison, not trigger more rank or site
tuning.
