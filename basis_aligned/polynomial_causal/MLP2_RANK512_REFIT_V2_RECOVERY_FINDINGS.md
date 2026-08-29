# MLP2 rank-512 free-factor refit: findings

Date: 2026-08-29

Decision artifact: `mlp2_rank512_refit_v2_recovery_result.json`

Receipt: `mlp2_rank512_refit_v2_recovery_receipt.json`

## Plain result

A learned rank-512 bilinear replacement recovered most of the final-language-model
damage caused by deleting MLP2, at exactly the same parameter and multiplication
price as the earlier failed 512-native-channel replacement. It was nevertheless
not faithful enough to replace MLP2 under the preregistered limits.

The best arm, `FULL512`, had held-out final-model results

| program | extra CE (nat) | teacher KL (nat) | centered-logit NRMSE | native top-1 agreement |
|---|---:|---:|---:|---:|
| native MLP2 | 0 | 0 | 0 | 100.00% |
| delete MLP2 (`ZERO`) | 0.16222 | 0.16894 | 0.24257 | 78.35% |
| retain 512 native products (`LOCAL512`) | 0.25419 | 0.26045 | 0.31036 | 74.31% |
| refit only output matrix and bias (`DOWN512`) | 0.10027 | 0.10669 | 0.20016 | 83.23% |
| refit all factors (`FULL512`) | **0.05147** | **0.05619** | **0.14538** | **87.52%** |
| same fit from random native support (`RANDOM512`) | 0.05739 | 0.06220 | 0.15270 | 86.74% |

Thus `FULL512` removed 68.27% of the CE damage and 66.74% of the KL damage of
deleting MLP2. It reduced CE damage by 48.67% relative to `DOWN512` and by 79.75%
relative to `LOCAL512`. All registered relative comparisons passed, including
simultaneous document-bootstrap lower bounds against `ZERO` and `LOCAL512`.

It did not pass the absolute limits: CE and KL had to be at most 0.02, logit NRMSE
at most 0.10, and top-1 agreement at least 90%. The observed values were 0.05147,
0.05619, 0.14538, and 87.52%.

## What function was learned?

The input (x\in\mathbb R^{1152}) is the RMS-normalized residual-stream vector
presented to MLP2 at one token position. The output is an
1152-dimensional replacement for MLP2's residual-stream write:

$$
\widehat f(x)=\widehat b+
\widehat D\left((\widehat Lx)\odot(\widehat Rx)\right).
$$

Here ⊙ means coordinate-by-coordinate multiplication. Both linear maps
\hat L and \hat R produce 512 numbers. Multiplying corresponding pairs gives
512 quadratic features, and \hat D maps those features back into the
1152-dimensional residual stream. All three matrices and the bias were learned in
`FULL512`.

Every rank-512 learned arm stored exactly 1,770,624 float32 coefficients (7,082,496
bytes), used 512 scalar products and three dense matrix multiplications per token,
and made zero calls to native MLP2. This fixed-price comparison is the simplicity
control: the improvement comes from a better coordinate system, not from purchasing
more products or parameters.

## What each control establishes

- `LOCAL512` asks whether simply selecting 512 of the model's original 4,608
  products works. It does not; it is even worse than deleting MLP2.
- `DOWN512` keeps those 512 original input products but relearns how they are mixed
  into the output, including the constant bias. Its large improvement shows that
  output mixing and the bias are essential parts of the compression.
- `FULL512` also relearns the two input directions that define each product. Its
  further large improvement shows that the original product coordinates are not the
  right indivisible units at this price.
- `RANDOM512` starts from a deterministic random set of native products and then
  learns all factors. It finishes only modestly behind `FULL512`. We therefore do
  not yet have evidence that the named native channels are privileged semantic
  atoms; optimization can find similarly useful mixed coordinates from a different
  starting support.

## Why the formal status is `optimization_failure`

The fitting objective was ordinary squared error between the candidate and native
MLP2 writes on native model states. On the training-development split, `FULL512`
still had write NRMSE 0.6866; the registered optimization threshold was 0.25. It
also reached the 1,200-step cap while its development curve was still improving, so
this run was not trained to convergence.

The experiment therefore does **not** prove that rank 512 is intrinsically too
small. It proves that this particular capped, activation-MSE generator did not make
rank 512 fully faithful. The surprisingly good final CE despite poor local write
NRMSE says that ordinary Euclidean write error weights many downstream-irrelevant
directions too heavily. A downstream/Fisher-weighted objective is now better
motivated than merely repeating the same local MSE fit.

The minimum-norm gauge check also failed its frozen absolute canary tolerance. The
post-gauge maximum absolute differences were 0.54--0.66, while the resulting canary
outputs had maximum magnitudes 875--1,190; for `FULL512`, the ratio was about
(4.9\times10^{-4}). This looks like float32 roundoff on large intermediate values,
not an algebraic failure of the rescaling, but the preregistered absolute
(10^{-4}) gate remains failed. We do not retroactively replace it with a relative
gate.

## Robustness and an odd distinction

`FULL512` CE damage was 0.04216, 0.04502, and 0.05147 on prefixes of 48, 96, and
192 held-out documents. KL was 0.05104, 0.05220, and 0.05619. Most metrics were
stable; centered-logit NRMSE moved from 0.13533 to 0.14538 between 96 and 192,
exceeding the 0.01 stability limit by about 0.000055.

The candidate's next-token accuracy was 41.58%, versus native 42.12%, a difference
of only 0.54 percentage points. Yet it chose the exact same top token as the native
model only 87.52% of the time. This is the distinction between task performance and
faithful emulation: `FULL512` is already a fairly good small predictor, but not yet
a faithful surrogate for what the original model predicts on each example.

## Scientific meaning and next experiment

This is the clearest positive result of the current MLP2 line: **a mixed rank-512
bilinear coordinate system is much better than 512 selected native coordinates at
the same executable price.** It validates joint factorization as an entry point.
It does not supply semantic labels for the 512 factors, prove OOD transport, compose
with compressed MLP0/MLP1, or justify selective removal.

The strict project ledger therefore does not move. The immediate high-information
test is a fresh-row composition telescope containing MLP0-C512 alone, this frozen
`FULL512` alone, and both together. Its interaction term will say whether the two
compressions expose a compatible interface or whether live downstream computation
was compensating for each independently. In parallel, the next MLP2 generator
should optimize a downstream-sensitive metric rather than unweighted local write
MSE and must use a new held-out evaluation role.
