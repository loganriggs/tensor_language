# MLP2 CMR v1 FIT_SELECTOR margin and frequency calibration

Frozen before opening any new FIT_SELECTOR logits for this calibration and before
opening `VALIDATION` rows, logits, targets, finite candidates, losses, or outcomes.

## Role-only projection prerequisite

The historical token artifact serializes all four roles in one PyTorch container.
`torch.load` necessarily deserializes the complete container before a caller can
index `FIT_SELECTOR`; therefore directly loading it in the calibration would also
open the protected `VALIDATION` and `REPLICATION` token tensors to that process.

Before calibration authority, a separate source-closed, model-free projection must
load the historical container once, validate its receipt/manifest and exact
FIT_SELECTOR tensor hashes, clone only the FIT_SELECTOR tensors, delete the combined
object, and publish a create-only role-only artifact plus receipt.  This unavoidable
projection process technically deserializes all roles, but performs no model load,
loss, candidate construction, selection, or scientific measurement and publishes no
protected role.  The calibration process is forbidden from reading the combined
container and may receive only the role-only bytes.

## Purpose and authority boundary

The main preregistration requires an $\epsilon$ grid for the finite-logit margin
certificate and target-frequency strata.  Neither was materialized by the completed
SUFFIX fit, whose logits were discarded.  This calibration opens only the already
spent `FIT_SELECTOR` role and runs the exact native model.  It may publish native
margin summaries, a frozen $\epsilon$ grid, and FIT_SELECTOR target-token counts.
It may not construct a finite MLP2 candidate, compute next-token loss or accuracy,
open `VALIDATION` or `REPLICATION`, or authorize a scientific compression claim.

## Native margins and epsilon grid

At every published eligible position, let the native post-softcap logits be
$\ell\in\mathbb R^{50304}$.  Define the top-1 margin

$$
m=\ell_{(1)}-\ell_{(2)},
$$

where $\ell_{(1)}\geq\ell_{(2)}$ are the largest two logits.  Use all 31,505
eligible FIT_SELECTOR positions, with no document or position subsampling.

The published positive epsilon grid is the sorted exact union of:

1. fixed dyadic values $2^k$ for integer $k=-10,-9,\ldots,5$; and
2. one half of each empirical native-margin quantile at

$$
q\in\{0.001,0.002,0.005,0.01,0.02,0.05,0.10,0.20,0.30,0.40,
0.50,0.60,0.70,0.80,0.90,0.95,0.98,0.99,0.995,0.998,0.999\}.
$$

Nonpositive or nonfinite candidates are rejected; exact duplicate float64 values
are removed.  Quantiles use `torch.quantile` on the complete CPU float64 margin
vector with its default linear interpolation.  The dyadic values prevent an
unusually narrow empirical margin range from making the later certificate grid
vacuous.  The quantile values give resolution where the first certificate term
$\Pr(m\leq2\epsilon)$ changes.

On validation, every candidate reports

$$
\max_{\epsilon\in\mathcal E}
\left[1-\Pr(m_{\rm native}\leq2\epsilon)
-\frac{D_2}{\epsilon^2}\right]_+,
$$

where $D_2$ is measured from that candidate's actual joint finite post-softcap
logit errors.  The grid is not changed using candidate outcomes.

Precisely, for $N$ eligible positions and the complete 50,304-entry post-softcap
logit vectors,

$$
D_2=\frac1N\sum_{i=1}^N\sum_{v=1}^{50304}
\left(\ell^{\rm candidate}_{iv}-\ell^{\rm native}_{iv}\right)^2.
$$

The inner operation is a vocabulary **sum**, not a mean per logit and not the
centered-logit NRMSE used by a separate distortion metric.  This is the norm in the
Markov bound underlying the certificate.

## Target-frequency reference

For token ID $t\in\{0,\ldots,50303\}$, let $n_t$ be the number of times $t$ is
the next-token target at an eligible FIT_SELECTOR position.  Publish the complete
50,304-entry integer count vector.  A later validation target is assigned with
`torch.bucketize(n_t, [1,2,4,8,16,32,64,128], right=True)`.

The nine bins therefore mean:

| Bin | FIT_SELECTOR target count |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2 | 2--3 |
| 3 | 4--7 |
| 4 | 8--15 |
| 5 | 16--31 |
| 6 | 32--63 |
| 7 | 64--127 |
| 8 | 128 or more |

Only target IDs below the 50,257-entry tokenizer support may occur in rows; the
extra model-logit entries remain zero-count references.

## Copy/repeat cells frozen for finite validation

At destination position $p$, find the nearest earlier position $j$ within 128
tokens for which the input token at $j$ equals the input token at $p$.  This source
choice uses input tokens only.  Let $k=j+1$ and compare the observed input token at
$k$ with the next-token target at $p$.

- `copy_positive`: an eligible $j$ exists and the target equals the token at $k$;
- `repeat_negative`: an eligible $j$ exists and the target differs;
- `nonrepeat`: no eligible $j$ exists.

All three masks are intersected with the published role eligibility mask and only
positions 64--255 are scored.  They partition `all_scored`.  This is the same
input-policy definition as `COPY_SOURCE_EDGE_DISCOVERY_PREREGISTRATION.md`, now
frozen before MLP2 validation.

## Publication and validation gate

The calibration publishes only aggregate margin quantiles, the epsilon
grid, the target-count vector, hashes, call/source/checkpoint ledgers, and a receipt.
Raw logits are never published.  Exactly 48 native forwards of four rows are
authorized; exactly one all-false document (source-row ordinal 82) remains in its
original batch but contributes no position.  No backward pass is authorized.  CUDA
on the RTX 5090 and bfloat16 model execution are mandatory; there is no CPU fallback.

The receipt may authorize construction of the `VALIDATION` runner only if all
31,505 eligible positions are counted, the frequency counts sum to 31,505, the
three frozen cells partition FIT_SELECTOR eligible positions in a token-only replay,
all source and parent hashes match, and the receipt is the last content write.
