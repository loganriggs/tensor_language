# Shared HOSVD improves the copy-gate price frontier

Status: exploratory weights-only factorization on exposed cached documents 33--128.

## Main result

A single rank-256 input basis shared across all eight L8 H3/H4 query/key projections
preserves **92.25%** of the exact copy edge's causal effect with 557,056 gate values.
That is 15% fewer gate values than the successful independent rank-64 SVD program
(655,360), while recovering slightly more behavior (92.25% versus 91.00%).

Including the fixed two-head writer, the selected local program uses 851,968 values
instead of 1,474,560 native values: a **42.2% local reduction**.

## What “shared HOSVD” computes

Stack the eight $128\times1152$ Q/K/Q2/K2 slices into a third-order tensor

$$
\mathcal W\in\mathbb R^{8\times128\times1152}.
$$

Unfolding along the 1152-dimensional input mode gives a $1024\times1152$ matrix.
Its right singular vectors form a shared input dictionary $V_R$.  At each token, the
program computes

$$
z=V_R^\top x
$$

once and reuses $z$ for all eight projections:

$$
W_i x\approx (W_iV_R)z.
$$

The stored price is

$$
1152R+8\cdot128R=2176R,
$$

instead of $8R(1152+128)=10240R$ for eight independent bases at the same nominal
rank.  The shared rank is larger, but reuse wins overall.

## Causal and scalar curves

| Shared rank | Gate values | Canonical copy recovery | Copy $\Delta$CE | All-scored $\Delta$CE |
|---:|---:|---:|---:|---:|
| 64 | 139,264 | 47.0% | +0.06992 | +0.00266 |
| 128 | 278,528 | 76.2% | +0.03137 | +0.00062 |
| 192 | 417,792 | 87.9% | +0.01590 | +0.00007 |
| **256** | **557,056** | **92.3%** | **+0.01022** | **-0.00033** |
| 320 | 696,320 | 93.9% | +0.00799 | -0.00020 |
| 512 | 1,114,112 | 94.9% | +0.00672 | -0.00025 |
| 1024 control | 2,228,224 | 95.7% | +0.00567 | -0.00029 |

At selected rank 256:

- H3/H4 scalar $R^2$: `0.9782 / 0.9621`;
- scalar correlation: `0.9902 / 0.9842`;
- mean absolute scalar error: `0.00414 / 0.00736`;
- repeat-negative $\Delta$CE: `-0.00251` nat;
- nonrepeat $\Delta$CE: `-0.00044` nat;
- all-scored native-to-arm KL: `0.00067`;
- copy top-1: `88.45%` versus native `88.86%`.

Rank 192 is also a useful cheaper operating point: 417,792 gate values, 87.9%
recovery, and nearly zero aggregate CE change.  Rank 320 exactly matches native
copy top-1 on this split, but costs slightly more than the independent rank-64 gate.
The selected rank 256 is the first point meeting the frozen 90%-recovery frontier.

## Did norm canonicalization help?

For basis selection, the canonical arm divided each slice by its Frobenius norm so
arbitrary RMSNorm-invisible scales could not determine HOSVD priorities.  Executable
cores then used the original weight scales, so full-rank behavior was unchanged.

| Rank | Raw recovery | Canonical recovery | Difference |
|---:|---:|---:|---:|
| 64 | 42.5% | 47.0% | +4.52 points |
| 128 | 74.0% | 76.2% | +2.18 points |
| 192 | 86.6% | 87.9% | +1.38 points |
| 256 | 91.3% | 92.3% | **+0.93 point** |
| 320 | 93.6% | 93.9% | +0.32 point |

Canonicalization helps consistently, especially at very tight rank.  But at the
registered rank-256 decision point it misses the required +2-point improvement gate.
The honest conclusion is:

> Norm canonicalization is valid and mildly beneficial here; shared factorization,
> not canonicalization, creates the important price improvement.

The reason is visible before fitting: slice Frobenius norms span only 82.4--97.2, so
the raw tensor was not badly gauge-imbalanced.  This does not refute canonicalization
for tensors with larger gauge disparities.

## Why this simplicity definition is now validated

The new representation has three operational benefits beyond reconstructing weights:

1. It reduces stored values and matrix-multiply width in an executable intervention.
2. It preserves downstream CE, KL, top-1, and selectivity on held-out documents.
3. It creates one shared latent state $z$ consumed sparsely by eight projection cores,
   a more composable interface than eight unrelated SVD coordinates.

This is exactly the kind of simplicity measure the project needed: sharing and rank
are rewarded only when they buy executable cost without losing causal behavior.

## Remaining boundary and next step

The gate still consumes the native 1152-dimensional contextual state entering L8.
The shared basis tells us that only a 256-dimensional projection of that state is
needed for this copy edge.  This creates a concrete downstream-defined interface for
earlier components:

$$
z_p=V_{256}^\top x^{(8)}_p.
$$

Rather than asking MLP0/MLP1/MLP2 to reconstruct an arbitrary full residual stream,
we can now ask which earlier writes are necessary to reconstruct or manipulate this
specific causal state $z$.  That is the natural composition telescope: compress
upstream components against a validated 256-dimensional consumer and measure whether
the copy program and whole-model CE survive.

Fresh-data confirmation should precede a final claim, but the exposed-split effect is
large and both full-rank controls pass.  The next discovery action is to test upstream
write-to-$z$ transport before spending effort semantically naming all 256 directions.

## Artifacts

- `COPY_EDGE_SHARED_HOSVD_PREREGISTRATION.md`
- `run_copy_edge_shared_hosvd.py`
- `copy_edge_shared_hosvd_results.json`
- `COPY_EDGE_LOWRANK_QK_FINDINGS.md`

