# Shared HOSVD basis for the contextual copy gate

Status: **exploratory; frozen before causal outcomes from this runner**.

Independent rank-64 SVD factors preserved 91.0% of the L8 H3/H4 exact copy-edge
effect using 655,360 gate values.  This experiment asks whether the eight Q/K slices
can share one input dictionary and whether norm canonicalization improves that shared
basis.

Evaluation reuses exposed cached documents 33--128.  Hash-pinned native and deletion
baselines come from `copy_edge_constant_scalar_results.json`, SHA-256
`3da06d79c0d28bbb6f4d13082aa8c0dcc1bd3315a5ef9ec485e347136774603f`.

## Tensor and shared input factorization

Stack the four Q/K/Q2/K2 slices for L8 H3/H4 into

$$
\mathcal W\in\mathbb R^{8\times128\times1152}.
$$

The mode-3 HOSVD basis is the right singular basis of the vertical unfolding
$W_{(3)}\in\mathbb R^{1024\times1152}$.  At shared input rank $R$,

$$
W_i x\approx (W_iV_R)(V_R^\top x).
$$

The latent $V_R^\top x$ is computed once at a token position and reused by all eight
projections.  The gate stores

$$
1152R+8\cdot128R=2176R
$$

values.  Frozen ranks are

$$
R\in\{64,128,192,256,320,384,512,1024\}.
$$

Rank 1024 spans the complete joint row space and is the numerical control.

## Raw versus norm-canonical basis

Because every projected vector is immediately head-RMS-normalized, positive scalar
rescaling $W_i\mapsto c_iW_i$ is functionally irrelevant apart from numerical epsilon.
Raw HOSVD is nevertheless sensitive to these gauge scales.  Two frozen bases are
compared:

1. `raw`: SVD of the original vertical stack;
2. `canonical`: first divide each slice by its Frobenius norm when selecting the
   shared basis, then form executable cores from the original unscaled slices
   $W_iV_R$.  Thus canonicalization changes basis priority but not the full-rank
   target function.

A weights-only audit before preregistration found slice norms between 82.43 and 97.18,
so a large canonicalization advantage is not presumed.  Raw/canonical shared bases
need ranks 226/228 for 80% Frobenius energy and 346/347 for 90%.

## Physical replacement and metrics

For each basis/rank, compute the original head RMS normalization, rotary transform,
product of two dot products, and shared $\lambda_8v_1$ successor payload.  Subtract
only the native exact mixed edge and add this approximation at all input-eligible
destinations.

Report copy recovery relative to edge deletion, all-cell CE/KL/top-1 with document
SEs, H3/H4 scalar $R^2$/correlation/MAE, exact factor price, and raw-versus-canonical
paired rank differences.

Frozen gates:

- H1: raw and canonical rank-1024 controls each recover at least 90%.
- H2: canonical rank 256 recovers at least 80%.
- H3: canonical rank 256 recovers at least 90%.
- H4: canonical rank-256 recovery exceeds raw rank-256 by at least 0.02.
- H5: canonical rank 256 is selective: repeat-negative/nonrepeat absolute $\Delta$CE
  each at most `0.02` nat and 25% of deletion copy damage.
- H6: a shared basis strictly improves the established price frontier: some rank
  with at most 655,360 factor values has recovery at least 90%, all-scored
  $\Delta$CE at most `0.001`, and repeat-negative/nonrepeat absolute $\Delta$CE at
  most `0.01`.

Select the lowest-price passing raw or canonical rank, breaking exact price ties by
higher recovery.  Failure of H1 invalidates lower-rank causal conclusions.  Passing
H6 replaces independent rank 64 as the local compiler.  Failure of H4 means the
norm-canonicalization idea is mathematically clean but practically redundant at this
site; it must not be advertised as an improvement.

