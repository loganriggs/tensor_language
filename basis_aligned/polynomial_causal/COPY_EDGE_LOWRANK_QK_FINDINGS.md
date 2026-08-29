# A causal rank curve for the contextual copy gate

Status: exploratory weights-only factorization, evaluated on exposed cached documents
33--128.  No activations or target labels were used to construct the factors.

## Result in one sentence

Factoring each L8 H3/H4 query/key projection to rank 64 preserves **91.0%** of the
exact copy edge's causal CE effect, with negligible collateral and 95% scalar
variance explained; rank 32 preserves only 57.6%.  The smallest preregistered
faithful point is rank 64.

## What was compressed?

The exact source and token payload were already localized.  The remaining contextual
gate for head $h$ is

$$
a_h(p,k)=
\left(\frac{q_h(p)^\top k_h(k)}{128}\right)
\left(\frac{q'_h(p)^\top k'_h(k)}{128}\right),
$$

where $p$ is the current position and $k$ is the source immediately after the nearest
earlier equal token.  H3 and H4 each have four $128\times1152$ projection slices:
$W_Q,W_K,W_{Q'},W_{K'}$.  There are eight slices total.

For each slice, ordinary singular value decomposition gives

$$
W=U\Sigma V^\top.
$$

At rank $r$, the program stores only $U_r\Sigma_r$ and $V_r^\top$ and computes the
projection as two smaller matrix multiplications.  It then performs the original
head RMS normalization, rotary transform, two dot products, and multiplication.  The
replacement writes the already validated shared $\lambda_8v_1$ payload.

This is a genuine executable compression.  It is not merely measuring whether a
probe can predict an activation.

## Price

The native H3/H4 gate slices contain

$$
8\times128\times1152=1{,}179{,}648
$$

stored values.  Independent rank-$r$ factors contain

$$
8r(128+1152)=10{,}240r.
$$

| Rank | Gate values | Fraction of native gate |
|---:|---:|---:|
| 8 | 81,920 | 6.9% |
| 16 | 163,840 | 13.9% |
| 32 | 327,680 | 27.8% |
| 64 | 655,360 | **55.6%** |
| 96 | 983,040 | 83.3% |
| 128 | 1,310,720 | 111.1% |

Rank 128 is intentionally more expensive because factor storage has overhead; it is
the numerical control.  The fixed two-head writer adds 294,912 projection values to
every arm.  Counting gate plus writer, rank 64 uses 950,272 values versus 1,474,560,
a 35.6% reduction for this localized program.  The shared value bus and the rest of
the model are not charged as newly duplicated storage.

## Causal curve

Recovery is relative to deleting the exact successor edge.  The value-side ceiling
uses the full native scalar but the shared payload and recovers 95.9%.

| Rank | Copy recovery | Copy $\Delta$CE | Copy top-1 | All-scored $\Delta$CE |
|---:|---:|---:|---:|---:|
| 8 | 8.8% | +0.12025 | 85.46% | +0.00600 |
| 16 | 32.6% | +0.08895 | 86.07% | +0.00359 |
| 32 | 57.6% | +0.05597 | 87.09% | +0.00168 |
| **64** | **91.0%** | **+0.01187** | **88.52%** | **-0.00020** |
| 96 | 94.8% | +0.00691 | 88.65% | -0.00008 |
| 128 control | 95.9% | +0.00547 | 88.65% | -0.00021 |
| native scalar + shared payload | 95.9% | +0.00537 | 88.79% | -0.00003 |

Native copy top-1 is 88.86%.  Thus rank 64 loses only 0.34 percentage point on the
copy cell while slightly improving aggregate CE.  Its repeat-negative $\Delta$CE is
`-0.00269` nat, nonrepeat is `-0.00034`, and all-scored native-to-arm KL is `0.00072`.
The document-mean copy $\Delta$CE is `0.01080 +/- 0.00176` nat standard error.

The rank-128 control correlates `0.99999` with both native scalars and passes the
registered executable-interface gate.  Lower-rank conclusions are therefore not an
artifact of a broken two-stage evaluator.

## Scalar reconstruction and causal behavior agree

| Rank | H3 held-out $R^2$ | H4 held-out $R^2$ | H3/H4 correlation |
|---:|---:|---:|---:|
| 8 | -0.108 | -0.083 | 0.520 / 0.475 |
| 16 | 0.136 | 0.214 | 0.754 / 0.743 |
| 32 | 0.529 | 0.578 | 0.890 / 0.861 |
| **64** | **0.954** | **0.947** | **0.986 / 0.973** |
| 96 | 0.996 | 0.995 | 0.998 / 0.998 |

Here local scalar accuracy happens to predict downstream causal recovery well.  That
was not assumed: the rank was selected by CE gates, not by $R^2$.  This agreement
makes projection rank a validated simplicity measure for this particular interface.
It gives us something operationally useful that plain matrix reconstruction alone
would not: a rank-64 executable gate that preserves the circuit's behavior.

## Mathematical interpretation and the next factorization

The knee between rank 32 and 64 means the contextual copy decision is moderately,
not extremely, low dimensional.  Independent SVD is also an intentionally weak
factorization: it gives every one of the eight projections its own input basis.

The next principled move is a simultaneous/HOSVD factorization of the tensor

$$
\mathcal W\in\mathbb R^{8\times128\times1152}
$$

formed by stacking all eight projection slices.  A shared input basis $V_R$ yields

$$
W_i\approx (W_iV_R)V_R^\top,
$$

so $V_R^\top x$ is computed once and reused by all projections.  Its storage is
$1152R+8\cdot128R=2176R$, far below independent factor storage at equal rank.

There is also an exact scalar gauge relevant to Zach Furman's norm-canonicalization
suggestion: head RMS normalization makes $W_i$ and $c_iW_i$ functionally equivalent
for positive scalar $c_i$ up to the tiny RMS epsilon.  Raw HOSVD can therefore be
misled by arbitrary slice norms.  Dividing every slice by its Frobenius norm before
finding the shared basis is a principled gauge canonicalization.  The next experiment
should compare raw versus norm-canonical shared bases on the same causal curve.

## Boundaries

- This is not fresh confirmation.
- The compressed gate still consumes the native contextual residual state entering
  L8.  It does not yet explain how MLP0/MLP1/MLP2 and earlier attention construct that
  state.
- It compresses one highly causal program, not a large fraction of total model
  parameters.  Its value is the precise editable mechanism and validated rank metric.
- Semantic interpretation of the 64 modes remains open; shared HOSVD may make those
  modes more coherent and substantially cheaper.

## Artifacts

- `COPY_EDGE_LOWRANK_QK_PREREGISTRATION.md`
- `run_copy_edge_lowrank_qk.py`
- `copy_edge_lowrank_qk_results.json`

