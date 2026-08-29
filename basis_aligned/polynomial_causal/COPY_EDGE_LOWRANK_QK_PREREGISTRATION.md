# Low-rank compilation of the contextual L8 copy gate

Status: **exploratory; frozen before causal outcomes from this runner**.

The payload and source policy of the natural-text L8 H3/H4 copy edge are now simple,
but constant, repeat-distance, and old static-matcher gates are not faithful.  This
experiment compresses the native contextual query/key computation itself, preserving
its product-of-two-bilinear-forms structure.

Evaluation uses exposed cached documents 33--128, exactly matching the preceding
fit/evaluation split.  Unchanged native/deletion/constant baselines are reused from
`copy_edge_constant_scalar_results.json`, SHA-256
`3da06d79c0d28bbb6f4d13082aa8c0dcc1bd3315a5ef9ec485e347136774603f`.

## Native scalar

For L8 head $h\in\{3,4\}$, destination $p$, and exact successor source
$k=j(p)+1$, the native pattern scalar is

$$
a_h(p,k)=
\left(\frac{q_h(p)^\top k_h(k)}{128}\right)
\left(\frac{q'_h(p)^\top k'_h(k)}{128}\right).
$$

Each of $q,k,q',k'$ is produced by a separate $128\times1152$ weight slice,
head-wise RMS normalization, and the checkpoint's rotary transform.

## Frozen low-rank replacement

For each of the eight weight slices (four projections times two heads), compute its
ordinary float32 singular value decomposition

$$
W=U\Sigma V^\top
$$

and retain the leading $r$ terms:

$$
W_r=U_{:r}\Sigma_{:r}V_{:r}^\top.
$$

The executable projection is two linear maps, first $V_{:r}^\top x$ and then
$U_{:r}\Sigma_{:r}z$.  Head RMS normalization, rotary frequencies, the product of
dot products, the input-only source rule, and the shared $\lambda_8v_1$ payload stay
unchanged.  No activations or target labels fit the factors.

Frozen ranks are

$$
r\in\{8,16,32,64,96,128\}.
$$

The original gate stores $8\cdot128\cdot1152=1{,}179{,}648$ query/key values.  The
rank-$r$ factors store

$$
8r(1152+128)=10{,}240r
$$

values: 6.9%, 13.9%, 27.8%, 55.6%, 83.3%, and 111.1% of the original respectively.
Rank 128 is deliberately included as a numerical/executable-factorization control;
it is more expensive than storing the unfactored matrices.

A weights-only audit before preregistration found that individual slices require
67--79 singular directions for 90% Frobenius energy and 84--96 for 95%, so the curve
is not assumed to be low rank.

## Physical intervention

At every input-eligible repeat destination, subtract the exact native mixed-value
successor edge and add

$$
\widetilde a_{h,r}(p,k)P_h\bigl(\lambda_8v_{1,h}(k)\bigr).
$$

Every other source contribution remains native.  Scoring positions and cells remain
`copy_positive`, `repeat_negative`, `nonrepeat`, and `all_scored`.

## Metrics and frozen decisions

For every rank report:

- copy causal recovery relative to exact edge deletion;
- CE, native-to-arm KL, and top-1 accuracy in all four cells;
- held-out scalar $R^2$, correlation, and mean absolute error for H3/H4;
- stored factor values and multiply counts relative to the native sliced gate;
- document-mean $\Delta$CE and standard error.

Frozen gates:

- Q1, executable control: rank 128 copy recovery $\ge0.90$.
- Q2, useful half-price gate: rank 64 recovery $\ge0.80$.
- Q3, strong quarter-price gate: rank 32 recovery $\ge0.70$.
- Q4, rank-64 selectivity: repeat-negative and nonrepeat absolute $\Delta$CE each at
  most `0.02` nat and at most 25% of edge-deletion copy damage.
- Q5, rank curve coherent: recovery at 96 is not below recovery at 64 by more than
  0.05, and recovery at 64 is not below recovery at 32 by more than 0.05.

Select the smallest rank with copy recovery $\ge0.90$, all-scored $\Delta$CE
$\le0.001`, and absolute repeat-negative/nonrepeat $\Delta$CE $\le0.01`.  If no rank
passes, report none.  Passing Q2 or Q3 would justify data-aware simultaneous
factorization/HOSVD.  Failure of Q1 means the two-stage BF16 factorized evaluator is
not a valid numerical interface and no lower-rank causal conclusion may be drawn.

