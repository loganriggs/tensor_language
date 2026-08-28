# Execution addendum: MLP1 empirical-moment tensor discriminator

Date: 2026-08-28

Status: prospective and outcome-blind. This addendum closes ambiguities found in
the implementation-readiness audit of
`MLP1_EMPIRICAL_MOMENT_TENSOR_DISCRIMINATOR_PREREGISTRATION.md`. It authorizes no
row harvest, model load, activation capture, fit, or validation result. A later
source-closed execution protocol must bind this file and may narrow the experiment,
but may not silently change these rules after seeing an outcome.

## 1. Exact row counts, masks, and nesting

The eligible support of a 256-token window is the 192 positions with zero-indexed
indices `64:256`. Rows are concatenated in the frozen document order, then in
increasing position order. Therefore:

- every 400,000-row role contains 2,083 complete windows (399,936 rows) and
  positions `64:128` of its 2,084th window;
- `FIT100` is the first 100,000 FIT rows: 520 complete windows and positions
  `64:224` of FIT window 521;
- `FIT200` is the first 200,000 FIT rows: 1,041 complete windows and positions
  `64:192` of FIT window 1,042; and
- `FIT400` is the complete 400,000-row FIT role above.

No padding position contributes. The row receipt records, for every shard, the
ordered `(source_document_index, position_index, token_id)` triples and their hash.
The implementation must assert these exact counts and nested-prefix hashes.

The row freezer enumerates all document indices in the pinned local FineWeb parquet,
subtracts the recursively collected union of every committed and worktree registry
exclusion, and orders the survivors by the bytewise SHA-256 digest of
`bilin18_mlp1_empirical_moment_v1\0<decimal_document_index>`, breaking the
cryptographically negligible digest tie by numeric document index. The first 2,084
documents are FIT, the next 2,084 VALIDATION, and the next 2,084 REPLICATION. The
pre-outcome authority binds the complete exclusion census, parquet row count and
hash, tokenizer/model revision, ordered role indices, and role hashes. Any collision
or insufficient source supply fails closed.

## 2. Native trajectory and immutable cache format

All inputs and teacher writes come from the unmodified native checkpoint trajectory.
At physical site 1, the input is `EarlyMLPEvent.state`, the post-attention normalized
state consumed by MLP1, and the teacher write is exactly
`event.block.mlp(event.state)`. A compiled-program trajectory, a previously cached
global-gate trajectory, and recomputation from a substituted lower layer are invalid.

The collector uses a separately audited native prefix ending immediately after the
MLP1 write; it need not execute layers 2--17. It freezes both input and teacher write
as little-endian contiguous float32. Each role is stored in create-only shards of at
most 25,000 rows per tensor (about 115.2 MB for one 25,000 by 1,152 tensor), plus
integer metadata. A temporary shard is flushed, hashed, atomically renamed, and only
then entered into a create-only receipt. The receipt is published last and binds
shape, dtype, byte count, SHA-256, row interval, document set, source commit, native
call ledger, and absence of substitutions. FIT and VALIDATION may be captured in the
first authorized transaction; REPLICATION remains physically absent until one
candidate is frozen.

## 3. Moment and PCA conventions

All fitted moments use population normalization:

\[
 \mu=N^{-1}\sum_n x_n,\qquad
 \Sigma=N^{-1}\sum_n(x_n-\mu)(x_n-\mu)^T.
\]

Streaming sufficient statistics accumulate in float64 in a fixed shard/row order.
Covariance is symmetrized as `(Sigma + Sigma.T)/2` before a deterministic float64
symmetric eigendecomposition. Eigenpairs are ordered by decreasing eigenvalue. A
vector sign is fixed by making its largest-absolute coordinate positive, breaking a
coordinate tie by smallest index. The authority freezes the resulting basis and,
more importantly, each subspace projector; later scoring consumes the frozen bytes
rather than rerunning an eigensolver. An eigengap below
`100 * eps_float64 * max(1, lambda_max)` is reported as a degenerate boundary and
forbids semantic claims about individual PCs, but does not invalidate the frozen
projector.

`mean norm / input RMS` means

\[
 \|\mu\|_2 / \sqrt{N^{-1}\sum_n\|x_n\|_2^2}.
\]

The mean direction is present iff this ratio exceeds `1e-8`. Centered PCs are first
projected orthogonal to the normalized mean, then re-orthogonalized by deterministic
float64 QR. Every projector must be symmetric and idempotent to relative Frobenius
tolerance `1e-10` in float64. Float32 deployed projectors are separately serialized
and replay-tested.

## 4. Exact 48-probe residual bank

Every probe is an output-valued residual map `H(x) = teacher_write_without_bias -
candidate_write_without_bias`; all biases cancel. Seeds are derived by SHA-256 from
the experiment ID and the probe label and are frozen in the later authority.

1. **Eight projection residuals:** for each `k` in `{64,96,128,256}`, one
   `mean+PCA-k` and one `PCA-no-mean-k` map,
   `H(x)=F(x)-b-[F(P_k x)-b]`.
2. **Sixteen random-projection residuals:** four `mean+random-k` Haar projectors per
   `k`, with the same residual convention. Gaussian matrices are generated in
   float64 by NumPy `PCG64DXSM`, projected off the mean when present, and reduced by
   deterministic QR before their projectors are frozen.
3. **Eight signed native-gate perturbations:** generate a length-4,608 Rademacher
   vector `s_j` and set
   `H_j(x)=0.1 * D(s_j * ((Lx)*(Rx)))`. This is the residual of a candidate whose
   native gate contributions were changed by the corresponding signed ten percent.
4. **Eight native-gate dropout residuals:** select exactly 461 distinct gates
   uniformly without replacement for each seed and set
   `H_j(x)=D(m_j * ((Lx)*(Rx)))`, where `m_j` is their binary mask. Supports and
   sorted indices are frozen.
5. **Eight random paired-factor residuals:** use rank 512. Draw `A` and `B` iid
   `N(0,1/1152)` and `C` iid `N(0,1/512)` in float64 with the frozen generator.
   Let `R_j(x)=C((Ax)*(Bx))`. On FIT100 only, set
   `alpha_j=sqrt(E||F(x)-b||^2/E||R_j(x)||^2)` and
   `H_j(x)=F(x)-b-alpha_j R_j(x)`. Zero or nonfinite denominator fails closed.

All factors, masks, projectors, alphas, and their float32 deployed copies are frozen
before VALIDATION is accessible. The empirical Gram uses float64 accumulation in
fixed order. Implicit tensor/Wick code must agree with direct evaluation on a small
known-answer model before authority.

## 5. Uncertainty and Wick gate semantics

The four Wick calibration conditions in the parent preregistration apply to point
statistics. “Each error statistic” in the 100k/200k/400k stability rule means the
relative Gram Frobenius error, the maximum eligible diagonal relative error, and
`1 - Spearman`; each may increase by at most 0.01 at either doubling. FIT moments are
frozen fitted objects and are not re-estimated inside the validation bootstrap.
Their sampling sensitivity is measured by the nested FIT100/200/400 ladder. The
shared document bootstrap resamples VALIDATION documents, recomputes the empirical
Gram and all candidate empirical losses, and holds every FIT-derived object fixed.

For candidate promotion, absolute-loss upper bounds and matched-control lower bounds
use the registered simultaneous 95% document-bootstrap bands. The metric-calibration
point gates above must pass as written; simultaneous bootstrap bands for those same
statistics are reported as robustness diagnostics but do not replace their fixed
thresholds. REPLICATION repeats the same convention with no reselection.

## 6. Staged execution and present blockers

The licensed implementation order is:

1. pure CPU known-answer machinery for streaming moments, frozen projectors, direct
   and implicit probe Grams, noncentral Wick contractions, document sufficient
   statistics, Spearman, and simultaneous bootstrap;
2. outcome-blind freezer for the exact 6,252 role documents, while only FIT and
   VALIDATION activation files may initially be materialized;
3. source-closed no-outcome authority and native prefix collector;
4. FIT moment computation and a second create-only authority freezing the PCA,
   Haar, and 48-probe bundle before VALIDATION is opened; and
5. only then, the no-optimizer validation projection screen and empirical-versus-Wick
   comparison.

No existing activation cache is reusable as an observation: existing rows are only
registry exclusions. The pinned FineWeb parquet, tokenizer/model snapshot, registry
utilities, native model facade, lifecycle utilities, bootstrap implementation, and
implicit noncentral-Wick algebra may be reused only after their exact source hashes
are bound. The current GPU occupancy is a scheduling condition, not a scientific
blocker; all stage-1 work is CPU-only.

