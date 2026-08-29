# Hierarchical shared-trunk plus site-private residual RRR v1

**Status:** prospective CPU mathematical contract only. No model/checkpoint tensor,
row tensor, GPU, authority, result, failure, or receipt may be opened from this file.
The factor tensors returned by the pure core are test objects; this protocol does not
authorize serializing a deployable program.

## Motivation and pinned parent

The completed real shared-output RRR discovery found that a global rank-64/128 basis
beats the strongest independently allocated ranks at equal storage, but every global
basis loses to independent maps at equal per-site rank. At rank 512 the global basis
also loses at equal storage. This is evidence for reused output directions plus
important site-private directions, not for one universal output language.

Any future numerical implementation must pin these immutable parents:

- `shared_output_rrr_real_v2_recovery_authority.json`, SHA256
  `32106d80f43bd73853ca25841b81ea1297d0e9c5d2bf6f404510a1c1b217c7db`;
- `shared_output_rrr_real_v2_recovery_results.json`, SHA256
  `19d65e2c6d4a0cff19ddfb76ddbe62dcd26c462a695e006c457da85a89adc053`;
- `shared_output_rrr_real_v2_recovery_receipt.json`, SHA256
  `57f699d680a7ea010f6ec8b12c3c33d61f1b3f540ad2891517fe751074dbdd56`.

The parent is discovery-only and repeatedly exposed. This successor is likewise a
discovery proposal; it cannot increase any global explained fraction or support a
generalization claim.

## Exact constrained problem

For site (j), retain the parent's float64 fit-only sufficient-statistic merit

\[
M_j=C_j^\top(G+\lambda I)^{-1}C_j\succeq0.
\]

For a prospectively fixed global rank-(q_0) projector (P=V_0V_0^\top), where
(V_0) is the top eigenspace of \(\sum_jM_j\), set (Q=I-P). The hierarchical map is

\[
\widehat Y_j=XA^{(0)}_jV_0^\top+XA^{(p)}_jU_j^\top,
\qquad V_0^\top U_j=0,\quad U_j^\top U_j=I.
\]

The shared projector is fixed before residual fitting; this is not an alternating or
jointly reoptimized decomposition. Conditional on (P), the exact rank-(r_j)
private projector consists of the top eigenvectors of (QM_jQ) restricted to
`range(Q)`. The input maps are

\[
A^{(0)}_j=(G+\lambda I)^{-1}C_jV_0,
\qquad A^{(p)}_j=(G+\lambda I)^{-1}C_jU_j.
\]

Thus explained penalized fit merit is exactly

\[
\sum_j\operatorname{tr}(V_0^\top M_jV_0)
+\sum_j\sum_{k<r_j}\mu_{jk},
\]

where \(\mu_{jk}\) are descending residual eigenvalues.

## Exact storage and allocation

For 36 sites and dimension 1152, literal map storage is

\[
S=37\cdot1152\,q_0+2\cdot1152\sum_jr_j.
\]

The first term stores 36 shared input maps and one shared output dictionary. Each
private rank slot stores one input and one output vector. For a fixed total budget,
the number of private slots is

\[
R=\frac{S-37\cdot1152\,q_0}{2\cdot1152}.
\]

It must be a nonnegative integer. Allocate the (R) slots by the globally largest
fit-only residual eigenvalues, with site then eigen-index tie breaking. Selected
directions at every site must be a spectral prefix. No CE or evaluation row can alter
the ranks.

The endpoint identities are registered controls:

- (q_0=0): (Q=I), so the construction is exactly the strongest exact-price
  independent allocation.
- (R=0): the construction is exactly the fixed global RRR map.

## Frozen discovery storage grid

The future numerical runner, if separately preregistered and source-closed, must fit
the following cells before any evaluation role is deserialized:

| total map budget | floats | shared ranks (q_0) | private slots at (q_0=0/64/128/256) |
|---|---:|---|---|
| global rank 512 | 21,823,488 | 0, 64, 128, 256, and zero-private 512 | 9472 / 8288 / 7104 / 4736 |
| typed rank 512 | 22,413,312 | 0, 64, 128, 256 | 9728 / 8544 / 7360 / 4992 |
| independent rank 512 | 42,467,328 | 0, 64, 128, 256 | 18432 / 17248 / 16064 / 13696 |

The common exact 5,419-token tables cost 224,736,768 floats in every arm and must be
reported separately and in the full-program price. The three full-program totals are
therefore respectively 246,560,256, 247,150,080, and 267,204,096 floats. Dense
multiplies are
`2*1152*(q0+r_j)` at site (j); equal storage does not imply equal compute.

Frozen comparators are:

- global-budget cells: parent `global_q512` and `price_global_q512` endpoints;
- typed-budget cells: parent `typed_q512` and `price_typed_q512`;
- independent-budget cells: parent uniform `independent_q512` and the newly realized
  (q_0=0) exact-price endpoint;
- all cells: native reference and exact covered-table identity.

No parent CE value selects (q_0), a private rank, or a budget.

## Frozen whole-model predictions

All CE differences are all-position nats per scored token and must hold separately on
`skip7000`, `skip11000`, and `skip1200`.

1. **Primary global-price prediction.** At least one interior
   (q_0\in\{64,128,256\}) at 21,823,488 floats beats both endpoint programs
   (`global_q512` and the (q_0=0) exact-price independent allocation) by at least
   0.01 nat on every role.
2. **Typed-price prediction.** At least one interior cell at 22,413,312 floats beats
   both `typed_q512` and its (q_0=0) exact-price endpoint by at least 0.01 nat on
   every role.
3. **Large-budget diagnostic.** At 42,467,328 floats, report whether any nonzero
   shared trunk beats the (q_0=0) exact-price independent endpoint by at least
   0.005 nat on every role. Failure says sharing is only useful under compression.
4. Covered CE must be bit-identical across arms within each role to `1e-6`; all
   literal prices, private allocation prefixes, factor hashes, call counts, and
   parent anchors must replay exactly.

Local fit merit is diagnostic. The (q_0=0) endpoint is optimal only within the
independent-site grammar at its exact price. An interior cell is the conditional
optimum for its prospectively fixed (P,q_0), but neither endpoint nor interior is a
global optimum over all hierarchical decompositions. Whole-model CE remains the
primary causal currency.

## Identifiability and claim boundary

Only the shared projector, each combined site projector, and deployed coefficient map
are identified. Columns are arbitrary up to sign and rotations within tied
eigenspaces. A numerical runner must report shared and per-site boundary eigengaps and
hash the float32 deployed projectors/coefficient maps, not raw basis columns. At a tied
allocation cutoff, the deterministic tie rule makes one implementation reproducible
but does not make its gauge or chosen site scientifically unique.

Fit matrices and eigensolvers use CPU float64. Deployment has exactly one cast and
evaluation order: independently cast every input map and basis with contiguous
`.float()`; compute the shared factor product first and private factor product second;
add the private term exactly once. Literal prices and hashes are computed from these
deployed float32 objects. Dense coefficient matrices are diagnostic hashes only and
must not be serialized or substituted for the priced factor execution.

More importantly, “shared” and “private” are conditional on the frozen global
projector. This core does not prove that the split is the globally optimal hierarchical
factorization, and a direction excluded from (P) may recur in several private bases.
Claims are limited to literal compression and discovery-role whole-model CE. Factors
may be hashed in a future result but are not licensed for serialization, extraction,
semantic naming, validation/final scoring, or intervention by this protocol.
