# Decision addendum: MLP1 implicit folded tensor

Date: 2026-08-28

Status: prospective interpretation and pricing only. This addendum authorizes no
checkpoint load, tensor collection, fitting, row use, result, or promotion. It adds
no decision to the frozen v1 diagnostic. Its purpose is to prevent a future HOSVD
spectrum from being promoted beyond what it can establish.

## Established lower bound and what it does not say

The existing `mlp_product_rank_audit_results.json` gives MLP1 numerical rank
`1152/1152` in each of two 1,280-column Gaussian evaluation sketches at registered
relative tolerances `1e-4`, `1e-5`, and `1e-6`. In the frozen product grammar,

\[
 F(x)-b=\sum_{a=1}^q c_a(u_a^Tx)(v_a^Tx),
\]

the output-by-evaluation matrix has rank at most $q$. The audit is therefore a
randomized numerical lower-bound certificate $q\ge1152$ at those tolerances. The
same sampled-output argument implies that an exact/tolerance-matched Down
factorization needs output rank $r_D\ge1152$.

This is not a construction at $q=1152$, a symbolic-rank proof, or a lower bound for
approximation on natural activations, Fisher response, suffix KL, or CE. In
particular, full numerical rank is compatible with a steep energy spectrum. The
implicit folded-tensor diagnostic can measure that spectrum without contradicting
the audit.

## Exact structural crossovers for bilin18 MLP1

Here $d=o=1152$, $h=4608$, bias is included, and prices use the v1 structural
currency. The native map stores

\[
 P_N=15{,}926{,}400\ \text{floats}
\]

and executes 4,608 products and 15,925,248 multiply-adds.

| program | float storage | products | exact crossover against native |
|---|---:|---:|---|
| Down rank $r_D$ | $10{,}617{,}984+5{,}760r_D$ | 4,608 | fewer floats iff $r_D\le921$; the audit forces $r_D=1152$ for a tolerance-matched exact map, whose price is 17,253,504 |
| CP/product rank $q$ | $3{,}456q+1{,}152$ | $q$ | fewer floats and products iff $q\le4607$; the existing exact numerical lower bound leaves only $1152\le q\le4607$ |
| symmetric dense Tucker $(r_o,r_i)$ | $1152(r_i+r_o+1)+r_o p$, $p=r_i(r_i+1)/2$ | $p$ | fewer floats iff $r_o\le\left\lfloor[15{,}926{,}399-1152(r_i+1)]/(1152+p)\right\rfloor$; fewer products iff $r_i\le95$ |

At the lower-bound endpoint, `CP(1152)` costs 3,982,464 floats. At the last strict
native win, `CP(4607)` costs 15,922,944 floats. Thus the rank audit permits at most a
fourfold exact gate reduction, but it neither attains nor predicts it.

Because the output certificate is full, a tolerance-matched exact Tucker candidate
must use $r_o=1152$. Its price reduces to

\[
 P_T(1152,r_i)=1{,}328{,}256+576r_i(r_i+3).
\]

The known product lower bound also forces $p\ge1152$, hence $r_i\ge48$. Such a
dense Tucker candidate uses fewer floats than native iff $r_i\le157$, and fewer
products iff $r_i\le95$. Consequently the only presently possible
tolerance-matched, certificate-compatible exact
coordinatewise-better window is

\[
 r_o=1152,\qquad 48\le r_i\le95,
\]

subject to an actual reconstruction certificate. The endpoints are:

- $r_i=48$: 2,738,304 floats and 1,176 products;
- $r_i=95$: 6,690,816 floats and 4,560 products;
- $r_i=157$: 15,797,376 floats and 12,403 products, a storage-only win;
- $r_i=158$: 15,980,544 floats, so even the storage win ends.

Against CP with the same product count $q=p$ and $r_o=1152$, dense Tucker stores
fewer floats exactly when $r_i\ge34$. This comparison does not dominate a CP core
whose fitted rank is strictly below $p$.

The currently registered equal-mode projected cores are necessarily approximate
screens in light of the full-output certificate. Their complete dense prices are:

| $(r_o,r_i)$ | floats | products |
|---:|---:|---:|
| $(16,16)$ | 40,192 | 136 |
| $(32,32)$ | 91,776 | 528 |
| $(64,64)$ | 281,728 | 2,080 |

For a COO core, the exact float-only crossover is

\[
 s\le15{,}926{,}399-1152(r_i+r_o+1),
\]

and a strict product win requires at most 4,607 distinct active input pairs. It also
stores $3s$ indices. Because the current ledger deliberately reports floats and
integers separately, no total-byte or coordinatewise storage win over the
zero-index native program is licensed until float precision, index dtype, and a
scalar byte rule are frozen. The number of nonzero core coefficients $s$ is not
the number of products; the latter is the number of distinct active input pairs.

## Outcome-to-action boundary

A HOSVD result licenses a **Tucker executable assay**, not a simplicity claim, only
when all of the following are true at a prospectively chosen registered threshold:

1. the actual projected-core energy, not merely the separate mode-energy summaries,
   meets the coefficient-Frobenius target;
2. the retained $(r_o,r_i)$ and top-COO support are numerically stable under the
   registered replay and under rotations within unresolved eigenspaces;
3. the complete dense or COO price is Pareto-better than the relevant native and
   balanced-Down baseline in the frozen cost currency; and
4. a later, separately frozen candidate beats matched random/core and Down controls
   first on native-input write error and then on causal suffix response.

The standard shared-subspace HOSVD residual bound can screen a rank pair, but the
projected core must still be evaluated directly: separate output and input tail
fractions can add, and neither one alone is the tensor reconstruction error.

The strategic branches are:

- If a stable retained input rank has $r_i\le95$ and the direct core residual
  passes, test the factor-complete Tucker program first. It already has a native
  product-count runway; CP is secondary unless factorizing the core can improve that
  frontier.
- If $96\le r_i\le157$ with $r_o=1152$, dense Tucker can be a storage
  compression but not a product-gate compression. This is the region where a bounded
  CP or block-term search may be useful, because its gate count scales linearly in
  $q$, not quadratically in $r_i$.
- If the registered approximate input rank exceeds 157 with essentially full output
  rank, prune dense Tucker as a native-storage simplification at that error target.
  A smaller approximate $r_o$ must be judged by the general integer crossover
  above, not by this full-output shortcut.
- If the registered 16/32/64 projected cores retain too little energy or their
  sparse curves have no knee before their complete-price cutoff, prune this
  **low-dimensional Tucker/core-first path**. Do not call all CP ranks pruned.

HOSVD can license only a bounded CP search: it may provide a smaller stable core on
which to fit and a complete price window. It cannot certify low CP rank. Conversely,
unfavorable HOSVD spectra do not prove high CP rank; tensors with full multilinear
rank can still have CP rank between the certified lower bound 1,152 and native upper
bound 4,608. The entire CP simplification is pruned only by a valid product-rank
lower bound reaching 4,608 at the target tolerance. A prospectively bounded
executable CP search that fails its reconstruction, conditioning, matched-control,
and causal gates prunes only its frozen ranks, optimizer budget, and metric—not
unsearched CP.

## Inferences that remain forbidden

- `rank 1152/1152` does not mean “incompressible,” and it does not mean a rank-1152
  CP decomposition exists.
- A low HOSVD energy rank is a mode-subspace statement, not a gate count, semantic
  feature count, or CP-rank estimate.
- A low balanced-Down rank retains all 4,608 products and is not a tensor-rank result.
- Top-COO coefficient sparsity is basis-dependent, may rotate inside degenerate
  HOSVD subspaces, and is not CP rank.
- Scalar gate balancing conditions the native factors; it does not make the balanced
  Down spectrum a full-gauge invariant. The folded tensor spectrum is the physical
  object.
- Coefficient-Frobenius retention does not imply natural-state, suffix, CE, OOD,
  editing, removal, or MLP0/MLP2 compositional fidelity.
- Structural float counts are neither serialized bytes nor MDL. Gauge dimension is
  not subtracted unless a real canonical codec removes it, and index/precision costs
  cannot be silently omitted.

The v1 diagnostic is therefore high-value as a cheap branch selector: it can expose
a priced Tucker window or terminate that window. It is deliberately weak evidence
for CP and no evidence for a causal replacement until an executable zero-native-call
program passes the common intervention and composition contract.
