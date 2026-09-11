# First joint sparse path implementation

11 September2026. Implements the first comparison in the [interaction-path proposal](explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md). Native status belongs to the managed pilot receipt or current runner, not this method description. No circuit has been identified by this implementation alone.

The new object is the **joint residual/residual, residual/attention and attention/attention tensor**, with the whole unembedding retained in the objective. Earlier sparse-core work used a fixed spectral basis for the last MLP alone. This implementation optimizes path feature bases and selects interactions, while comparing shared versus independent path bases at equal cost.

## Exact output elimination

Let the formal two-port input be $\xi=(r,z')$ with $z'=s z$, where $s=\|O\|_F/\sqrt{1152}$. Its write is $r+(O/s)z'$. Thus the native readers are concatenated as

$$
\mathcal L=[L,LO/s],\qquad
\mathcal R=[R,RO/s].
$$

The scalar gain convention prevents the known attention-output coordinate amplification from silently dominating the comparison. It does not whiten natural inputs or prove that the two source ports are independent. Both arms use the identical convention and retain its exact adapter.

Choose orthonormal feature readers within each source. Embed them in the combined coordinate space to obtain unit columns $b_i$. The normalized symmetric coefficient features are

$$
H_{ii}=b_i b_i^\top,
\qquad
H_{ij}=\frac{b_i b_j^\top+b_j b_i^\top}{\sqrt2}\quad(i<j).
$$

Their scalar computations are $(b_i^\top\xi)^2$ and $\sqrt2(b_i^\top\xi)(b_j^\top\xi)$. These matrices form an orthonormal family in coefficient Frobenius inner product. This is coefficient orthogonality, not distributional or causal independence.

Let $J^\top J=U^\top U$ and $Z=JD$. It is sufficient to work with the1152-dimensional output coordinates transformed by $J$ rather than instantiate50304 vocabulary slices. For an edge $e=(i,j)$, its optimal output coefficient vector is

$$
c_{ij}=Z\,\frac{
(\mathcal Lb_i)\odot(\mathcal Rb_j)
+(\mathcal Lb_j)\odot(\mathcal Rb_i)
}{\sqrt2}\quad(i<j),
$$

with $c_{ii}=Z[(\mathcal Lb_i)\odot(\mathcal Rb_i)]$. An executable physical writer is $J^{-1}c_e$. All output directions contribute to the fitting loss; no suffix labels or corpus examples enter it.

For fixed readers and selected edges $S$, the exact minimized squared coefficient error is

$$
\|T\|_F^2-\sum_{e\in S}\|c_e\|^2.
$$

Selecting the largest96 edge energies is therefore the exact fixed-basis optimum. This is **edge sparsity with dense output vectors**, not entry sparsity in a learned output-mode Tucker core. Later output-factor/DAG comparisons remain open.

For fixed edge support, standard manifold conjugate gradients adjusts the feature bases while preserving orthonormality. The output vectors are eliminated exactly at every evaluation. The outer loop reselects edges. A fixed-support stationary point is not automatically stationary against support changes, so the pilot records both.

## Matched comparison

The joint arm has16features per source, shared across every incident path block. The independent arm has8features for each side of each relevant block: one residual basis for the residual self block, separate residual/attention bases for the mixed block, and one attention basis for its self block. Coefficient features belonging to different independent blocks remain orthogonal because their source-coordinate blocks are disjoint; within a block, orthonormal readers ensure the same property.

Both arms use32dense reads,36864reader floats,96product edges and110592writer floats:147456fitted floating-point parameters. Edge indices, constants, gain adapter and native dependencies are reported separately. Independent features are initialized from weight sketches; the joint source spans contain the union of those initial block-specific spans. This does not ensure equal initial sparse fit, so initial losses are reported.

Joint feature reuse is counted when a feature supplies both a same-source and a mixed-source edge. That establishes computational incidence in a proposed program, not semantic reuse. All input normalizations, attention routing/value generation, direct residual writing and the final tail remain explicit native dependencies.

## Completed controls and remaining claims

[CPU controls](COUPLED_SPARSE_PATH_V1_CONTROL.json) pass exact dense projection, coefficient residual, feature orthogonality, exhaustive support selection and tangent finite-difference checks; the largest relative check is7.07e-10. Known-basis planted recovery error is2.22e-16. Two small independent-start planted fits recover the function to numerical precision, but one stops with tangent norm2.92e-8 above its1e-8bar. Its convergence miss is preserved; two tiny recoveries do not prove reliable native optimization.

[Native preregistration](COUPLED_SPARSE_PATH_PILOT_V1_PREREGISTRATION.md) specifies four arms, support/convergence conditions, gain and reuse predictions, time limits and the continuation rule. The first native pilot does not exhaust optimization. A negative result with unresolved stationarity cannot reject the structure hypothesis. Conversely, a successful coefficient comparison would still need frozen-program prediction, extraction, removal and composition tests.

[Kernel](coupled_sparse_path_v1.py) reuses the earlier orthogonal-core computation and standard manifold solver. [CPU check](check_coupled_sparse_path_v1.py). No new audit or publishing framework was introduced.

Execution version note: V1 passed model-free dry-run but the enqueue gate did not recognize keyword-style prediction keys. The immutable [V2 runner](../bilinear_quotient/ops/run_coupled_sparse_path_pilot_v2.py) uses explicit keys with unchanged scientific predictions; native results will use `COUPLED_SPARSE_PATH_PILOT_V2_RESULT.json`. No native V1 result exists.

## Executable edge and node interventions

[The executable](sparse_path_program_v1.py) maps either arm's selected-edge indices to scalar products and full physical writes, preserving the source gain and supplied native input RMS squared. It returns both the numerator and normalized write. Bias, direct residual and final tail are separate native operations.

An edge removal subtracts one product's write. A shared-feature node removal removes every incident edge. Removing two nodes must count their common edges once. If $w_A,w_B$ denote sums of incident writes and $w_{A\cap B}$ their overlap, then

$$
\Delta_{A\cup B}=-w_A-w_B+w_{A\cap B}.
$$

This is a computation-level intervention: deleting an internal candidate node does not alter the native input normalization denominator. A source intervention instead changes the source values and requires the caller to recompute that denominator. The interfaces should not be conflated when evaluating selective removal.

[CPU execution controls](SPARSE_PATH_PROGRAM_V1_CONTROL.json) pass dense output replay, nonunit gain absorption, edge removal, node/incident-edge equivalence and two-node inclusion-exclusion for both arm conventions. Maximum error2.01e-15, with deliberately nonzero shared-edge effects. These synthetic identities prepare subsequent native validation; they are not themselves evidence of circuit extraction or behavioral composition. [Check source](check_sparse_path_program_v1.py).
