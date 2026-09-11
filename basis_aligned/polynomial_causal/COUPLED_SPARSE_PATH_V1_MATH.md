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

## First native pilot: small advantage, unresolved support optimization

[PilotV2](COUPLED_SPARSE_PATH_PILOT_V2_RESULT.json) completed22:25:24. Instrument and reuse-incidence predictions hold; convergence and the10%advantage predictions miss. Joint capture2.9343/2.7719% versus independent2.7262/2.7461%, with equal147456fitted-float budgets. Only the second independent arm is fully converged; the other three still have support changes. Joint programs have30/33mixededges and5/13features reused within/across sources. This describes proposed arithmetic incidence, not semantic or causal reuse.

The [active-edge kernel](coupled_sparse_path_v2.py) avoids forming inactive coefficient columns during a fixed-support solve while keeping the original full-column support selection. [CPU objective and all-factor gradient replay](COUPLED_SPARSE_PATH_V2_CONTROL.json) holds within2.59e-16. [Longer continuation](COUPLED_SPARSE_PATH_CONTINUE_V1_PREREGISTRATION.md) starts from these exact final artifacts, retains ranks/costs/criteria, and tests native speed and replay before continuing. Its state is in the managed runner/receipt. Original misses remain; no absent-structure inference is justified yet.

## Converged fits and reproducible coefficient structure

[Continuation](COUPLED_SPARSE_PATH_CONTINUE_V1_RESULT.json) completed22:31:54 with all four arms meeting the original support and gradient conditions. Instrument/convergence/speed predicates hold; the10%joint advantage still misses. Final joint capture2.97992/2.78099% versus independent2.72962/2.74606%. The active-column gradient is2.15times faster for both joint cases. These are local constrained solutions, not globally optimal sparse programs.

[Full coefficient stability and token atlas](SPARSE_PATH_STABILITY_ATLAS_V1.json) passes its registered screen: joint totalfunctioncosine0.94429 and21one-to-one edges above0.9, compared with independent0.87081 and3edges. This comparison multiplies both the source-feature and full-U writer inner products; reader sign gauges compensated in writers change nothing. It is stronger than writer-only similarity, but is still coefficient-space identification evidence.

Common vocabulary writing contributes55.83/58.95%of the joint programs' coefficient energy. [Centering red-team](SPARSE_PATH_CENTERED_STABILITY_V1.json) retains a joint function cosine0.90789 and21original-pair matches above0.9; independent falls to0.77839 and2matches. The full and centered counts of21 are **not the same sets**:18pairs pass both. Earlier commentary saying the same21survived was too strong. [Frozen bank receipt](STABLE_PATH_BANK_V1.json) and [artifact](STABLE_PATH_BANK_V1.pt) preserve the18-pair intersection, unchanged, before native-state validation. Each replica retains57600fitted floats,18products and the original source/normalization dependencies.

The quotient test does not assume common pre-tanh writing is behaviorally irrelevant. It tests whether a common coefficient direction alone explains the stability. The centered screen passing does not establish selective effects.

## Prior-art aliases are recovered components, not new circuits

The atlas's quotation-related attention/attention square matches the already documented square150. [All256-square path alias audit](SPARSE_PATH_PRIOR_SQUARE_ALIAS_V1.json) folds each old reader through the residual and attention-output ports, including rr/ra/aa terms, and checks old fitted and exact native-projection writers. The quote-AA function cosine is0.99722/0.99674 in the two new starts. There are3/1path-piece aliases above0.95across the old bank in total, for either writer reference. Coefficient-shape agreement need not imply equal amplitudes or a complete old-square match.

The known square150 removal/gating results in the dossier remain authoritative, including their failures. These new path pieces are not additional quotation circuits. Other unmatched edges are unassigned computations, not novel circuits merely because this alias bank did not explain them.

## First native-state replication fails

[Developmental native-cache check](STABLE_PATH_NATIVE_CACHE_V1.json) uses the existing64pair/128endpoint morphology and neighboring-inflection cache, with both18edge banks frozen beforehand. It performs no parameter fitting. Instrument passes; physical-write, swap-effect and removal-CE replica criteria fail. This is an already inspected cache for earlier programs, not fresh corpus/OOD evidence.

Physical-write symmetric relativeRMS disagreement is28.19%, above25%. Swap-margin disagreements are40.35/54.80/22.29/44.02%for agreement verbs/count nouns/past/progressive. Signs agree87.5/87.5/100/100%. Bank-removal CE mean absolute disagreements are0.19493/0.25495/0.05739/0.12616nats, all above0.02. These are replica-disagreement statistics, not claims that every change is harmful or intended-task selective. Native sources, remaining model and final normalization/tanh remain present.

[Exact midpoint accounting](STABLE_PATH_REPLICA_ACCOUNTING_V1.json) separates the physical-write difference after weight-metric sign alignment. With feature amplitudes $a_0,a_1$ already divided by the common native input RMS squared,

$$
a_0W_0^\top-a_1W_1^\top
=(a_0-a_1)\left(\frac{W_0+W_1}{2}\right)^\top
+\left(\frac{a_0+a_1}{2}\right)(W_0-W_1)^\top.
$$

The first, reader/amplitude term has norm0.370times the total discrepancy; the writer term0.757times. They have cosine0.518, so these ratios are not additive variance shares. Identity error7.75e-16. The registered reader-dominance hypothesis fails: output-writing differences are larger in this accounting. This does not by itself assign the CE difference to one term or repair the native replication failure.

The current result is therefore **converged sparse path fits with reproducible coefficient structure, but insufficient native behavioral replication for circuit promotion**. A0.9coefficient cosine is a correspondence screen, not a25%error guarantee: even equal-norm tensors atcosine0.9differ by44.7%relativeRMS. Next comparisons should test output grouping/sharing and actual deeper composition, without fitting these validation outcomes or silently tightening a threshold until this cache passes.

## Output grouping follow-up

The [22:56 mathematical review and executed experiments](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-11_2256.md) applies exact output/function SVD to the frozen96edge programs. Leading-eight centered output subspaces align closely, but complete coefficient groups miss stability. Native group writes and removal effects replicate; past/progressive swaps still fail. The full96control already passes the same level/removal checks, so no grouping-specific repair is claimed. Executed mean/change and native-tail accounting shows why endpoint agreement is weaker than predicting context-sensitive interventions. Original18edge failures remain; no new circuit promotion.
