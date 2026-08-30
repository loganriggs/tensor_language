# Three-hour mathematical review — 2026-08-30 09:42 UTC

## Outcome first

The strict scientific ledger is unchanged: 5.348245316% certified removable storage,
10.923302467% named deletion-CE, 4.72714 nat / 89.076697533% unexplained deletion-CE,
and 0/68 terminal circuits. The audited signed-response FIT has not opened authority
or an outcome; it is still queued behind the live three-seed a8 learned-grouping job.

This review found and repaired a prospective **cost-definition error** rather than a
new model result. The response-factor preregistration priced 384 scattered scalar
cells as if each were one measurement. Physically, a `(phase, source)` intervention
forward returns all 49 target cells. Scattered cells can therefore touch nearly all 98
expensive arms. A simpler executable program must be priced in source-intervention
blocks, not favorable scalar entries.

The three genuinely new moves, ranked by current return, are:

1. block-D-optimal causal-response tomography;
2. sparse Volterra/polarization tomography of intervention composition;
3. orbit-closure/border-rank stability certificates for response factors.

## 1. Block-D-optimal causal-response tomography — executed

### Exact object in bilin18

After fitting a response candidate on 229 FIT-training documents, materialize only its
observation-mode basis

$$
B\in\mathbb R^{(2\cdot49\cdot49)\times K}.
$$

One physical sensor is the block of 49 target rows associated with a phase/source arm
$a=(p,s)$. For an orthonormal basis $Q$ of `range(B)`, its information contribution is

$$
M_a=Q_a^\top Q_a.
$$

At an arm budget $m\in\{2,4,8,16\}$, greedily maximize

$$
f(S)=\log\det\left(10^{-8}I+\sum_{a\in S}M_a\right).
$$

The program then infers a small document code from the selected complete blocks and
predicts all nonselected source/target responses.

### Theorem or operational definition

For additive positive-semidefinite information matrices and a positive prior,
log-determinant is a monotone submodular set function; cardinality-constrained greedy
selection obtains the standard $(1-1/e)$ objective guarantee. Q-DEIM supplies the
closely related pivoted-QR interpolation construction, sharper conditioning bound, and
the key invariance: replacing an orthonormal subspace basis by $Q\Omega$ does not
change the selection operator. See [Drmač and Gugercin,
2016](https://doi.org/10.1137/15M1019271) and [Shamaiah, Banerjee, and
Vikalo](https://sidbanerjee.orie.cornell.edu/docs/CDC_sensorsel.pdf).

Our block objective is gauge-invariant under arbitrary invertible code changes because
we first replace $B$ by an orthonormal basis of its column space; two such bases differ
by an orthogonal matrix, which conjugates each $M_a$ without changing determinants.

### Assumptions that may fail

- The fitted response span may not transport to new documents.
- D-optimality minimizes information-ellipsoid volume, not worst-block response MSE.
- Missing target support can make a nominally informative arm unusable on some
  documents.
- A candidate-specific panel can overfit the FIT-training span; validation and EVAL
  remain necessary.
- If the response is nonlinear in simultaneous edits, a good single-edit sensor panel
  need not predict composed removals.

### Consequence beyond reconstruction

At the same number of physical intervention forwards, the selected arms must predict
unmeasured signed cells with lower validation error or higher document support than an
outcome-blind block panel. If eight arms suffice, deployment calibration falls from 98
source interventions to eight—about a 12.25-fold reduction—while retaining a tensor
program and an explicit code solve. The learned arms also identify which interventions
are maximally diagnostic of the entire response library.

### Cheapest falsifier

For every frozen candidate and arm budget, compare block-D-optimal and hash-selected
panels on the 114 internal-validation documents. Reject the computational-simplicity
claim if D-optimal does not strictly improve non-anchor signed MSE/support at matched
arm count, if worst owner-pair NRMSE worsens, or if fewer than 90% of documents support
the code solve.

### Executed proof check

`CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_1.md` now controls the physical price.
Nine CPU tests pass in 6.11 seconds. They prove complete-block masking, deterministic
selection, monotone log-determinant progress, invariance under an arbitrary invertible
code gauge, and rejection of scattered-cell false pricing.

The planted noisy receipt
`causal_response_block_design_planted_toy_v1.json` used four physical arms, an
eight-dimensional code, and 200 documents. At identical arm price:

- D-optimal non-anchor MSE: `0.0006586049`;
- hash-block non-anchor MSE: `0.0125900453`;
- ratio: `0.05231` (D-optimal used 5.23% as much MSE);
- smallest selected-design singular value: `8.8616` versus `2.4603`.

This is a planted known answer, not evidence that bilin18 is low-dimensional.

## 2. Sparse Volterra/polarization tomography of composition

### Exact object in bilin18

Let $a\in\mathbb R^{49}$ contain amplitudes of source-direction deletions and let
$F_d(a)\in\mathbb R^{49}$ be the signed target-response vector on document $d$. Around
the native model,

$$
F_d(a)\approx F_d(0)+J_da+rac12\mathcal H_d[a,a].
$$

$J_d$ is the single-edit causal Jacobian. The symmetric tensor $\mathcal H_d$ records
pairwise synergy or compensation—the missing object when independently simplified
MLP0/1/2 components fail to compose.

### Theorem or operational definition

Finite Volterra models are polynomial system-identification models. When only a small
number of kernels are nonzero, random bounded designs plus sparse regression can
recover the active terms with sample complexity controlled by sparsity rather than the
full polynomial feature count. A directly relevant primary treatment is [Kekatos and
Giannakis, *Sparse Volterra and Polynomial Regression Models: Recoverability and
Estimation*](https://arxiv.org/abs/1103.0769); earlier sparse-kernel identification is
given by [Yao, Sethares, and Hu](https://doi.org/10.1109/ICSYSE.1992.236898).

For a pair $i,j$, central polarization at amplitudes $\pm\epsilon$ isolates the
quadratic interaction while canceling the constant and first-order terms. Random
Rademacher amplitude vectors can estimate many sparse terms simultaneously.

### Assumptions that may fail

- Unit deletion may be too far from the Taylor/Volterra regime.
- RMSNorm and later bilinear layers can create important third and higher orders.
- Interaction tensors may be dense rather than sparse.
- CE responses may be noisy and document-dependent.
- Simultaneous projections at different sites need an exact typed execution order;
  commutativity cannot be assumed.

### Consequence beyond reconstruction

A successful second-order program predicts the effect of unseen pairs and sparse edit
sets, directly diagnoses MLP2 compensation for MLP0 simplification, and identifies
edits whose collateral effects cancel or amplify. This improves composition and
selective removal rather than only fitting single-edit responses.

### Cheapest falsifier

After the single-edit FIT tensor, choose six sources spanning different owner
components. On fresh FIT-only documents, measure central `±0.25` singles and 15 pairs.
Test whether the polarized quadratic model predicts held-out pairs and whether doubling
to `±0.5` preserves coefficient sign/order. Reject before a 49-source compressed run if
the quadratic model fails the pair holdout or higher-order residual dominates.

## 3. Orbit-closure and border-rank stability certificate

### Exact object in bilin18

Apply this to each fitted global/private response block, not to the raw whole model.
CP and block-term factors have scale, permutation, and block-basis gauge actions. A
candidate can approach a low reconstruction error while individual factor norms
diverge and cancel. That is a border-rank degeneration: the tensor lies near the
closure of a lower-rank orbit, but the proposed atoms are not stable editable objects.

### Theorem or operational definition

For reductive group actions, the Kempf–Ness norm-minimization problem selects a
minimum-norm balanced representative when the orbit is closed; failure to attain a
nonzero minimum/capacity diagnoses null-cone or orbit-closure pathology. Tensor and
operator-scaling algorithms make this constructive; see [Bürgisser et al.,
*Alternating minimization, scaling algorithms, and the null-cone problem*](https://arxiv.org/abs/1711.08039)
and the tensor-network-specific [minimal canonical form](https://arxiv.org/abs/2209.14358).

The operational certificate is stricter than ordinary norm balancing:

1. balance all registered exact gauges;
2. compare minimized norm/capacity across seeds and document bootstraps;
3. compute the quotient Jacobian's smallest non-gauge singular value;
4. perturb each balanced atom and verify predicted response changes remain bounded.

### Assumptions that may fail

- The exact real response-factor gauge may not be the complex reductive action used by
  the theorem.
- Block sparsity and zero loadings create singular strata.
- Numerical capacity estimation can mistake very small for zero.
- A closed stable orbit can still lack semantic meaning or causal selectivity.

### Consequence beyond reconstruction

A passing candidate has reproducible atoms modulo known gauges, bounded edit
condition numbers, and a principled reason to expect factor removal to mean the same
thing across seeds. A failing candidate is retained only as a compressed predictor,
not an editable circuit decomposition.

### Cheapest falsifier

On the planted factor toy and then each FIT frontier point, track reconstruction error,
balanced factor norm, and quotient-Jacobian gap across the three frozen seeds. Reject
atom-level interpretation if error improves while balanced norms diverge, excess
nullity remains, or resampled atoms fail to align after quotienting gauges. No new
model forward is required.

## Pruned after reconsidering the requested mathematics

- **Scalar-row Q-DEIM:** mathematically sound but prices the wrong physical unit;
  replaced by complete target-block D-optimal selection.
- **Another raw-weight norm-minimization/HOSVD pass:** scalar gate balancing and the
  Kempf–Ness interpretation are already documented, and bilin18 was already surveyed
  near its raw MLP scalar gauge. The new use is only the response-factor border-rank
  certificate.
- **Generic token-prefix Hankel/weighted automata:** the old splice object was severely
  OOD and did not stabilize at a small rank. The exact Hankel-rank/minimal-realization
  theorem remains valid ([Carlyle and Paz](https://doi.org/10.1016/S0022-0000(71)80005-3)),
  but the present token object violates the useful assumptions. Retain only a future
  action-Hankel after composed interventions exist.
- **Information bottleneck:** estimating mutual information adds an unstable scalar
  objective and gives no executable decoder, intervention panel, or selective-removal
  guarantee. Block response prediction is a stronger operational sufficiency test.
- **MDL/prequential coding alone:** retain as a price/statistical-efficiency axis after
  a functioning program exists; it cannot validate editability or causal composition.
- **Unconstrained sparse program synthesis:** premature before response atoms pass
  gauge stability and pair-composition tests.
- **Whole-model polynomial expansion:** exact local bilinearity does not prevent degree
  and term explosion across 18 RMSNorm/residual interfaces. Sparse second-order
  intervention tomography is the falsifiable bounded alternative.
- **Causal bisimulation now:** single-edit response signatures are observations, not a
  transition-closed action system. Bisimulation becomes meaningful only after the
  Volterra/action panel tests composition.

## Ranked next actions

1. Preserve and later run the block-D-optimal versus hash-block comparison on the FIT
   frontier; this is now implemented and prospectively controlled.
2. After the signed single-edit tensor exists, preregister the six-source central
   polarization pilot before any simultaneous-edit outcome.
3. Add orbit-closure diagnostics to the existing quotient-Jacobian toy and apply them
   to frontier points before EVAL or atom naming.

The active GPU priority does not change: finish the already audited signed-response
FIT. This review used only CPU planted data and primary literature; it opened no model
outcome.

## Execution update after the mathematical review

At 09:46 UTC the a8 owner reported 26.24 GiB free and explicitly cleared concurrent
second-lane use. The exact audited no-argument FIT owner therefore started rather than
waiting another 30–40 minutes. Its authority binds source commit `583b2244`, closure
`2d3fddb1`, 12,400 forwards, 496 rows, and 343 documents; FIT uses 5.18 GiB alongside
a8's 5.86 GiB with more than 21 GiB headroom. No response result exists yet.

The shell's optional `tee` target directory was absent, so no run log is attached.
The sole Python owner had already acquired authority and remains live; it must not be
relaunched. This wrapper failure is operational provenance, not a scientific failure
artifact. The audited lifecycle remains responsible for publishing either its exact
receipt or exact failure.

## Receipt-bound analysis boundary completed at 09:52 UTC

The highest-priority CPU-side prerequisite is now executable rather than merely
described. `causal_response_factorization_v1_fit_adapter.py` accepts only an
already-loaded FIT payload plus its expected authority digest. It:

1. replays the complete semantic FIT-bundle validator;
2. rejects a payload bound to any other authority;
3. computes the signed member-minus-off response from additive sums and counts;
4. applies the frozen 229/114 document split in production;
5. derives the six fixed owner groups from the sealed source order; and
6. returns independent contiguous CPU clones to the factor optimizer.

It has no filesystem, corpus, model, or EVAL capability. It therefore cannot by
itself prove an artifact digest; the next lifecycle layer must first verify the exact
bundle bytes against the terminal FIT receipt, then pass the in-memory payload through
this adapter. This division prevents either code layer from silently opening EVAL or
substituting another FIT artifact.

The adapter and the factor, bundle, and semantic-tamper suites pass 32/32 tests in
24.43 seconds. Tests verify the exact signed-response arithmetic, frozen role
partition, owner topology, non-aliasing, authority rejection, semantic tamper
rejection, and absence of a file/EVAL surface. This is infrastructure validation, not
bilin18 evidence. The audited FIT remained live during the test at 5.6 GiB; no result
or failure artifact existed yet.
