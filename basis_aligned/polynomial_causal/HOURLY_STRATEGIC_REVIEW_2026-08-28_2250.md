# Hourly strategic review: test the missing early-MLP/context interface

**Time:** 2026-08-28 22:50 UTC

**Executed CPU action:** froze and fail-closed a new 8 by 8 early-MLP/context
tensor-cross design. No model outcome, final role, or GPU forward was opened.

## Bottom line

The largest scientific gap is no longer a missing local decomposition. We already
have replacements for all 36 sites and very compact local descriptions at several
sites. The missing object is the small state, if one exists, that predicts how an
early replacement changes the behavior of the contextual suffix. Independent local
choices have repeatedly failed when composed, and MLP2 changes from harmful to
helpful depending on MLP0/MLP1. A whole-interface interaction test is therefore more
informative than another standalone PCA, HOSVD, SAE, or rank sweep.

This review implemented that test's statistical core. Its rows are the eight subsets
of MLP0/1/2. Its columns are eight suffix substitutions ranging from empty through a
single layer-3 site to all attention and MLP sites in layers 3--17. After removing
the additive row and column effects, a frozen rank-three cross must predict seven
untouched expansion cells; a nested rank-four cross must predict nine final heldout
cells. Cross-entropy chooses the masks; top-1 is an outcome.

Independent audit caught a genuine staged-access defect before execution: the first
implementation algebraically ignored heldout cells but still validated and copied
the full 64-cell tensor. The replacement API accepts exactly the cells licensed for
each stage and rejects every extra cell, including a poisoned NaN. Six focused tests
now pass. Launch remains explicitly **NO-GO** until a source-closed two-role collector,
bootstrap/ALS scorer, immutable program bank, and lifecycle receipts are implemented
and reviewed.

## Honest fraction explained

These figures have different denominators and must not be collapsed into one score.

| Question | Current evidence | Unexplained part |
|---|---:|---:|
| Structural surrogate exists for each attention/MLP site | 36/36 | semantics and composition |
| Complete-program storage consequence-certified removable | 5.3481% | 94.6519% |
| Older human-readable behavioral account | 32.1% +/- 6.4% | most measured behavior |
| Strict named causal CE headroom recovered | 10.923% | 4.72714 nats = 89.077% |
| Final early-MLP/suffix causal actions scored | 0/68 | entire final response tensor |

The full-rank per-token program exactly reaches the model's own length-one ceiling on
covered positions. Roughly 2.74 nats still separate that context-free ceiling from
the live model. That is the cleanest current localization of what remains: contextual
computation and its interfaces, not more rank within a current-token lookup.

## Largest remaining gaps

1. **Interface state.** We cannot yet predict when individually good substitutions
   will compose. This is the direct target of the new cross.
2. **Causal response tensor.** Physical plumbing for the 68 early-program actions
   exists, but the reviewed semantic producer for objective, transport, gauge, SVD,
   and difference-in-differences gates does not. The final role remains unopened.
3. **Executable closure.** Many useful probes still call the native module and then
   replace its output. They test behavioral abstractions, not zero-native-call small
   programs.
4. **OOD transport.** The two proposed roles are document-disjoint replications but
   have been used before; a pass would not establish fresh-corpus or code OOD.
5. **Semantic coordinates.** Low rank is known in several downstream-weighted spaces,
   but the axes are not yet stable named variables that support selective edits.

## Why tensor cross is the right next mathematical move

For a CE-cost grid \(H\), define the nonadditive interaction

\[
\Delta_{ij}=H_{ij}-H_{i0}-H_{0j}+H_{00}.
\]

If the early package communicates with the suffix through a small state, \(\Delta\)
should be approximately low rank. A rank-\(r\) cross measures only selected rows and
columns and predicts every other entry by

\[
\widehat\Delta=\Delta[:,J]\,\Delta[I,J]^{-1}\,\Delta[I,:].
\]

This is stronger than a retrospective SVD: it predicts programs never used to fit
the factorization. It also exploits the tensor network's physical order. Passing the
same state size at adjacent cuts would license a tensor-train or minimal action-state
model; failure would cheaply prune that route.

The frozen design uses rank three for discovery and rank four for replication because
the earlier CE rank-four maximum-volume pivot repeated in 1,999/2,000 document
bootstraps. Top-1 chose the same pivot only 860/2,000 times, so top-1 is too
discontinuous to choose experimental masks. A CE rank-two pivot was perfectly stable
but left median interaction error 0.4741: stability alone does not imply adequacy.

## Candidate actions, pruned and ranked

### 1. Close and run the prospective early-MLP/context cross

- **Information gain:** highest; directly tests unseen compositions at the main gap.
- **Causal relevance:** high for behavioral substitution, with top-1 as a protected
  secondary outcome.
- **Whole-model composability:** high conditional on transfer to an adjacent cut.
- **Falsifiability:** excellent; masks, pivots, stages, and gates are frozen.
- **GPU cost:** moderate and bounded at 64 masks across two 192-row roles.
- **Current state:** statistical registry implemented; launch NO-GO pending the
  source-closed runner/scorer/lifecycle amendment.

### 2. Finish the 68-action semantic producer, then score vector consequences

- **Information gain / causal relevance:** highest once semantics are fixed.
- **Composability:** directly covers early program to 18-layer suffix.
- **Falsifiability:** strong after comparators and point-versus-bootstrap conventions
  are frozen.
- **GPU cost:** high, but most physical execution machinery already exists.
- **Reason second:** the remaining semantic ambiguity can invalidate a final run;
  the cross can first establish whether a low-dimensional response state is plausible.

### 3. Repair the projected MLP0/MLP1 plus exact-MLP2 cube

- Tests whether the observed 64-dimensional output interfaces preserve conditional
  causal value jointly.
- It is complementary to singleton PCA because exact MLP2 exposes compensation.
- It remains an oracle-interface assay, not executable compression, because it calls
  native MLPs. Source/lifecycle closure remains NO-GO.

### 4. Fit joint downstream-curvature-weighted early-MLP corrections

- Optimize errors in the directions later computation distinguishes, rather than
  Euclidean output MSE.
- Must fit MLP0/1/2 jointly and beat equal-cost PCA/table baselines on untouched CE;
  a sitewise version is redundant with failed independent reductions.
- Defer until the cross identifies whether a small shared consequence state exists.

### 5. Recover a minimal action-state realization after two adjacent crosses pass

- Factor a shifted action-Hankel block into a small state and layer-indexed
  transitions, producing the clearest candidate executable tensor program.
- Highest eventual composability and editability, low marginal GPU cost once response
  banks exist.
- Not yet identified: attempting it before two adjacent predictive crosses would
  repeat the earlier OOD token-splice Hankel failure.

## Pruned this hour

- Another rank-one/rank-two completion: already prospectively failed; CE rank two is
  stable but wrong.
- Sparse exception repair: rank-two residual energy is diffuse across roughly 16--21
  effective cells.
- Standalone weight SAE, PCA, HOSVD, or gauge norm minimization: these may compress a
  coefficient object but currently predict no new whole-model consequence.
- Independent per-site table selection or scalar gain repair: both miss the observed
  sign-changing interactions.
- More local MSE: a locally closer replacement has already produced worse downstream
  behavior.

## Exact action completed and remaining launch blockers

Added:

- `early_mlp_context_cross_v1.py`
- `test_early_mlp_context_cross_v1.py`
- `EARLY_MLP_CONTEXT_CROSS_V1_PREREGISTRATION.md`

The registry contains 15 anchors, 33 rank-three cross entries, seven rank-four
expansion cells, and nine heldout cells. Bootstrap seeds, ALS seed/restarts, cell
capabilities, rank-three/rank-four pivots, and CE/top-1 claim boundaries are fixed.
The physical program is corrected to the committed section-1786 rank-64 covered-table
plus learned rank-64 uncovered-map family; output nearest-neighbour is only a control.

Six tests pass, including exact synthetic rank-three and rank-four recovery,
forbidden fit-cell rejection, and exact validation/heldout score capabilities. The
independent auditor's first two NO-GO verdicts are preserved; after the repairs its
third review found no remaining pre-commit blocker. The remaining blockers are
implementation, not a missing scientific choice: full two-role document collector,
token-weighted bootstrap scorer, specified ALS execution, exact full-hash/source and
row pins, create-only namespace/lock, failure preservation, and terminal receipts.

No explanatory ledger value moves from freezing a prospective design. The next safe
action is to implement that lifecycle without opening any model outcomes.

## Concurrent work

Claude repaired and re-queued the untracked `rank_to_ceiling.py`; at this review the
GPU was idle and its owning shell was waiting for the queue. This branch did not edit
that source or its logs. Concurrent joint early-MLP PCA files also remain untouched.
