# Hourly strategic review — 2026-08-29 00:07 UTC

## Bottom line

The project has fully described one restricted function class—the 36-site,
context-free token-row program—but has not reverse engineered the live contextual
model. The newest prospective result prunes a single dense rank-3/rank-4 law across
the early-MLP/downstream boundary. The failure is repeatable rather than noisy: the
complete 49-cell interaction grids correlate 0.9963 across two document-disjoint
roles.

This review executed the preregistered fallback: an exact product-poset Möbius
decomposition plus sparse held-cell prediction. A 16-term sparse hierarchy predicts
each omitted cell with normalized error 0.365/0.347 and R2 0.749/0.767. Directly
transporting a 16-term program from one role to the other gives normalized error
0.194/0.197. More terms worsen leave-one-cell-out prediction. This is a real
simplicity-versus-prediction curve, but it is retrospective and remains a grammar
hypothesis, not a promoted causal interface.

The immediate mathematical ambiguity is now concrete: the suffix registry never
measured MLP-only layers 3--8. Consequently, the apparently stable “broad MLP”
Möbius terms alias the early-prefix×MLP interaction with a three-way
early-prefix×attention×MLP contrast. The cheapest prospective experiment is to add
that missing suffix column across all eight early prefixes on both roles.

## What fraction is actually explained

| Object | Current status | Honest interpretation |
|---|---:|---|
| Structural site coverage | 36/36 | Every attention/MLP output has a surrogate hook; this is engineering coverage, not semantic or causal understanding. |
| Certified original storage removed | 5.3481% | The only settled whole-program storage certificate. |
| Context-free position-wise class | optimum attained on covered tokens | Full-rank rows attain the model-defined length-1/per-token ceiling, so further optimization inside this class cannot recover context. |
| Live behavioral gap | about 2.74 CE nats | The dominant unexplained term is contextual computation, not residual rank or table fitting. |
| Context-free top-1 | 13.9--14.7% versus live 38.9--42.4% | The compiled class preserves only a minority of live next-token decisions; accuracy is not an “explained fraction” with linear arithmetic. |
| Older named behavior | 32.1% ± 6.4% | Historical behavioral ledger, weaker than the strict causal ledger. |
| Strict named causal CE recovery | 10.923% | 4.72714 nats remain unexplained in that ledger. |
| Dense early-MLP/context interface | 0 selected ranks | Both prospective rank 3 and rank 4 failed on two roles. |
| Final extraction/removal/OOD actions | 0/68 | Physical routing/capture exists, but the final semantic reducer and consequence certification remain incomplete. |

No explained-fraction ledger moves in this review. The new 16-term result concerns
prediction of a 49-cell scalar interaction surface; it is not a fraction of the model.

## Largest remaining gaps

1. **Context is still missing.** The exact context-free ceiling remains roughly 2.74
   nats from live behavior. Rank, coverage, fallback, and independent table choice
   cannot close it.
2. **The interface between early token-local writes and later contextual computation
   is not yet composable.** Dense low rank failed; sparse hierarchy is only
   retrospective.
3. **MLP0's continuous code lacks downstream-defined semantic variables.** Gauge
   rotations make raw coordinates arbitrary. Stable causal response classes—not
   visually named axes—must define the quotient.
4. **Scalar CE can conceal incompatible behavior.** No vector-valued token/logit or
   consumer response has yet promoted the new interface.
5. **The final 68-action result remains 0/68.** The missing scientific reducer must
   supply reviewed gauge/SVD/difference-in-differences replays and frozen
   copy/frequency comparator plus interval semantics.
6. **No genuine OOD or selective-removal certificate exists for a simplified whole
   program.** The two roles used here are disjoint FineWeb documents, not a new
   distribution.

## New result: sparse causal hierarchy after dense low rank failed

For early replacement set $P_i$ and suffix replacement set $S_j$, the completed assay
measured the non-additive CE interaction

$$
\Delta_{ij}=C_{ij}-C_{i0}-C_{0j}+C_{00}.
$$

The early masks are the full Boolean lattice of subsets of MLP0/1/2. The suffix masks
are a nested registry of downstream macro-replacements. Their product-poset Möbius
transform writes

$$
\Delta = Z_P M Z_S^\top,
$$

where $Z_P$ and $Z_S$ record set containment and $M$ contains hierarchical
interaction contrasts. This is an exact change of coordinates; exact reconstruction
alone is tautological. The operational test is whether a small support in $M$ predicts
an omitted cell.

Orthogonal matching pursuit was run inside each leave-one-cell-out fold. It selected
only from the other 48 cells, with train-fold column normalization and unsupported
principal-upset terms assigned zero rather than leaked from the omitted cell.

| role | Möbius terms | leave-one-cell-out NRE | leave-one-cell-out R2 |
|---|---:|---:|---:|
| skip7000 | 8 | 0.4449 | 0.6283 |
| skip7000 | 16 | **0.3654** | **0.7493** |
| skip7000 | 24 | 0.4683 | 0.5881 |
| skip11000 | 8 | 0.4230 | 0.6539 |
| skip11000 | 16 | **0.3469** | **0.7672** |
| skip11000 | 24 | 0.3904 | 0.7052 |

NRE is RMSE divided by the zero-interaction/additive baseline; values below one are
useful. The optimum at 16 rather than 24 or 32 terms is evidence that complexity
control improves prediction, not merely description.

The grammar also transports across document roles:

| source → target | terms | direct-value NRE | target-refit-on-source-support NRE |
|---|---:|---:|---:|
| skip7000 → skip11000 | 16 | 0.1943 | 0.1728 |
| skip11000 → skip7000 | 16 | 0.1967 | 0.1689 |

The complete source grid transfers at NRE 0.096/0.104, consistent with the raw
cross-role correlation of 0.9963. This demonstrates stable same-corpus population
structure, not OOD.

Six eight-term macro-contrasts are selected in at least 80% of 1,000 independent
document bootstraps on each role. They concentrate on MLP2 with the additional
attention4--8 and local attention3/MLP3 contrasts, MLP1+MLP2 with attention3, MLP1
with additional attention4--8, MLP0+MLP1 with MLP3, and MLP0 with MLP3. This supports
the earlier finding that MLP0/1/2 must be modeled jointly.

### Critical claim boundary

The suffix side is not a physical factorial. MLP-only layers 3--8 and 9--17 never
appear alone. Suffix Möbius coefficients are therefore registry-macro contrasts,
not identified native-site mechanisms or executable tensor-program terms. The zeta
basis is nonorthogonal, so squared coefficient magnitudes are not variance/energy.
Only held-cell prediction is used as a usefulness measure.

## Updated simplicity picture

There is no single correct scalar simplicity measure yet. Three operational axes now
survive:

1. **Executable description cost:** stored real numbers, zero-native calls, FLOPs,
   and graph size. The coverage×rank Pareto frontier is the settled example.
2. **Predictive interface complexity:** number of hierarchical causal terms required
   to predict unseen compositions. The new curve has a clear 16-term knee.
3. **Consequence complexity:** how well a representation predicts extraction,
   selective removal, collateral damage, and OOD transport at matched fidelity. This
   axis remains unmeasured for the final program.

The newly completed rank frontier reinforces that these cannot be collapsed. CE is
monotone with rank, but rank 1024 has slightly better top-1 than full rank on all
three roles. A representation can be closer in CE without maximizing discrete
decision agreement.

## Candidate actions considered and pruned

- **Increase dense cross rank or choose a new pivot on the revealed grid:** pruned.
  This would tune after failure, worsens conditioning, and violates the registered
  claim boundary.
- **Call the exact Möbius transform a compression:** pruned. The full transform has
  the same 49 degrees of freedom and no executable savings.
- **Use coefficient squared norm as explained energy:** pruned. The zeta basis is
  nonorthogonal.
- **Treat the two FineWeb roles as OOD:** pruned. They are document-disjoint
  replication roles from the same corpus.
- **Fit MLP0 semantics directly from rotated rank coordinates:** deferred. Gauge
  freedom makes this arbitrary until downstream response roles define a quotient.
- **Continue rank/coverage mapping inside the context-free class:** demoted. That
  curve is now dense and replicated; it cannot address the contextual gap.
- **Generic low-rank plus a few outliers:** already pruned. Prior residual effective
  support was 16--21 cells, and the current failure is organized by intervention
  class rather than a few anomalous cells.

## Top five priorities

### 1. Add the missing MLP-only layers-3--8 suffix column prospectively

Cross the new suffix with all eight early-prefix masks on both document roles: 16
role-cell evaluations. Together with the existing empty, attention-only, and
all-sites layers-3--8 columns, this completes the broad attention×MLP square. It
identifies attention×MLP suffix synergy on every early-prefix background and tests
whether the early-prefix×MLP interaction is invariant to attention replacement—the
largest ambiguity in the new sparse grammar. It is cheap, falsifiable, causally
relevant, and composes at the same physical boundary. Freeze support, signs, gates,
and document bootstrap before opening outcomes.

### 2. Close the 68-action semantic reducer and make the interface vector-valued

The physical 68-action path exists, but scalar CE cannot validate extraction or
selective removal. Finish the reviewed gauge/SVD/difference-in-differences replays,
copy/frequency comparator, and uncertainty semantics. Reuse these response vectors
as outputs of the composition test. This is the highest direct route to semantic and
editability credit, but costs more implementation and GPU time than priority 1.

### 3. Test the frozen sparse grammar on an adjacent physical cut

If priority 1 de-aliases cleanly, move the same hierarchy one boundary deeper without
changing support selection. Passing two adjacent cuts would provide evidence for a
composable program interface; failing would localize the grammar to one layer family.

### 4. Fit a joint downstream-weighted MLP0/1/2 dictionary

Use the stable causal macro-contrasts—not arbitrary rank coordinates—as targets for a
shared sparse dictionary over folded MLP weights/outputs. Price nonzeros, dictionary
atoms, and downstream response error jointly. Require heldout composition or action
prediction; do not promote weight reconstruction alone.

### 5. Compare simplicity definitions through consequences

At matched CE/KL, compare rank, stored bits, sparse causal terms, graph nodes/edges,
and gauge-quotiented degrees of freedom on extraction, selective removal, collateral,
OOD transport, and runtime. This validates a simplicity definition by what it enables,
not by self-reconstruction.

## Action executed in this review

Implemented and ran:

- `early_mlp_context_mobius_diagnostic.py`;
- `test_early_mlp_context_mobius_diagnostic.py` — 4/4 tests pass;
- `early_mlp_context_mobius_diagnostic_results.json`;
- `EARLY_MLP_CONTEXT_MOBIUS_DIAGNOSTIC.md`.

Then froze and mathematically red-teamed
`BROAD_MLP_SUFFIX_DEALIAS_V1_PREREGISTRATION.md` before any MLP-only layers-3--8
outcome. It fixes the 16 role-cell experiment and the no-fit prediction
$\widehat D^M=D^{AM}-D^A$. Red-team caught and corrected the initial estimand label:
$Q=D^{AM}-D^A-D^M$ is the three-way early×attention×MLP contrast, while raw
attention×MLP suffix synergy $R_i$ is measured separately on every prefix background.
The final protocol has two independent 2,000-document bootstraps, conditional
cross-role intervals, exact old/new document joins, and fail-closed claim boundaries.
Execution remains NO-GO pending a source-closed implementation amendment and audit.

The diagnostic verifies the sealed measurement receipt, performs an exact
product-poset round trip, computes singular spectra only descriptively, performs
train-fold-only sparse selection for every omitted cell, transfers support and values
between roles, and repeats support selection over 1,000 document bootstraps per role.

Independent mathematical red-team ruling: GO only as descriptive grammar search;
NO-GO for causal-interface, executable compression, or OOD claims. Its cheapest
recommended falsifier agrees with priority 1.

## Jobs and coordination

- The early-MLP/context GPU measurement and sealed CPU score are complete.
- Claude's 16,110-type rank/top-1 frontier job completed in 954 seconds during this
  review. Its files are currently uncommitted and were not modified or staged here.
- The GPU is free at the end of this review.
- No unrelated runlog or frontier worktree changes were touched.
