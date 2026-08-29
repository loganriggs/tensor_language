# Hourly strategic review — 2026-08-29 23:35 UTC

## Bottom line

The newest result sharpens the shipped-program rank allocation but does not explain
more of the native model. Raising only MLP17's table rank from 768 to 1,152 improves
cross-entropy (CE) on all three document roles at both tested table coverages. At
coverage 5,419 the improvements are `0.000631 / 0.000296 / 0.000509` nat; at coverage
16,110 they are `0.001642 / 0.000821 / 0.001779` nat. The latter result is larger,
passes all controls, and clears the frozen storage price on every role. MLP16 is less
stable and remains a worse allocation target.

This is an economic knee for one table, not a semantic interpretation: it trades
2.523 million extra stored values at coverage 5,419 (6.629 million at coverage 16,110)
for a small CE gain inside a program whose pooled CE remains about `5.86--5.94`, versus
native CE near `3.14`.

The highest-information native-model experiment remains the MLP2 error-Rayleigh
validity pilot. Its independent audit correctly issued a **NO-GO before row access**:
one direct metric test was missing from source closure and the wrapper leaked temporary
configuration into its parent module. Both defects are repaired at commit `87475ab7`;
the combined outcome-blind suite now passes `32/32`. A fresh audit is running. No rows,
model responses, or outcomes have been opened.

## What fraction is actually explained?

The strict whole-model balance sheet is unchanged:

- all `36/36` component sites can be intervened on structurally;
- `5.348245316%` of storage has certified removal;
- `10.923302467%` of causal CE has a named explanation;
- `4.72714` nat, or `89.077%`, remains causally unexplained;
- `0/68` proposed terminal extraction/removal/OOD actions are certified.

The latest rank results improve one deployable table program. They do not move these
strict native-model numbers.

## Terms and computations

The **shipped program** replaces all 36 attention/MLP components by token-indexed
tables. A table's **coverage** is the number of token identities with fitted rows;
uncovered tokens use a registered fallback. A table of rank $r$ stores a factorization
instead of a dense token-by-output matrix.

For a site $j$, define the benefit of increasing its rank as

$$
B_j = CE(\text{rank-768 shipped program})
-CE(\text{same program with site }j\text{ at rank 1,152}).
$$

Positive $B_j$ means the larger table predicts the next token better. Pooled over
92,160 paired token losses at coverage 5,419,

$$
B_{16}=0.000602\pm0.000103\ \text{nat},\qquad
B_{17}=0.000472\pm0.000132\ \text{nat},
$$

where the displayed uncertainty is one standard error. MLP16 nevertheless worsens
one complete document role by `0.000080` nat, so it fails the preregistered stability
criterion. MLP17 improves every role and improves even more when coverage triples.

Increasing both ranks yields `0.000962` nat pooled, smaller than
`0.000602+0.000472`. Thus the gains are **sub-additive**: once one late table is made
more accurate, part of what the other could repair has already been repaired. This is
the opposite-sign counterpart of the super-additive loss observed when both tables
were truncated.

The **error-Rayleigh pilot** addresses a different question. If $E$ is a rank-512
MLP2 program's write error and $J_c$ is the local derivative of a downstream consumer
$c$, it measures consequences such as

$$
q_{\mathrm{logit}}(E)=(J_{\mathrm{logit}}E)^T
F_{\mathrm{categorical}}(J_{\mathrm{logit}}E),
$$

plus separate attention-5 and attention-6 response energies. This asks whether an
MLP2 error points in a direction downstream computation can actually see. The metric
must predict the already-observed finite MLP0-C512 by MLP2 composition penalty on held-
out documents; good local reconstruction alone is not enough.

## Largest remaining gaps

1. **Missing native interface metric.** We do not yet know whether Fisher/consumer-
   weighted MLP2 error predicts actual finite CE and composition.
2. **Lossy all-table abstraction.** Its CE gap from native is about 2.7--2.8 nat; rank
   tuning inside it cannot recover attention-dependent computation that the tables
   removed.
3. **Unclosed early composition.** MLP0-C512 and each MLP2 rank-512 program work alone,
   but their joint CE has a reproducible `0.0074--0.0086` nat mixed penalty.
4. **Sparse verified consumer bank.** Copy has a causal four-head bundle, but its
   position-mean replacement fails collateral-damage gates; capitalization, numeric,
   syntax, and entity consumers remain unverified.
5. **No terminal edit.** Nothing yet passes extraction, selective removal, collateral,
   and OOD transport end to end.
6. **Incomplete causal rank allocation.** The shipped program has causal curves at
   only a few sites; independent curves also require joint validation.

## Pruning and ranked next actions

Scored qualitatively by expected information gain, native causal relevance,
composability, falsifiability, GPU cost, and non-duplication:

1. **Finish the audited MLP2 error-Rayleigh validity pilot.** It can validate or kill
   the proposed definition of consequence-weighted simplicity before another fit and
   directly targets native composition. The lifecycle audit, not GPU cost, is the
   current critical path.
2. **Adopt/test the MLP17 rank-1,152 candidate as a shipped-program allocation change.**
   It passed two-coverages controls and the frozen price gate; preserve it as an
   executable compression-quality improvement, while keeping its scope explicit.
3. **Fit the direct mixed MLP0×MLP2 functional only if the Rayleigh metric predicts
   finite effects.** Otherwise the pilot falsifies its proposed geometry cheaply.
4. **Run the C512×MLP1×MLP2 factorial.** This distinguishes a special MLP0→MLP2 fault
   from a general early-interface failure and tests whole-model composition.
5. **Expand the verified late-consumer bank, then form a causal quotient/Hankel state.**
   Capitalization and numeric-formatting are useful next consumers because their late
   outputs can provide interpretable coordinates for what early MLPs write.

Pruned branches remain: uniform rank allocation, fit-energy allocation, mean-row
deletion, more unweighted local-MSE fitting, sparse document gates, and calling a
runner or preregistration an experimental result.

## Executed action, runtime, and blockers

- The rank-1,152 discovery took `267.4 s`; its independent two-coverage candidate
  replay took `199.8 s` and passed all registered controls.
- The Rayleigh wrapper repair and expanded combined test suite took under four seconds;
  `32/32` tests pass.
- Commit `87475ab7` is pushed and bound for a fresh independent audit.
- The GPU is currently free. It is not the blocker. The exact blocker is an independent
  source-bound GO for the repaired row transaction. Launching before that GO would
  invalidate the prospective split, so no rows or model outcomes have been opened.

The eight-hour entry-point deadline expired at 12:00Z. Its literal status remains six
measured negative cells (E1.1, E1.3, E2.1, E2.2, E3.1, E3.2), three scientifically
pruned cells (E1.2, E2.3, E3.3), and E4.1--E4.3 open. Family F has a preserved numerical
negative. No plan or unrun code is counted as an outcome.

## Post-review execution update — 23:42 UTC

The fresh re-audit returned GO: `53/53` combined tests passed, 156 registry artifacts
were inspected recursively, and audit SHA-256 is
`bfbf140f9d68723ef0960bf8e476d88e45a8e5ac1c5f801f83020530c4900e22`.

The authorized row freeze then completed in `27.3 s`. It created 32 DESIGN and 32
HELDOUT source-document rows of 257 tokens. All eight freshness/uniqueness gates pass;
the two roles have no full-row overlap; installed file and tensor hashes replay; and
the receipt records `model_loaded=false` and `training_run=false`. This is a real
completed evidence boundary, not a model result. The next unblocked step is to
source-close and audit the model-response collector; HELDOUT remains unopened for
model selection.
