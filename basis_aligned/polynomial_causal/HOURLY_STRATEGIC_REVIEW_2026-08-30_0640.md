# Hourly strategic review — 2026-08-30 06:40 UTC

## What changed since the 06:25 review

The proposed next step—fit shared/private blocks to the a8/a16/M16 causal response
tensor—was audited before fitting.  That audit found a missing interface that makes the
fit invalid with current artifacts.

The stored cells are not causal effects.  They are concentration ratios

\[
C_{s,t}=
\frac{\mathbb E[|\Delta CE_s|\mid\text{member positions of }t]}
     {\mathbb E[|\Delta CE_s|\mid\text{off-slice positions of }t]}.
\]

They are excellent localization scores, but ratios of absolute values do not add when
interventions compose.  They also discard sign, scale, numerator/denominator, and
document variation.  CP, BTD, HOSVD, or a DAG fitted to these ratios would model the
measurement convention rather than a composable tensor program.

I stopped the exploratory nonlinear fit rather than report an attractive but
mathematically meaningless result.  A CPU contract for a lawful response tensor was
implemented and tested instead.

## Strict whole-model balance

No new model outcome was opened, so the ledger remains:

- certified storage: **5.348245316%**;
- named causal CE: **10.923302467%**;
- unexplained CE: **4.72714 nat = 89.076697533%**;
- terminal actions jointly passing extraction, selective removal, and OOD: **0/68**.

The gap is not a shortage of localized behavior slices.  Sixty-two curated circuits
localize on held-out rows.  The gap is a lawful shared interface that says which
localized effects compose, which are broad services, and which edits are independent.

## Exact interface now required

For every intervention phase `p`, source direction/program `s`, target behavior `t`,
and evaluation document `d`, retain:

\[
(S^m,N^m,S^o,N^o,A^m,A^o)_{p,s,t,d},
\]

where:

- `S^m`, `S^o` are signed sums of per-position `ΔCE` on member and off-slice positions;
- `A^m`, `A^o` are the corresponding absolute sums, retained for localization only;
- `N^m`, `N^o` are position counts.

The additive response used for factorization is

\[
R_{p,s,t,d}=\frac{S^m}{N^m}-\frac{S^o}{N^o}.
\]

This is a linear functional of the signed `ΔCE` vector.  It retains document identity,
so train/evaluation splits and document bootstraps remain valid.  Direction fitting
must use documents disjoint from response evaluation; otherwise the same rows choose
and score the direction.

The absolute concentration ratio can be reconstructed from the same record, so this
interface strictly contains the old diagnostic rather than replacing it.

## CPU proof/contract executed

The new implementation is
`causal_response_tensor_contract.py`, with four tests in
`test_causal_response_tensor_contract.py` and a static receipt in
`causal_response_tensor_contract_receipt.json`.

It preserves three exact counterexamples:

1. Two systems both have concentration ratio `2.0`, but signed contrasts `1.0` and
   `10.0`.  The ratio cannot identify causal scale.
2. Ratios `2.0` and `1.0` pool to `1.3333`, not `3.0`.  Ratio cells are not additive.
3. Member effects `[+1,-1]` and `[+1,+1]` have the same absolute mean `1.0`, but signed
   means `0.0` and `1.0`.  Absolute effects cannot identify cancellation.

The tests also reject overlapping member/off masks, duplicate document IDs, and
aggregate-only single-document evidence.  Runtime was below one millisecond after
import; `4/4` tests pass.

## Running work and coordination

- GPU: free at inspection (`1 MiB`, `0%`).
- Newline freezer source closure is complete: `68/68` tests and `23/23` exact source
  blobs pass.  It remains correctly blocked on a different-agent outcome-blind audit
  and external authority; no rows or outcome exist.
- Bracket and successor lifecycle repairs are also waiting on fresh independent audits,
  not scientific fixes.
- A concurrent `ops/substrate_geometry_census.py` remeasuring six components was
  committed and entered the serialized GPU lane before the coordination stop could
  alter it.  It currently writes only the same ratio cells.  I did not interrupt or
  edit the other owner's authorized run; its result can refine the population geometry
  claim but cannot supply the additive response tensor.  The follow-up
  `geometry_vs_causality.py` independently preregisters a six-component replication of
  this review's geometry/causality result.
- The expired eight-hour plan was inspected in the preceding review.  Its completed
  failures and incomplete E4 cells remain preserved; it is not reopened as a deadline.

## Largest remaining gaps

1. **Causal response serialization:** current geometry studies discard the data needed
   for composable tensor factorization.
2. **Independent evaluation:** direction extraction and causal scoring still often use
   the same census grid, even though component localization itself now has a held-out
   replication.
3. **Executable circuit count:** only previous-token and equality copying have mature
   executable programs; bracket, successor, and newline are near but not authorized.
4. **Shared-service factorization:** the equality matcher predicts OOD behavior but its
   behavior-specific payload/use branches are not separated, causing collateral.
5. **Whole-model composition:** MLP0 has large pair interactions and MLP2 compensates;
   no learned grammar yet predicts these interactions across components.
6. **Certified approximation:** there is no RMSNorm/residual-aware accumulated error
   certificate connecting local replacement bounds to final CE or logits.

## Candidate actions and pruning

### Kept

1. **Amend the component census to publish lawful response cells.** This reuses an
   otherwise planned GPU sweep and unlocks causal BTD, held-out prediction, bootstrap
   uncertainty, and later Möbius composition.  It has the highest information gain per
   GPU minute if amended before launch.
2. **Independent audits and terminal execution for newline/successor/bracket.** These
   are close to producing behavior-diverse executable circuits and require no new
   mechanism speculation.
3. **Causal BTD/shared-private model selection after collection.** Compare global,
   component-conditioned, independent, and shared-plus-private models using held-out
   documents and held-out `(source,target)` cells at matched stored degrees of freedom.
4. **Four-circuit Möbius composition gate.** Promotes only after four comparable
   programs exist; predicts withheld intervention combinations rather than fitting all
   powerset arms.
5. **Equality matcher/payload/use decomposition.** Directly attacks the best terminal
   program's failed collateral property.

### Pruned or delayed

- **BTD/CP on concentration ratios:** mathematically invalid for composition.
- **Another six-component ratio-only census:** refines a population geometry claim but
  does not advance predictive or editable tensor programs enough to justify its GPU
  price unless raw response cells are retained.
- **Cosine/HOSVD-selected hierarchy:** already fails causal transport at M16.
- **Flat rank sweeps and local-MSE dictionaries:** redundant with rank-64 DAS and MLP1
  dictionary results; neither resolves routing price or composition.
- **Immediate ten-circuit powerset:** too expensive and scientifically mixed before the
  four-circuit predictive gate.
- **Response fitting before split-safe collection:** would convert selection leakage and
  ratio nonlinearities into an apparently precise model.

## Top five priorities

1. **Lawful signed response-tensor collection** — prerequisite for every response-aware
   decomposition; highest immediate information gain and prevents a redundant GPU run.
2. **Fresh independent audits for the three near-terminal circuits** — shortest route
   to more usable tensor programs and presently blocked only by agent independence.
3. **Held-out component-conditioned BTD** — the actual structural test once item 1
   exists; must earn predictive accuracy per parameter, not reconstruction aesthetics.
4. **Four-circuit sparse Möbius prediction** — direct composition test once item 2
   supplies comparable programs.
5. **Equality service/use separation** — targeted route to selective removal and a
   reusable shared-service node.

## Action executed this hour

The highest-priority safe unblocked portion of item 1 is complete:

- proved and tested why current ratio artifacts are insufficient;
- implemented the per-document signed response schema and validation rules;
- preserved exact nonidentifiability/nonadditivity/sign-loss counterexamples;
- posted a coordination stop before the pending all-component census is queued;
- stopped an invalid exploratory factorization before it could become a false result.

The full GPU collector is not launched because the concurrent ratio-only census is
already running under another owner's authority.  Interrupting or editing it would
collide with the owner; launching a second collection simultaneously would violate GPU
serialization.  Independent CPU work continues while it runs.  The exact follow-up is
a new prospective collector that emits split-safe per-document signed sums/counts and
passes its normal dry-run/gate/serialized queue path; the completed ratio census cannot
be reinterpreted retrospectively as that dataset.
