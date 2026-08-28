# Prospective amendment: typed final action capability

Date frozen: 2026-08-28

Status: source/interface amendment only. No fit, validation, or final role has been
materialized or opened under this amendment, and it grants no execution authority.

## Why this amendment is necessary

The original implementation stopped the shared trace identity at validation and left
the final scientific callback unspecified. Reusing a validation identity on final
rows would spend the wrong role while producing mechanically valid reductions. Giving
the callback raw rows, logits, residuals, or a model would defeat the source-closed
terminal owner. This amendment introduces a distinct final phase and the typed
action/reduction boundary that must exist before the final role can be opened.

## Final identity

Every fitted-program final batch has:

- role `early_mlp_suffix_transport_v1_final` and phase `final`;
- the final-role tensor SHA-256 in the historical `fit_role_tensor_sha256` field;
- canonical row indices `4b,...,4b+3` for batch ordinal `b`;
- epoch zero and optimizer step equal to batch ordinal;
- one frozen program snapshot and true teacher-mapping identity;
- physical state `P/P/N` or `P/P/E` only.

Fit and validation remain `P/P/N`. Neither can authorize `P/P/E`, and a final context
cannot authorize a fit or validation identity.

## N and E backgrounds

Under deployed-MLP2 background N, a fitted L/R/S/T program executes P/P/N and returns
per-row CE, copy CE, and its registered primary statistic. L's primary statistic is
the frozen-denominator coordinate loss; R/S/T use exact O/O/N teacher KL. Native MLP0,
MLP1, and MLP2 calls on the student path remain zero.

Under exact-restored-MLP2 background E, the same fitted program executes P/P/E. Native
MLP0/1 calls remain poisoned and exactly one literal native MLP2 call is required. E
is CE-only in the preregistration: no OON teacher is constructed and no zero-valued KL
field may be inserted. A distinct `FinalCEBatchReductions` type enforces this absence.

## Action lattice and allowed observations

The final capability owns the canonical 34 arms by two backgrounds, 68 actions total,
in frozen order. Callers cannot select, skip, duplicate, or reorder actions. A failure
poisons the capability. Every action must use one common scored-support identity.

Only the following may cross the observed boundary:

- per-row scalar sums and counts for CE, registered N-background teacher KL/local
  error, copy CE, and nine inherited token-frequency bins;
- per-row teacher/student/error/dot inner products for registered MLP1-code and
  centered-logit intervention responses;
- per-row scalar norm-ratio sums/counts for every one of the 18 live consumers;
- tensor-free action, program, support, execution-closure, and ledger identities.

No role rows, target tokens, logits, codes, residual states, model/module/dispatcher
handles, multidimensional response tensors, CUDA tensors, graph-bearing tensors, or
storage aliases may escape. Consumer norm ratios are integrity diagnostics and cannot
select a program, amplitude, grammar, or scientific route.

## Current implementation boundary

The observed adapter presently implements true fitted-program L/R/S/T batches under
N and E. The complete 68-action executor remains NO-GO until source-closed backends
also exist for QQ/LL/RR/singleton/removal/baseline/shuffle/null arms, finite physical
code edits and their unedited pairs, nine-bin aggregation, consumer norm aggregation,
canonical program-bank routing, and complete bundle assembly.

This partial backend cannot mint a full final action capability and cannot be used to
open the role. Its purpose is to close and test the shared final identity, exact N/E
semantics, and permitted batch reductions before the larger action router is written.
