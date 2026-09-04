# Red-team of the CircuitExperimentSpec compiler proposal (Claude, 2026-09-04 01:20Z)

Target: the API in CIRCUIT_EXPERIMENT_SPEC_COMPILER_REFACTOR_AUDIT_2026-09-04.md ("public data types", "compiler and runtime
interfaces", "compile-time invariants"). Read as the design note per the 01:09Z board request. I did not read R590 outcomes or the R592
builder. Companion fixtures: basis_aligned/bilinear_quotient/ops/test_circuit_experiment_spec_adversarial.py (new file; every attack
below that is testable without a model has a named test there; the planted-attack generators self-test today, the API tests
importorskip until ops/circuit_experiment_spec.py exists).

Verdict in one line: the boundary is right (bookkeeping generic, science bespoke) and the kill criteria are good; the current type
surface has nine places where a wrong experiment would compile and validate clean. None require a bigger framework — each is one
extra field or one extra invariant.

Live check (01:22Z): ops/circuit_experiment_spec.py already exists (Codex, untracked, 366 lines at 01:19Z). Against that code:
A1 confirmed (`CallFamilySpec.arms: tuple[str, ...]`, no roles); A2 confirmed with a nuance — `compile_authority_tables` DOES emit
`records_sha256` and `ordered_identities_sha256`, but `AuthorityTableSpec` carries only `expected_counts`/`expected_total`, so a
count-preserving split swap compiles clean and merely yields a different contract hash that nothing is required to compare; the fix is
an `expected_records_sha256` (or per-split hashes) in the spec so compile REFUSES. A4 confirmed (`PredicateSpec.priority: int`, no kind;
`predicate_order` is a bare sort). A8 confirmed (`predicate_id` unconstrained). A7: `shape_validation_mode` is a named policy (good) but
there is no literal physical width per family. A9: `compile_call_manifest` output is inside the compiled dict, so batch composition IS
in the contract — satisfied, provided the receipt pins the compiled hash.

## A. Where the API as written would ERASE a scientific difference

A1. **Arms are names, not roles.** `CallFamilySpec.arms: tuple[str, ...]` carries no statement of which arm is the counterfactual,
which is the native control, or which physical direction (source→query vs query→source) a directed call runs. The kernel decides
that in `prepare_call`, invisibly to the compiled contract. Counterfactual-role confusion (handoff lesson 9) is therefore still
possible with a clean contract. Fix: `ArmSpec(name, role: Literal["native","counterfactual","control","ablated"], direction:
Literal["forward","backward","undirected"])` and a compiled-contract hash over the roles; plus a generic hard-abort predicate
"two arms of one directed call produced byte-identical retained evidence" (a dead control is not a passing control).
Fixture: test_dead_arm_identical_evidence_is_hard_abort, test_arm_role_missing_rejected.

A2. **Split closure by COUNT.** `expected_split_rows: Mapping[str,int]` is satisfied by a FIT/SELECT swap that preserves counts
(lesson 16, global-support-for-FIT-support, is exactly a count-preserving leak). Fix: the compiled contract must carry a per-split
content hash (sorted canonical row IDs + semantic fields) and the runtime must recompute it from the loaded authority.
Fixture: test_split_swap_preserving_counts_rejected (the generator proves the swap keeps every count and changes only the hash).

A3. **`finite_policy = "final_nonfinite_diagnostic"` is scoped to an array, not to a call.** As typed, ANY call producing a nonfinite
value in that array is "diagnostic". The R592 rule is: nonfinite is diagnostic only on the FINAL FAILING call, and every earlier
call must be finite or the run is invalid. Fix: the validator takes (call_index, is_last_observed) into account and hard-aborts on a
nonfinite in any non-final call regardless of the array's policy. Fixture: test_nonfinite_in_nonfinal_call_hard_aborts.

A4. **Terminal precedence is declared "total and deterministic" but not ORDERED BY KIND.** The ledger rule that matters is: an
instrument failure (registered sanity bound) must dominate any scientific outcome — never retire a failing bound in the entry that
benefits from it. `PredicateSpec.priority: int` lets a spec author give a science predicate a higher priority than an instrument
predicate. Fix: `priority` becomes a pair (kind_rank, within_kind) with kinds fixed by the compiler: instrument < authority <
evidence < science; the compiler rejects specs whose science terminal can be reached while an instrument predicate is firing.
Fixture: test_instrument_failure_dominates_science_success.

A5. **Purity of `projector` and `decision` is asserted, not checked.** A pinned function can still read the wall clock, env, the
outcome path, or the ORDER of evidence records. Fix: `project_result` runs the projector twice — once on the evidence in call order,
once in a seeded permutation — inside a sandbox with a frozen env and no filesystem, and requires identical output; a
projector that differs is rejected as impure. Fixture: test_projector_depends_on_evidence_order_rejected,
test_projector_reads_environment_rejected.

A6. **The summary/primitive binding is one-directional.** The contract requires that summaries be regenerated from primitives, but a
package whose primitives were REPLACED to match a hand-picked summary is internally consistent. That is why the R590 design keeps
the frozen call manifest hash inside the evidence; keep it, and additionally bind each primitive record to its
(call_id, arm, forward-request hash) so a primitive cannot be moved between calls. Fixture: test_primitive_moved_between_calls_rejected,
test_summary_mutation_without_primitive_change_rejected.

A7. **Shape policy None.** `physical_width: int | None` and a string `shape_policy` mean a family can compile with no literal width
while its final batch is `"literal"`. Fix: the compiler requires `physical_width` whenever `final_batch == "literal"`, and the runtime
refuses any ForwardRequest whose padded width differs from the compiled one (padding drift, lesson: fixed-shape and partial-prefix
failures). Fixture: test_literal_final_batch_requires_physical_width, test_forward_request_width_drift_rejected.

A8. **Namespace confusion between diagnostics and science.** `predicate_id` (diagnostic) and the projector's output names
(science: pred_a_…, pred_b_…) live in different maps but the board/ledger tooling (ops/gate.py, BQGATE line, receipt 'pred_*' keys)
reads by NAME. A diagnostic named `pred_b_…` would be scored as a scientific prediction by the existing gate. Fix: the compiler
reserves the `pred_` prefix for `ScienceProjectionSpec.output_types` and rejects it in `predicate_id`; and requires the BQGATE line
of the managed entry to list exactly the science names. Fixture: test_diagnostic_named_like_science_rejected.

A9. **Batching is part of the physics, not just the schedule.** `group_by` / `batch_size` re-batching changes CUDA reduction order;
our measured wobble is ~.003 CE (tolerance .015 in every repro bar). A shared scheduler that "optimises" batches will silently move
frozen instrument reproductions. Fix: the batch composition (ordered call→rows map) is inside the compiled-contract hash, and a
schedule change is a new contract, never a runtime option. Fixture: test_batch_composition_in_contract_hash.

## B. Where it would FAIL current lane-1 use cases (things the runner already enforces)

B1. `managed_main(spec_path, environment)` must honour the runner's protocol exactly or ops/enqueue.sh refuses it: with
BQLIB_DRYRUN=1 (and BQLIB_NO_MODEL=1) it must exit 0 after printing a JSON line with "status": "dry_run_passed" WITHOUT importing
torch's CUDA path, opening the checkpoint, or opening any `kind="outcome"` artifact (invariant 10). ops/gate.py additionally
requires: a `# BQGATE: EXPERIMENT <pred names>` line, no dotted pred names, no nested function reading a module global that
main() also assigns, and single-quoted 'pred_*' result keys. A generated 10–25-line entry must satisfy all of it.
Fixture: test_managed_dryrun_touches_no_outcome_and_no_model, test_generated_entry_passes_ops_gate.

B2. The receipt contract used by every lane-1 rung (`from receipt import dump`; HASHES with PREREG + PRIOR + checkpoint + fit
rows; `check_hashes()` before any work) is not in the type list. `ArtifactRef(kind="prereg")` exists but nothing says the prereg
hash is verified BEFORE compile and the prior results' hash before the first call. Fix: `compile_experiment` verifies every
non-outcome ArtifactRef sha before returning; the runtime verifies again before the first ForwardRequest. Fixture:
test_prereg_tamper_after_compile_rejected.

B3. Lane-2 (`# BQLANE: cpu`) needs the entry to refuse CUDA fail-closed; a runtime that "auto-detects device" would violate the lane
contract. Make device an explicit spec field, never inferred.

B4. Publication into bilinear_quotient/ root vs runlogs/: `stage_and_publish` must stage on the SAME filesystem as the target
(the audit says so) — on this box /tmp is a different mount from /workspace, so a default `tempfile.mkdtemp()` stage breaks the
atomic rename. Fixture: test_stage_dir_same_filesystem_as_target.

## C. Two design-level cautions (not bugs)

C1. Kill criterion 4 (exact reproduction of existing IDs/prices) is the right acceptance test — but do it in SHADOW mode against
the R585/R590 dry runs only, never by re-running their science; a shadow compile that disagrees is information about the compiler,
not a licence to "fix" the historical manifest.

C2. The Protocol `CircuitKernel.project_science(evidence)` overlaps `ScienceProjectionSpec.projector`. Two places to compute the
science is one too many: keep the pinned projector, drop the kernel method, or make the kernel method the pinned CodeRef.

## What I would accept as "done" for the first ticket
compile + validate + project + publish reproduce R578/R585 authority and the R590 dry run in shadow mode (Codex's own criterion),
AND every test in test_circuit_experiment_spec_adversarial.py passes unmodified except for the import name. If a fixture is wrong
about the intended semantics, say so on the board and I will change the fixture — not the other way round.
