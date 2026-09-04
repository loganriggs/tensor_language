# Circuit battery → experiment framework integration contract

Date: 2026-09-04  
Status: CPU-only integration boundary; no model execution or scientific claim  
Code: `ops/circuit_battery_integration_contract.py`

## Outcome

The reusable battery should be a thin client of the approved circuit framework, not a second framework. A task
generator owns prompts and semantic counterfactuals. A model executor owns hooks and the scientific computation. The
existing modules own everything between them:

- `circuit_experiment_spec.py`: typed artifacts, authority hashes, arm roles, call manifests, arrays, predicates, and
  literal forward shapes;
- `circuit_artifact_package.py`: primitive-evidence validation, evidence-derived decisions, and atomic receipt-last
  publication;
- `circuit_managed_entry.py`: hash-first capture and execution of the exact verified producer bytes.

The new integration module adds only battery-specific types and invariants that those general modules should not know:
the four semantic transformations, generated-example grouping, phase receipts, exact phase price, and physical joint
arms. It contains no prompt generator, tokenizer, model import, hook, or scientific threshold.

## Typed task boundary

A `BatteryTaskSpec` declares:

```text
task_id, generator_role, answer_role
transforms = [
  A1: independently generated answer change, expected toward-donor effect,
  A2: a second independent answer change, expected toward-donor effect,
  P:  answer-preserving change, expected invariant effect,
  C:  active control with a registered task-specific effect,
]
joint_arms = [(physical joint arm ID, two or more singleton arm IDs)]
group_id_field, row_id_field
```

Every generated `group_id` must contain exactly one A1, A2, P, and C row in one split. Group IDs may never cross
splits. This is stronger than merely balancing four marginal row sets: the generated example is the statistical unit,
so a bootstrap draw samples a group and carries all four related rows with it. A task-specific control may move—the
registered A1/A2/P/C effect vector, not a universal zero-control assumption, defines selectivity.

`validate_rows()` returns the canonical row-table digest only after enforcing that panel. The generic scorer must save
row-level primitives with both `row_id` and `group_id`; its confidence intervals resample sorted group IDs, never row
endpoints. The SHA draw grammar and replicate count belong in the protocol authority, not in a task generator.

## Physical split closure

Each phase is a separate managed invocation and a separate atomic result package:

```text
FIT package --pass + frozen selection hash--> SELECT package
SELECT package --pass + same selection hash--> TEST package
TEST package --pass + same selection hash--> OOD package
```

`authorize_phase()` requires the exact ordered prior receipt prefix. A failed/invalid receipt, another task's receipt,
a missing phase, or a changed selection hash refuses the later phase. `phase_artifacts()` rejects any future-split
artifact from the current invocation. OOD is a distinct required phase after TEST, not a label for reused TEST rows.

This separation is necessary because `managed.dispatch()` intentionally captures every artifact in a real invocation
before importing the producer. Putting FIT, SELECT, and TEST rows in one `CircuitExperimentSpec` would therefore open
held-out bytes early even if later Python branches did not use them. Instead, the battery compiler creates one
phase-local `CircuitExperimentSpec` containing:

1. the protocol/preregistration and executable sources;
2. only the current split's frozen row authority (plus already-public prior receipts if needed);
3. only call families whose `split` equals the current phase;
4. the phase's arrays, instrument predicates, and scientific projection.

It then calls `compile_experiment(...)`, passes that exact spec to `managed.dispatch(...)`, validates retained primitive
evidence and its projection, and publishes via `stage_package(...)` / `publish_staged_package(...)`. The receipt is the
last visible file. TEST is not a flag inside a FIT process.

## Exact price

Every opened phase declares one `ExactPhasePrice` with five literal integer fields:

```text
forward_calls, example_evaluations, backward_calls, model_updates, evidence_bytes
```

`validate_price()` derives forward calls and example evaluations from the compiled call manifest; the latter is the
sum of `logical_batch_size`, not padded tensor width. The executor ledger must bind measured backward calls, updates,
and retained bytes, and `validate_price_receipt()` requires exact equality to the declaration. Updates are fixed at
zero. Report both calls and example evaluations: R590's 457 calls and 20,212 examples demonstrate why either number
alone is ambiguous.

Conditional phase prices are recorded per phase, not as one misleading total. The campaign may additionally report
FIT, FIT+SELECT, FIT+SELECT+TEST, and full FIT+SELECT+TEST+OOD ceilings by exact sums of these phase records.

## Joint-arm evidence

A joint-reader claim must have its own typed arm in the existing `CallFamilySpec`; it is not `effect(reader8) +
effect(reader10)`. For every executed joint-arm row, retained primitive evidence includes:

```text
joint_arm_id, row_id, group_id, joint_call_id,
member_call_ids {member -> physical singleton call},
singleton_effects {member -> signed effect},
joint_effect, interaction
```

`validate_joint_arm_evidence()` checks exact manifest membership, row/group authority, identical row batches for joint
and singleton calls, complete coverage, and

```text
interaction = joint_effect - sum(singleton_effects)
```

The separately saved `joint_effect` drives concentration or sufficiency gates. Singleton sums and the interaction are
reported diagnostics. This directly prevents the R590 mistake of interpreting a singleton sum as a physically
executed joint intervention.

## Adapter required for the live task-bank draft

Claude's live `ops/circuit_battery_tasks.py` is owned by that lane and was not edited or imported by this work. Its
current rows are not yet valid inputs to this boundary:

- rows lack a shared `group_id`; A1/A2/P/C are independently sampled rather than emitted as one generated-example
  panel;
- the family field is named `family`, not the typed `transform_id`;
- RNG seeds use Python's process-randomized `hash(...)`, so row bytes are not reproducible across interpreters;
- `bank_digest()` reopens its mutable path instead of relying on bytes already captured by the managed adapter.

The executor must not paper over the first issue by assigning arbitrary groups after generation. The generator should
draw one base semantic coordinate per group, derive all four transformations from it, assign a canonical SHA-defined
group ID, and use SHA-derived integer seeds. Renaming `family` to `transform_id` is then a lossless adapter step.

The contract implements FIT/SELECT/TEST/OOD. Claude's current live draft implements only the first three and therefore
needs a separately generated, split-disjoint OOD authority before adoption; it must not silently reuse TEST rows or
receipts.

## Integration test and stopping rule

The focused model-free test constructs one synthetic task, compiles its FIT call manifest with the approved compiler,
checks managed artifact closure, binds literal price, and plants missing/cross-split groups, reselection, missing-OOD,
future-byte, invented-joint-call, duplicate joint execution, wrong-interaction, and incomplete-evidence failures. It
does not import the live task generator.

This is enough integration infrastructure. Stop adding generic types unless a real campaign task cannot express a
needed semantic transformation, physical arm, phase gate, or primitive-evidence join. The next useful work is to adapt
one existing circuit task to this boundary and compare its generated manifest/result reconstruction with the audited
bespoke result—not to expand the framework speculatively.
