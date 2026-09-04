#!/usr/bin/env python3
"""CPU-only Task 14 adapter for the reusable 55-site causal screen.

The adapter reuses the frozen Task 14 rows without changing a prompt, token,
answer, or donor.  A1 and A2 change grammatical subject number through two
syntactic constructions.  P changes noun identity while preserving number.  C
changes the nearest attractor's number while the coordinated subject and correct
plural verb remain fixed.  The generic scorer therefore treats C as an active
same-answer negative control: a selective subject-number state should move A1/A2
but should not move P or C.

This is a cheap full-state residual/module-output screen at the final prediction
position.  It is not DAS, a learned rotation, a rank search, or a new behavior
discovery.  The module imports no model and accesses no GPU or queue.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import circuit_battery_integration_contract as battery
import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_spec as screen


ROOT = Path(__file__).resolve().parent.parent
TASK_ID = "subject_verb.number_agreement"
SCHEMA = "circuit_fast_screen_task14_adapter_v1"
RELATION = "extension"

AUTHORITY_PATH = ROOT / "ops/circuit_battery_task14_agreement_fit_authority.json"
PARTITION_PATH = ROOT / "ops/circuit_battery_task14_fit_localization_partition_v2.json"
DONORS_PATH = ROOT / "ops/circuit_battery_task14_fit_localization_donors_v2.json"
CAPABILITY_RESULT_PATH = ROOT / "circuit_battery_task14_capability_fit_v1_results.json"
TASK_DOSSIER_PATH = ROOT / "circuits/task_subject_verb_number_agreement.json"
CAPABILITY_EVIDENCE_PATH = ROOT / "circuit_battery_task14_capability_fit_v1_evidence/calls"

EXPECTED_SOURCE_SHA256 = {
    "authority": "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f",
    "partition": "1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3",
    "donors": "ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a",
    "capability_result": "4239a25df47602dc07fce8602328f555a6bebc237f9dd897f34e812cf69dba12",
    "task_dossier": "6146cca76807a03ba0219204b87f7bcba2ba95449c3faba88e6355a2b81d24bf",
}

ANSWER_TOKEN_IDS = {" is": 318, " are": 389}

SOURCE_PATHS = {
    "authority": AUTHORITY_PATH,
    "partition": PARTITION_PATH,
    "donors": DONORS_PATH,
    "capability_result": CAPABILITY_RESULT_PATH,
    "task_dossier": TASK_DOSSIER_PATH,
}


class Task14AdapterError(ValueError):
    """The frozen Task 14 semantics cannot satisfy the generic screen."""


@dataclass(frozen=True)
class CapabilityCell:
    family: str
    cell_id: str
    row_count: int
    base_correct: int
    donor_correct: int
    base_accuracy: float
    donor_accuracy: float
    minimum_accuracy: float
    numerical_bar_passed: bool


@dataclass(frozen=True)
class CompatibilityReport:
    schema: str
    task_id: str
    relation: str
    authority_rows: int
    groups: int
    partition_records: int
    donor_records: int
    residual_sites: int
    module_sites: int
    total_sites: int
    capability_correct: int
    capability_expected: int
    capability_errors: int
    all_numerical_cell_bars_pass: bool
    generic_semantics_compatible: bool
    c_control_semantics: str
    v2_answer_changing_c_donors_are_target_related: bool
    capability_cells: tuple[CapabilityCell, ...]


TASK_SPEC = battery.BatteryTaskSpec(
    task_id=TASK_ID,
    generator_role="reuse_frozen_task14_fit_rows",
    answer_role="score_is_versus_are_at_final_prediction_position",
    transforms=(
        battery.TransformSpec(
            "A1", "subject_head_number_prepositional_phrase", True, "toward_donor"
        ),
        battery.TransformSpec(
            "A2", "subject_head_number_relative_clause", True, "toward_donor"
        ),
        battery.TransformSpec(
            "P", "nearest_attractor_identity_same_number", False, "invariant"
        ),
        battery.TransformSpec(
            "C", "nearest_attractor_number_coordinated_plural_subject",
            False, "registered_active",
        ),
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if type(value) is not dict:
        raise Task14AdapterError(f"source is not a JSON object: {path}")
    return value


def load_sources() -> dict[str, dict[str, Any]]:
    """Load and hash-check every immutable artifact used by this audit."""
    output: dict[str, dict[str, Any]] = {}
    for name, path in SOURCE_PATHS.items():
        observed = _sha256(path)
        expected = EXPECTED_SOURCE_SHA256[name]
        if observed != expected:
            raise Task14AdapterError(
                f"immutable {name} changed: expected={expected}, observed={observed}"
            )
        output[name] = _load_json(path)
    return output


def _cell_id(row: Mapping[str, object]) -> str:
    family = str(row["transform_id"])
    template = str(row["base_template_id"])
    if family in {"A1", "A2"}:
        direction = f"{row['base_subject_number']}_to_{row['donor_subject_number']}"
    elif family == "P":
        if row["base_subject_number"] != row["donor_subject_number"]:
            raise Task14AdapterError("P changes subject state")
        direction = f"subject_{row['base_subject_number']}"
    elif family == "C":
        if row["base_subject_number"] != row["donor_subject_number"]:
            raise Task14AdapterError("frozen C changes complete-subject state")
        base = "plural" if row["base_attractor_plural"] else "singular"
        donor = "plural" if row["donor_attractor_plural"] else "singular"
        direction = f"attractor_{base}_to_{donor}"
    else:
        raise Task14AdapterError(f"unknown family: {family}")
    return f"{family}/{template}/{direction}"


def adapted_rows() -> list[dict[str, Any]]:
    """Return interface-enriched copies; source prompts and donors stay unchanged."""
    sources = load_sources()
    authority = sources["authority"]
    if authority.get("task_id") != TASK_ID or authority.get("split") != "FIT":
        raise Task14AdapterError("authority identity changed")
    rows = authority.get("rows")
    if type(rows) is not list or len(rows) != 128:
        raise Task14AdapterError("authority must contain exactly 128 FIT rows")
    output = []
    for original in rows:
        if type(original) is not dict:
            raise Task14AdapterError("authority row is not an object")
        row = dict(original)
        if row.get("base_answer_id") != ANSWER_TOKEN_IDS.get(str(row.get("base_answer"))):
            raise Task14AdapterError("base answer token differs from frozen is/are IDs")
        if row.get("donor_answer_id") != ANSWER_TOKEN_IDS.get(str(row.get("donor_answer"))):
            raise Task14AdapterError("donor answer token differs from frozen is/are IDs")
        row["base_foil_id"] = ANSWER_TOKEN_IDS.get(str(row.get("base_foil")))
        row["donor_foil_id"] = ANSWER_TOKEN_IDS.get(str(row.get("donor_foil")))
        if type(row["base_foil_id"]) is not int or type(row["donor_foil_id"]) is not int:
            raise Task14AdapterError("foil is outside the frozen is/are vocabulary")
        if row.get("base_prediction_position") != len(row.get("base_ids", ())) - 1 \
                or row.get("donor_prediction_position") != len(row.get("donor_ids", ())) - 1:
            raise Task14AdapterError("frozen prediction position is not the final input token")
        row["base_semantic_position"] = row["base_prediction_position"]
        row["donor_semantic_position"] = row["donor_prediction_position"]
        row["capability_cell_id"] = _cell_id(row)
        output.append(row)
    battery.validate_rows(TASK_SPEC, output, required_phases=("FIT",))
    return output


def authority_sha256() -> str:
    """Digest of the exact immutable rows plus the adapter's cell labels."""
    return battery.validate_rows(TASK_SPEC, adapted_rows(), required_phases=("FIT",))


def validate_rows(rows: Sequence[Mapping[str, object]]) -> str:
    """Require byte-for-byte semantic equality to the deterministic frozen adapter."""
    materialized = [dict(row) for row in rows]
    expected = adapted_rows()
    if canonical_sha256(materialized) != canonical_sha256(expected):
        raise Task14AdapterError("rows differ from the deterministic Task 14 adapter")
    return battery.validate_rows(TASK_SPEC, materialized, required_phases=("FIT",))


def _validate_evidence_files(result: Mapping[str, object]) -> None:
    evidence = result.get("evidence_files")
    if type(evidence) is not list or len(evidence) != 24:
        raise Task14AdapterError("capability result evidence manifest changed")
    root = CAPABILITY_RESULT_PATH.parent / "circuit_battery_task14_capability_fit_v1_evidence"
    for record in evidence:
        if type(record) is not dict:
            raise Task14AdapterError("evidence record is not an object")
        path = root / str(record["path"])
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise Task14AdapterError(f"capability evidence missing or wrong size: {path}")
        if _sha256(path) != record["sha256"]:
            raise Task14AdapterError(f"capability evidence hash changed: {path}")


def capability_cells() -> tuple[CapabilityCell, ...]:
    """Recompute ordered-cell capability from the frozen per-row logits."""
    sources = load_sources()
    result = sources["capability_result"]
    _validate_evidence_files(result)
    rows = adapted_rows()
    by_id = {str(row["row_id"]): row for row in rows}
    observed: dict[tuple[str, str], dict[str, list[bool]]] = {}

    call_dirs = sorted(path for path in CAPABILITY_EVIDENCE_PATH.iterdir() if path.is_dir())
    if len(call_dirs) != 8:
        raise Task14AdapterError("expected exactly eight frozen native capability calls")
    for call_dir in call_dirs:
        call = _load_json(call_dir / "call.json")
        pieces = str(call["call_family"]).split("_")
        if len(pieces) < 2 or pieces[-2] not in {"base", "donor"}:
            raise Task14AdapterError("native capability call family changed")
        side, family = pieces[-2], pieces[-1]
        answers = np.load(call_dir / "answer_logit.npy", allow_pickle=False)
        foils = np.load(call_dir / "foil_logit.npy", allow_pickle=False)
        row_ids = call["row_ids"]
        if len(row_ids) != len(answers) or answers.shape != foils.shape:
            raise Task14AdapterError("capability array shape differs from call rows")
        for row_id, answer, foil in zip(row_ids, answers, foils):
            row = by_id[str(row_id)]
            if row["transform_id"] != family:
                raise Task14AdapterError("capability call family differs from authority")
            key = (family, str(row["capability_cell_id"]))
            cell = observed.setdefault(key, {"base": [], "donor": []})
            cell[side].append(bool(float(answer) > float(foil)))

    thresholds = {
        "A1": kernel.MIN_A1_CAPABILITY_ACCURACY,
        "A2": kernel.MIN_A2_CAPABILITY_ACCURACY,
        "P": kernel.MIN_P_CAPABILITY_ACCURACY,
        "C": kernel.MIN_C_CAPABILITY_ACCURACY,
    }
    output = []
    for (family, cell_id), sides in sorted(observed.items()):
        if len(sides["base"]) != len(sides["donor"]) or not sides["base"]:
            raise Task14AdapterError("ordered capability cell coverage is incomplete")
        base_correct, donor_correct = sum(sides["base"]), sum(sides["donor"])
        count = len(sides["base"])
        threshold = thresholds[family]
        base_accuracy, donor_accuracy = base_correct / count, donor_correct / count
        output.append(CapabilityCell(
            family=family,
            cell_id=cell_id,
            row_count=count,
            base_correct=base_correct,
            donor_correct=donor_correct,
            base_accuracy=base_accuracy,
            donor_accuracy=donor_accuracy,
            minimum_accuracy=threshold,
            numerical_bar_passed=(
                base_accuracy >= threshold and donor_accuracy >= threshold
            ),
        ))
    return tuple(output)


def compatibility_report() -> CompatibilityReport:
    sources = load_sources()
    rows = adapted_rows()
    partition = sources["partition"]
    donors = sources["donors"]
    result = sources["capability_result"]
    cells = capability_cells()

    groups = {str(row["group_id"]) for row in rows}
    partition_records = partition.get("records")
    donor_records = donors.get("records")
    if type(partition_records) is not list or len(partition_records) != 32:
        raise Task14AdapterError("partition must contain 32 group records")
    if {str(record["group_id"]) for record in partition_records} != groups:
        raise Task14AdapterError("partition group coverage differs from authority")
    if type(donor_records) is not list or len(donor_records) != 1088:
        raise Task14AdapterError("donor authority must contain 1088 records")

    answer_changing_c = [
        record for record in donor_records
        if record["family"] == "C"
        and record["expected_relation"] == "opposite_subject_toward_donor"
    ]
    c_related = bool(answer_changing_c) and all(
        record["source_contract"] == "v2_complete_subject_Q"
        and record["arm"] == "C_to_ordinary_singular"
        for record in answer_changing_c
    )
    frozen_c = [row for row in rows if row["transform_id"] == "C"]
    if len(frozen_c) != 32 or any(row["answer_changes"] for row in frozen_c):
        raise Task14AdapterError("frozen C no longer has its registered invariance meaning")
    if any(row["base_answer_id"] != row["donor_answer_id"] for row in frozen_c):
        raise Task14AdapterError("frozen C answer identity changed")

    projection = result["decision"]["projection"]
    correct = round(
        float(projection["base_accuracy"]) * 128
        + float(projection["donor_accuracy"]) * 128
    )
    cell_correct = sum(cell.base_correct + cell.donor_correct for cell in cells)
    if correct != 249 or cell_correct != correct:
        raise Task14AdapterError("capability evidence no longer reproduces 249/256")

    return CompatibilityReport(
        schema=SCHEMA,
        task_id=TASK_ID,
        relation=RELATION,
        authority_rows=len(rows),
        groups=len(groups),
        partition_records=len(partition_records),
        donor_records=len(donor_records),
        residual_sites=len(screen.RESIDUAL_SITE_IDS),
        module_sites=len(screen.MODULE_SITE_IDS),
        total_sites=len(screen.CEILING_SITE_IDS),
        capability_correct=correct,
        capability_expected=256,
        capability_errors=256 - correct,
        all_numerical_cell_bars_pass=all(cell.numerical_bar_passed for cell in cells),
        generic_semantics_compatible=True,
        c_control_semantics="same_answer_active_negative_control",
        v2_answer_changing_c_donors_are_target_related=c_related,
        capability_cells=cells,
    )


def audit_sha256() -> str:
    """Stable digest of the complete compatibility finding."""
    return canonical_sha256(asdict(compatibility_report()))


def build_rows(task_id: str = TASK_ID) -> list[dict[str, Any]]:
    """Return exact frozen rows after all source and compatibility checks."""
    if task_id != TASK_ID:
        raise KeyError(task_id)
    report = compatibility_report()
    if not report.generic_semantics_compatible:
        raise Task14AdapterError("Task 14 is incompatible with the generic screen")
    return adapted_rows()


def build_spec(rows: list[dict[str, Any]] | None = None) -> screen.CircuitFastScreenSpec:
    """Build the exact prospective full-state screen specification."""
    materialized = build_rows() if rows is None else [dict(row) for row in rows]
    observed_authority = validate_rows(materialized)
    if observed_authority != authority_sha256():
        raise Task14AdapterError("adapted Task 14 authority changed")
    return screen.CircuitFastScreenSpec(
        experiment_id="fast-screen-task14-subject-verb-agreement-full-state-v1",
        hypothesis=screen.CandidateHypothesis(
            behavior=TASK_ID,
            answer_score=screen.ANSWER_SCORE,
            information_read=(
                "complete grammatical subject number rather than nearest-noun number"
            ),
            proposed_operation=(
                "carry subject number across prepositional phrases and relative clauses"
            ),
            proposed_write="evidence for the next-token choice between is and are",
            candidate_sites=screen.CEILING_SITE_IDS,
            alternative_explanation=(
                "nearest-noun number, construction-specific state, or generic output state"
            ),
            circuit_prediction=(
                "one site transfers both answer-changing subject-number families while "
                "remaining small for noun-identity and attractor-number controls"
            ),
            opposing_null_prediction=(
                "native capability fails or no common selective full-state site exists"
            ),
        ),
        task=TASK_SPEC,
        authority_sha256=observed_authority,
        expected_fit_rows=len(materialized),
        batch_size=32,
        semantic_position=screen.SemanticPositionSpec(
            role="final input token immediately before the predicted verb",
            recipient_field="base_semantic_position",
            donor_field="donor_semantic_position",
        ),
        fields=screen.AuthorityFieldSpec(),
        bars=kernel.FIXED_BARS,
        declared_max_price=battery.ExactPhasePrice(
            phase="FIT",
            forward_calls=264,
            example_evaluations=8448,
            backward_calls=0,
            model_updates=0,
            evidence_bytes=67584,
        ),
    )
