#!/usr/bin/env python3
# BQLANE: cpu
"""Compile task14's capability-only FIT invocation without opening later splits.

This outcome-blind CPU module has no model-facing entry point. A future producer
must consume this exact reviewed contract through a separately authorized adapter.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import circuit_artifact_package as package
import circuit_battery_integration_contract as battery
import circuit_battery_task14 as task14
import circuit_experiment_spec as framework
import circuit_managed_entry as managed


EXPERIMENT_ID = "circuit-battery-task14-subject-verb-agreement-capability-fit-v1"
REPLACEMENT_REVIEW_COMMIT = "ea7efad782c088ba91a2ce338a9f740563c4e7c1"
TASK14_AUTHORITY_SHA256 = "1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1"
FIT_RECORDS_SHA256 = "3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1"
FIT_AUTHORITY_FILE_SHA256 = "e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f"
CALL_MANIFEST_SHA256 = "4b4da44c5090914f87d52e018bc9a8d18b74a202bdb82667283a9f1564682e0e"
METRIC_MANIFEST_SHA256 = "5da9f66829156e352afe087c75f92a7a6a37f06fe1ec5177efeffd9442609dcc"
SPEC_SHA256 = "9cad5272e1f49712ba218ad54d577770dbb244c5e18065a592910f841895ec40"
COMPILED_CONTRACT_SHA256 = "84f8e1cf85323dba94d13c7c716afef448b8621bff6b534c2025715420e86a82"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIT_AUTHORITY_PATH = (
    "basis_aligned/bilinear_quotient/ops/"
    "circuit_battery_task14_agreement_fit_authority.json"
)
PREREG_PATH = (
    "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK14_SUBJECT_VERB_AGREEMENT_CAPABILITY_FIT_PREREGISTRATION.md"
)
DESIGN_MEMO_PATH = (
    "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK14_SUBJECT_VERB_AGREEMENT_DESIGN_REPAIR_2026-09-04.md"
)
REVIEW_PATH = (
    "basis_aligned/polynomial_causal/"
    "TASK14_SUBJECT_VERB_AGREEMENT_REPLACEMENT_AUTHORITY_REVIEW_2026-09-04.md"
)

FROZEN_ARTIFACTS = (
    (
        "task14_generator",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_task14.py",
        "33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94",
        "source", "PROTOCOL",
    ),
    (
        "task14_generator_tests",
        "basis_aligned/bilinear_quotient/ops/test_circuit_battery_task14.py",
        "254fe3798efd8a4426f30e054fd8e5646a5bd6635df69815f376311ac2023694",
        "source", "PROTOCOL",
    ),
    (
        "battery_contract",
        "basis_aligned/bilinear_quotient/ops/circuit_battery_integration_contract.py",
        "b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e",
        "source", "PROTOCOL",
    ),
    (
        "experiment_spec",
        "basis_aligned/bilinear_quotient/ops/circuit_experiment_spec.py",
        "64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c",
        "source", "PROTOCOL",
    ),
    (
        "artifact_package",
        "basis_aligned/bilinear_quotient/ops/circuit_artifact_package.py",
        "6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc",
        "source", "PROTOCOL",
    ),
    (
        "managed_entry",
        "basis_aligned/bilinear_quotient/ops/circuit_managed_entry.py",
        "1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81",
        "source", "PROTOCOL",
    ),
    (
        "repaired_design_memo", DESIGN_MEMO_PATH,
        "3cb4556d1ad2c1564f2708028e5d624c4519fbc4d52a38cac27b9d10d8312f68",
        "prereg", "PROTOCOL",
    ),
    (
        "replacement_authority_review", REVIEW_PATH,
        "7249991dd727f6385d3269cce23b0e5f83c588bcef3488dce33ae19dfd223fd1",
        "prereg", "PROTOCOL",
    ),
    (
        "preregistration", PREREG_PATH,
        "06a9747b4707999e11637a45cf83588bfd9cb8671d6b3a25790518af62900f8b",
        "prereg", "PROTOCOL",
    ),
    (
        "fit_authority", FIT_AUTHORITY_PATH, FIT_AUTHORITY_FILE_SHA256,
        "authority", "FIT",
    ),
)

PRICE = battery.ExactPhasePrice(
    phase="FIT",
    forward_calls=8,
    example_evaluations=256,
    backward_calls=0,
    model_updates=0,
    evidence_bytes=2_048,
)

_PRIMITIVE_FIELDS = frozenset({
    "call_id", "row_id", "side", "transform_id", "incongruent",
    "answer_changes", "answer_logit", "foil_logit",
})
_PROJECTION_FIELDS = (
    "base_accuracy", "donor_accuracy", "cell_accuracies", "cell_mean_margins",
    "incongruent_accuracies", "incongruent_mean_margins", "capability_pass",
)


class CapabilityCompileError(ValueError):
    """The capability-only invocation is not the exact frozen FIT contract."""


def _artifact_refs() -> tuple[battery.PhasedArtifact, ...]:
    return tuple(
        battery.PhasedArtifact(
            framework.ArtifactRef(
                role=role, path=path, sha256=digest, kind=kind,
                executable=False, dryrun_access=True,
            ),
            visibility=visibility,
        )
        for role, path, digest, kind, visibility in FROZEN_ARTIFACTS
    )


def _call_family(side: str, transform: str) -> framework.CallFamilySpec:
    if side not in ("base", "donor") or transform not in ("A1", "A2", "P", "C"):
        raise CapabilityCompileError("native metric side or transform is invalid")
    arm = f"native_{side}_{transform}"
    return framework.CallFamilySpec(
        name=f"fit_native_{side}_{transform}",
        split="FIT",
        arms=(arm,),
        batch_size=32,
        call_kind="native_answer_foil_logits",
        guard="capability_only",
        call_id_template="{split}:" + side + ":" + transform + ":{batch}:{arm}",
        arm_specs=(framework.ArmSpec(arm, "native", "undirected"),),
        sequence_field=f"{side}_ids",
        filters=(("transform_id", (transform,)),),
        axis_order="arm_batch",
    )


def build_spec() -> framework.CircuitExperimentSpec:
    """Build the FIT-only typed spec from frozen artifact digests."""
    artifacts = battery.phase_artifacts("FIT", _artifact_refs())
    calls = tuple(
        _call_family(side, transform)
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    )
    spec = framework.CircuitExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        rung=14,
        artifacts=artifacts,
        phases=(framework.PhaseSpec(
            "FIT", opens_after=None, forbidden_splits=("SELECT", "TEST", "OOD")
        ),),
        authority_tables=(framework.AuthorityTableSpec(
            name="fit_rows",
            identity_fields=("row_id",),
            expected_records_sha256=FIT_RECORDS_SHA256,
            split_field="split",
            group_fields=("group_id",),
            expected_counts={"FIT": 128},
            expected_total=128,
        ),),
        calls=calls,
        arrays=(
            framework.ArraySpec(
                "answer_logit", ("native_answer_foil_logits",),
                "float32", ("batch",), True,
            ),
            framework.ArraySpec(
                "foil_logit", ("native_answer_foil_logits",),
                "float32", ("batch",), True,
            ),
        ),
        predicates=(
            framework.PredicateSpec(
                "metric_evidence_contract", "FIT", 0,
                "task14_metric_evidence", ("answer_logit", "foil_logit"),
                "hard_abort", "instrument",
            ),
            framework.PredicateSpec(
                "answer_relation_contract", "FIT", 1,
                "task14_answer_relation", ("answer_logit", "foil_logit"),
                "hard_abort", "authority",
            ),
            framework.PredicateSpec(
                "native_capability_gate", "FIT", 2,
                "task14_native_capability", ("answer_logit", "foil_logit"),
                "hard_abort", "science",
            ),
        ),
        science=framework.ScienceProjectionSpec(
            projector_role="task14_capability_projection",
            decision_role="capability_only_no_localization",
            allowed_terminals=("ok", "hard_abort"),
            output_types={
                "base_accuracy": "number",
                "donor_accuracy": "number",
                "cell_accuracies": "mapping",
                "cell_mean_margins": "mapping",
                "incongruent_accuracies": "mapping",
                "incongruent_mean_margins": "mapping",
                "capability_pass": "boolean",
            },
        ),
    )
    framework.validate_spec(spec)
    return spec


def _validate_fit_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    copied = [dict(row) for row in rows]
    if any(row.get("split") != "FIT" for row in copied):
        raise CapabilityCompileError("future split leaked into the FIT invocation")
    if framework.canonical_sha256(copied) != FIT_RECORDS_SHA256:
        raise CapabilityCompileError("frozen FIT authority changed")
    observed = battery.validate_rows(task14.TASK_SPEC, copied, required_phases=("FIT",))
    if observed != FIT_RECORDS_SHA256:
        raise CapabilityCompileError("task14 FIT authority digest changed at integration boundary")
    groups = {(str(row["split"]), str(row["group_id"])) for row in copied}
    if len(groups) != 32:
        raise CapabilityCompileError("task14 FIT group census changed")
    if {str(row["transform_id"]) for row in copied} != {"A1", "A2", "P", "C"}:
        raise CapabilityCompileError("task14 FIT transform census changed")
    if len({row[f"{side}_text"] for row in copied for side in ("base", "donor")}) != 256:
        raise CapabilityCompileError("task14 FIT row-side prompts are not all unique")
    for row in copied:
        transform = str(row["transform_id"])
        answer_changes = transform in ("A1", "A2")
        if row.get("answer_changes") is not answer_changes:
            raise CapabilityCompileError("task14 answer-change relation changed")
        if (row["base_answer_id"] != row["donor_answer_id"]) is not answer_changes:
            raise CapabilityCompileError("task14 answer-token direction changed")
        if row["base_answer"] not in (" is", " are") \
                or row["donor_answer"] not in (" is", " are"):
            raise CapabilityCompileError("task14 answer vocabulary changed")
        for side in ("base", "donor"):
            text = str(row[f"{side}_text"])
            answer = str(row[f"{side}_answer"])
            foil = str(row[f"{side}_foil"])
            ids, answer_id = task14._joint_encoding(text, answer)
            _, foil_id = task14._joint_encoding(text, foil)
            if ids != row[f"{side}_ids"] or answer_id != row[f"{side}_answer_id"]:
                raise CapabilityCompileError("task14 continuation tokenization changed")
            if {answer, foil} != {" is", " are"} or answer_id == foil_id:
                raise CapabilityCompileError("task14 answer/foil pair is degenerate")
            if row[f"{side}_prediction_position"] != len(ids) - 1:
                raise CapabilityCompileError("task14 prediction position changed")
        changed = [
            index for index, pair in enumerate(zip(row["base_ids"], row["donor_ids"]))
            if pair[0] != pair[1]
        ]
        expected_position = (
            row["base_head_positions"][0]
            if transform in ("A1", "A2") else row["base_attractor_positions"][-1]
        )
        if len(row["base_ids"]) != len(row["donor_ids"]) \
                or changed != [expected_position] \
                or changed != row["intervention_token_positions"] \
                or row["base_head_positions"] != row["donor_head_positions"] \
                or row["base_attractor_positions"] != row["donor_attractor_positions"]:
            raise CapabilityCompileError("task14 aligned intervention coordinates changed")
    return copied


def load_fit_authority_bytes(payload: bytes) -> list[dict[str, object]]:
    """Load the captured FIT artifact without calling any phase generator."""
    if hashlib.sha256(payload).hexdigest() != FIT_AUTHORITY_FILE_SHA256:
        raise CapabilityCompileError("frozen FIT authority artifact bytes changed")
    try:
        value = json.loads(payload)
        framework.canonical_json_bytes(value)
    except (UnicodeDecodeError, ValueError, TypeError) as error:
        raise CapabilityCompileError("FIT authority artifact is not strict JSON") from error
    if not isinstance(value, dict) or set(value) != {
        "schema", "task_id", "split", "task14_authority_sha256",
        "split_records_sha256", "groups", "rows",
    }:
        raise CapabilityCompileError("FIT authority envelope changed")
    if value["schema"] != "circuit_battery_task14_split_authority_v1" \
            or value["task_id"] != task14.TASK_ID \
            or value["split"] != "FIT" \
            or value["task14_authority_sha256"] != TASK14_AUTHORITY_SHA256 \
            or value["split_records_sha256"] != FIT_RECORDS_SHA256 \
            or value["groups"] != 32 \
            or not isinstance(value["rows"], list):
        raise CapabilityCompileError("FIT authority metadata changed")
    return _validate_fit_rows(value["rows"])


def _side_metric(row: Mapping[str, object], side: str) -> tuple[int, int, bool]:
    target = row[f"{side}_answer_id"]
    _, foil = task14._joint_encoding(str(row[f"{side}_text"]), str(row[f"{side}_foil"]))
    if type(target) is not int or target == foil:
        raise CapabilityCompileError("task14 answer/foil token IDs are invalid")
    incongruent = bool(
        row[f"{side}_head_plural"] != row[f"{side}_attractor_plural"]
    ) if row["transform_id"] in ("A1", "A2", "P") else False
    return target, foil, incongruent


def _metric_manifest(
    rows: Sequence[Mapping[str, object]], calls: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_id = {str(row["row_id"]): row for row in rows}
    output: list[dict[str, object]] = []
    for call in calls:
        parts = str(call["arm"]).split("_")
        if len(parts) != 3 or parts[0] != "native" \
                or parts[1] not in ("base", "donor") \
                or parts[2] not in ("A1", "A2", "P", "C"):
            raise CapabilityCompileError("non-native arm entered capability manifest")
        side, transform = parts[1], parts[2]
        row_ids = [str(value) for value in call["row_ids"]]
        selected = [by_id[row_id] for row_id in row_ids]
        if any(row["transform_id"] != transform for row in selected):
            raise CapabilityCompileError("task14 call mixes transform families")
        metrics = [_side_metric(row, side) for row in selected]
        output.append({
            "call_id": call["call_id"],
            "side": side,
            "transform_id": transform,
            "row_ids": row_ids,
            "target_token_ids": [target for target, _, _ in metrics],
            "foil_token_ids": [foil for _, foil, _ in metrics],
            "prediction_positions": [row[f"{side}_prediction_position"] for row in selected],
            "incongruent": [flag for _, _, flag in metrics],
            "answer_changes": [bool(row["answer_changes"]) for row in selected],
            "metric": "answer_logit_minus_opposite_copula_foil_logit",
            "strict_correct_rule": "margin_gt_zero",
        })
    return output


def compile_fit_invocation(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compile exact FIT calls, metrics, and price from already captured FIT rows."""
    frozen_rows = _validate_fit_rows(rows)
    spec = build_spec()
    compiled = framework.compile_experiment(
        spec,
        authority_tables={"fit_rows": frozen_rows},
        call_source_records=frozen_rows,
    )
    metric_manifest = _metric_manifest(frozen_rows, compiled["call_manifest"])
    compiled.update({
        "capability_scope": "FIT_native_only_no_reader_localization_or_selection",
        "replacement_review_commit": REPLACEMENT_REVIEW_COMMIT,
        "task14_authority_sha256": TASK14_AUTHORITY_SHA256,
        "fit_authority_sha256": FIT_RECORDS_SHA256,
        "metric_manifest": metric_manifest,
        "metric_manifest_sha256": framework.canonical_sha256(metric_manifest),
        "literal_price": asdict(PRICE),
        "later_split_generation": False,
        "later_split_artifacts": [],
    })
    _validate_compiled_contract(compiled)
    framework.canonical_json_bytes(compiled)
    return compiled


def evaluate_metric_evidence(rows):
    """Pure predicate: exact primitive types and finite scalar logits."""
    return len(rows) == 256 and all(
        type(row) is dict
        and set(row) == {
            "call_id", "row_id", "side", "transform_id", "incongruent",
            "answer_changes", "answer_logit", "foil_logit",
        }
        and type(row["call_id"]) is str
        and type(row["row_id"]) is str
        and row["side"] in ("base", "donor")
        and row["transform_id"] in ("A1", "A2", "P", "C")
        and type(row["incongruent"]) is bool
        and type(row["answer_changes"]) is bool
        and type(row["answer_logit"]) in (int, float)
        and type(row["foil_logit"]) in (int, float)
        and row["answer_logit"] == row["answer_logit"]
        and row["foil_logit"] == row["foil_logit"]
        and abs(row["answer_logit"]) < 1e308
        and abs(row["foil_logit"]) < 1e308
        for row in rows
    )


def evaluate_answer_relation(rows):
    """Pure authority predicate for answer-changing versus invariant families."""
    return len(rows) == 256 and all(
        row["answer_changes"] is (row["transform_id"] in ("A1", "A2"))
        for row in rows
    )


def _accuracy(rows):
    return sum(row["answer_logit"] > row["foil_logit"] for row in rows) / len(rows)


def _mean_margin(rows):
    return sum(row["answer_logit"] - row["foil_logit"] for row in rows) / len(rows)


def evaluate_native_capability(rows):
    """Pure frozen gate implementing every registered scientific threshold."""
    base = [row for row in rows if row["side"] == "base"]
    donor = [row for row in rows if row["side"] == "donor"]
    ordinary = [
        [row for row in rows if row["side"] == side and row["transform_id"] == transform]
        for side in ("base", "donor") for transform in ("A1", "A2", "P")
    ]
    incongruent = [[row for row in cell if row["incongruent"]] for cell in ordinary]
    coordinated = [
        [row for row in rows if row["side"] == side and row["transform_id"] == "C"]
        for side in ("base", "donor")
    ]
    return (
        len(base) == 128 and len(donor) == 128
        and all(len(cell) == 32 for cell in ordinary + coordinated)
        and all(len(cell) == 16 for cell in incongruent)
        and _accuracy(base) >= 0.85 and _accuracy(donor) >= 0.85
        and all(_accuracy(cell) >= 0.85 and _mean_margin(cell) > 0.0 for cell in ordinary)
        and all(_accuracy(cell) >= 0.85 and _mean_margin(cell) > 0.0 for cell in incongruent)
        and all(_accuracy(cell) >= 0.75 and _mean_margin(cell) > 0.0 for cell in coordinated)
    )


def project_capability(rows):
    """Pure order-independent capability summary with no localization output."""
    base = [row for row in rows if row["side"] == "base"]
    donor = [row for row in rows if row["side"] == "donor"]
    names = [
        side + ":" + transform
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    cells = [
        [row for row in rows if row["side"] == side and row["transform_id"] == transform]
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    ordinary_names = [
        side + ":" + transform
        for side in ("base", "donor") for transform in ("A1", "A2", "P")
    ]
    ordinary_cells = [
        [row for row in rows if row["side"] == side and row["transform_id"] == transform]
        for side in ("base", "donor") for transform in ("A1", "A2", "P")
    ]
    incongruent_cells = [[row for row in cell if row["incongruent"]] for cell in ordinary_cells]
    return {
        "base_accuracy": _accuracy(base),
        "donor_accuracy": _accuracy(donor),
        "cell_accuracies": {name: _accuracy(cell) for name, cell in zip(names, cells)},
        "cell_mean_margins": {name: _mean_margin(cell) for name, cell in zip(names, cells)},
        "incongruent_accuracies": {
            name: _accuracy(cell) for name, cell in zip(ordinary_names, incongruent_cells)
        },
        "incongruent_mean_margins": {
            name: _mean_margin(cell) for name, cell in zip(ordinary_names, incongruent_cells)
        },
        "capability_pass": True,
    }


def _expected_primitive_keys(compiled: Mapping[str, object]) -> set[tuple[object, ...]]:
    output: set[tuple[object, ...]] = set()
    for call in compiled["metric_manifest"]:
        for row_id, incongruent, answer_changes in zip(
            call["row_ids"], call["incongruent"], call["answer_changes"]
        ):
            output.add((
                call["call_id"], row_id, call["side"], call["transform_id"],
                incongruent, answer_changes,
            ))
    return output


def _validate_compiled_contract(compiled: Mapping[str, object]) -> None:
    """Rebind every mutable data surface before evidence can be interpreted."""
    if compiled.get("spec_sha256") != SPEC_SHA256:
        raise CapabilityCompileError("compiled experiment spec changed")
    manifest = compiled.get("call_manifest")
    if not isinstance(manifest, list) \
            or framework.canonical_sha256(manifest) != CALL_MANIFEST_SHA256:
        raise CapabilityCompileError("compiled physical call manifest changed")
    expected_ids = [
        f"FIT:{side}:{transform}:0:native_{side}_{transform}"
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    if [call.get("call_id") for call in manifest] != expected_ids:
        raise CapabilityCompileError("task14 physical call order changed")
    if any(
        call.get("split") != "FIT"
        or call.get("guard") != "capability_only"
        or call.get("arm_role") != "native"
        or call.get("logical_batch_size") != 32
        or call.get("call_kind") != "native_answer_foil_logits"
        or call.get("array_contracts") != [
            {"name": "answer_logit", "dtype": "float32", "shape": ["batch"],
             "finite_policy": "always"},
            {"name": "foil_logit", "dtype": "float32", "shape": ["batch"],
             "finite_policy": "always"},
        ]
        for call in manifest
    ):
        raise CapabilityCompileError("task14 call request surface changed")
    summary = framework.summarize_call_manifest(manifest)
    if compiled.get("call_summary") != summary \
            or summary["manifest_sha256"] != CALL_MANIFEST_SHA256 \
            or summary["call_count"] != PRICE.forward_calls:
        raise CapabilityCompileError("compiled call summary differs from its physical manifest")
    metric_manifest = compiled.get("metric_manifest")
    if framework.canonical_sha256(metric_manifest) != METRIC_MANIFEST_SHA256 \
            or compiled.get("metric_manifest_sha256") != METRIC_MANIFEST_SHA256:
        raise CapabilityCompileError("compiled native metric manifest changed")
    if compiled.get("literal_price") != asdict(PRICE):
        raise CapabilityCompileError("compiled literal price changed")
    battery.validate_price(compiled, PRICE)
    if PRICE.evidence_bytes != PRICE.example_evaluations * 2 * 4:
        raise CapabilityCompileError("task14 raw float32 evidence price changed")
    row_sides = [
        (row_id, call["side"])
        for call in metric_manifest for row_id in call["row_ids"]
    ]
    if len(row_sides) != 256 or len(set(row_sides)) != 256:
        raise CapabilityCompileError("task14 compiled row-side coverage is not unique")
    if framework.canonical_sha256(compiled) != COMPILED_CONTRACT_SHA256:
        raise CapabilityCompileError("compiled capability contract changed")


def decide_capability(
    compiled: Mapping[str, object], primitives: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate exact row/call coverage, then apply the fail-closed package gate."""
    if compiled.get("replacement_review_commit") != REPLACEMENT_REVIEW_COMMIT \
            or compiled.get("task14_authority_sha256") != TASK14_AUTHORITY_SHA256 \
            or compiled.get("fit_authority_sha256") != FIT_RECORDS_SHA256 \
            or compiled.get("capability_scope") != \
            "FIT_native_only_no_reader_localization_or_selection":
        raise CapabilityCompileError("compiled capability identity changed")
    _validate_compiled_contract(compiled)
    observed = [dict(row) for row in primitives]
    if any(set(row) != _PRIMITIVE_FIELDS for row in observed):
        raise CapabilityCompileError("primitive evidence contains localization or undeclared fields")
    keys = [(
        row["call_id"], row["row_id"], row["side"], row["transform_id"],
        row["incongruent"], row["answer_changes"],
    ) for row in observed]
    if len(keys) != len(set(keys)) or set(keys) != _expected_primitive_keys(compiled):
        raise CapabilityCompileError("primitive evidence differs from exact metric/call coverage")
    decision = package.decide_experiment(
        spec=build_spec(),
        compiled=compiled,
        primitives=observed,
        evaluators={
            "task14_metric_evidence": evaluate_metric_evidence,
            "task14_answer_relation": evaluate_answer_relation,
            "task14_native_capability": evaluate_native_capability,
        },
        projector=project_capability,
    )
    projection = decision["projection"]
    if set(projection) != set(_PROJECTION_FIELDS):
        raise CapabilityCompileError("capability projection surface changed")
    if any(
        forbidden in str(key).lower()
        for key in projection for forbidden in ("reader", "writer", "site", "component", "head", "mlp")
    ):
        raise CapabilityCompileError("capability decision attempted localization")
    return decision


def run_managed_dryrun(base_dir: Path = REPO_ROOT) -> dict[str, object]:
    """Capture the immutable FIT closure and compile without model calls."""
    spec = build_spec()
    managed.validate_dryrun_closure(spec)
    captured = managed.capture_frozen_artifacts(spec, base_dir=base_dir, dryrun=True)
    expected_roles = {reference.role for reference in spec.artifacts}
    if set(captured) != expected_roles:
        raise CapabilityCompileError("managed dryrun artifact closure changed")
    authority_roles = {
        reference.role for reference in spec.artifacts if reference.kind == "authority"
    }
    if authority_roles != {"fit_authority"}:
        raise CapabilityCompileError("managed invocation exposes non-FIT authority bytes")
    rows = load_fit_authority_bytes(captured["fit_authority"])
    compiled = compile_fit_invocation(rows)
    return {
        "schema": "circuit_battery_task14_capability_fit_dryrun_v1",
        "experiment_id": EXPERIMENT_ID,
        "captured_roles": sorted(captured),
        "authority_roles": sorted(authority_roles),
        "compiled_contract": compiled,
        "model_forwards_executed": 0,
        "model_backwards_executed": 0,
        "model_updates_executed": 0,
        "queue_touched": False,
    }


if __name__ == "__main__":
    print(json.dumps(run_managed_dryrun(), sort_keys=True))
