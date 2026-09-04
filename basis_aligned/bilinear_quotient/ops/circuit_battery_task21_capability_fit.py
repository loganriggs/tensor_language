#!/usr/bin/env python3
# BQLANE: cpu
"""Compile task 21's verbatim-copy capability-only FIT invocation without opening later splits.

This is an outcome-blind CPU compiler.  It does not import a model runtime and
does not provide a science entry point.  A later GPU producer must consume this
exact contract through a separately reviewed managed adapter.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import circuit_artifact_package as package
import circuit_battery_integration_contract as battery
import circuit_battery_task21 as task21
import circuit_experiment_spec as framework
import circuit_managed_entry as managed


EXPERIMENT_ID = "circuit-battery-task21-verbatim-copy-capability-fit-v1"
TASK21_AUTHORITY_SHA256 = "191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b"
FIT_RECORDS_SHA256 = "c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8"
FIT_AUTHORITY_FILE_SHA256 = "69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94"
CALL_MANIFEST_SHA256 = "ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e"
METRIC_MANIFEST_SHA256 = "e8cab6e2fb8000bd144f92182abd71c7774d3afcd2dc1b1de50f9c1a9ec79faf"
SPEC_SHA256 = "834f5b8f45facb7f585c96d7ee15b86e8a7a68f03117af929cf7843def8ab487"
COMPILED_CONTRACT_SHA256 = "5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2"

REPO_ROOT = Path(__file__).resolve().parents[3]
FIT_AUTHORITY_PATH = (
    "basis_aligned/bilinear_quotient/ops/circuit_battery_task21_copy_fit_authority.json"
)
PREREG_PATH = (
    "basis_aligned/polynomial_causal/"
    "CIRCUIT_BATTERY_TASK21_VERBATIM_COPY_CAPABILITY_FIT_PREREGISTRATION.md"
)

FROZEN_ARTIFACTS = (
    ("task21_adapter", "basis_aligned/bilinear_quotient/ops/circuit_battery_task21.py",
     "bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2", "source", "PROTOCOL"),
    ("battery_contract", "basis_aligned/bilinear_quotient/ops/circuit_battery_integration_contract.py",
     "b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e", "source", "PROTOCOL"),
    ("experiment_spec", "basis_aligned/bilinear_quotient/ops/circuit_experiment_spec.py",
     "64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c", "source", "PROTOCOL"),
    ("artifact_package", "basis_aligned/bilinear_quotient/ops/circuit_artifact_package.py",
     "6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc", "source", "PROTOCOL"),
    ("managed_entry", "basis_aligned/bilinear_quotient/ops/circuit_managed_entry.py",
     "1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81", "source", "PROTOCOL"),
    ("preregistration", PREREG_PATH,
     "da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3", "prereg", "PROTOCOL"),
    ("fit_authority", FIT_AUTHORITY_PATH, FIT_AUTHORITY_FILE_SHA256, "authority", "FIT"),
)

PRICE = battery.ExactPhasePrice(
    phase="FIT",
    forward_calls=8,
    example_evaluations=168,
    backward_calls=0,
    model_updates=0,
    evidence_bytes=1_344,
)

_PRIMITIVE_FIELDS = frozenset({
    "call_id", "row_id", "side", "transform_id", "answer_logit", "max_foil_logit",
})
_PROJECTION_FIELDS = (
    "base_accuracy", "donor_accuracy", "minimum_cell_accuracy",
    "base_mean_margin", "donor_mean_margin", "cell_accuracies", "capability_pass",
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


def _call_family(side: str) -> framework.CallFamilySpec:
    if side not in ("base", "donor"):
        raise CapabilityCompileError("native metric side is invalid")
    arm = f"native_{side}"
    return framework.CallFamilySpec(
        name=f"fit_native_{side}",
        split="FIT",
        arms=(arm,),
        batch_size=21,
        call_kind="native_answer_foil_logits",
        guard="capability_only",
        call_id_template="{split}:" + side + ":{batch}:{arm}",
        arm_specs=(framework.ArmSpec(arm, "native", "undirected"),),
        sequence_field=f"{side}_ids",
        axis_order="arm_batch",
    )


def build_spec() -> framework.CircuitExperimentSpec:
    """Build the FIT-only typed spec from frozen artifact digests."""
    artifacts = battery.phase_artifacts("FIT", _artifact_refs())
    spec = framework.CircuitExperimentSpec(
        experiment_id=EXPERIMENT_ID,
        rung=21,
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
            expected_counts={"FIT": 84},
            expected_total=84,
        ),),
        calls=(_call_family("base"), _call_family("donor")),
        arrays=(
            framework.ArraySpec(
                "answer_logit", ("native_answer_foil_logits",),
                "float32", ("batch",), True,
            ),
            framework.ArraySpec(
                "max_foil_logit", ("native_answer_foil_logits",),
                "float32", ("batch",), True,
            ),
        ),
        predicates=(
            framework.PredicateSpec(
                "metric_evidence_contract", "FIT", 0,
                "task21_metric_evidence", ("answer_logit", "max_foil_logit"),
                "hard_abort", "instrument",
            ),
            framework.PredicateSpec(
                "native_capability_gate", "FIT", 1,
                "task21_native_capability", ("answer_logit", "max_foil_logit"),
                "hard_abort", "science",
            ),
        ),
        science=framework.ScienceProjectionSpec(
            projector_role="task21_capability_projection",
            decision_role="capability_only_no_localization",
            allowed_terminals=("ok", "hard_abort"),
            output_types={
                "base_accuracy": "number",
                "donor_accuracy": "number",
                "minimum_cell_accuracy": "number",
                "base_mean_margin": "number",
                "donor_mean_margin": "number",
                "cell_accuracies": "mapping",
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
    observed = battery.validate_rows(task21.TASK_SPEC, copied, required_phases=("FIT",))
    if observed != FIT_RECORDS_SHA256:
        raise CapabilityCompileError("task21 FIT authority digest changed at integration boundary")
    groups = {(str(row["split"]), str(row["group_id"])) for row in copied}
    if len(groups) != 21:
        raise CapabilityCompileError("task21 FIT group census changed")
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
        "schema", "task_id", "split", "task21_authority_sha256",
        "split_records_sha256", "groups", "rows",
    }:
        raise CapabilityCompileError("FIT authority envelope changed")
    if value["schema"] != "circuit_battery_task21_split_authority_v1" \
            or value["task_id"] != task21.TASK_ID \
            or value["split"] != "FIT" \
            or value["task21_authority_sha256"] != TASK21_AUTHORITY_SHA256 \
            or value["split_records_sha256"] != FIT_RECORDS_SHA256 \
            or value["groups"] != 21 \
            or not isinstance(value["rows"], list):
        raise CapabilityCompileError("FIT authority metadata changed")
    return _validate_fit_rows(value["rows"])


def _side_metric(row: Mapping[str, object], side: str) -> tuple[int, list[int]]:
    target = row[f"{side}_answer_id"]
    if type(target) is not int:
        raise CapabilityCompileError("answer token ID is not an integer")
    values = set(row["base_tokens"]) | set(row["donor_tokens"])
    candidate_ids: set[int] = set()
    for value in values:
        encoded = task21.ENCODING.encode(" " + str(value))
        if len(encoded) != 1:
            raise CapabilityCompileError("registered answer candidate is not one token")
        candidate_ids.add(int(encoded[0]))
    foils = sorted(candidate_ids - {target})
    if not foils or target in foils:
        raise CapabilityCompileError("native side-specific answer/foil set is degenerate")
    return target, foils


def _metric_manifest(
    rows: Sequence[Mapping[str, object]], calls: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_id = {str(row["row_id"]): row for row in rows}
    output: list[dict[str, object]] = []
    for call in calls:
        arm = str(call["arm"])
        if arm not in ("native_base", "native_donor"):
            raise CapabilityCompileError("non-native arm entered capability manifest")
        side = arm.removeprefix("native_")
        row_ids = [str(value) for value in call["row_ids"]]
        selected = [by_id[row_id] for row_id in row_ids]
        metrics = [_side_metric(row, side) for row in selected]
        output.append({
            "call_id": call["call_id"],
            "side": side,
            "row_ids": row_ids,
            "transform_ids": [row["transform_id"] for row in selected],
            "target_token_ids": [target for target, _ in metrics],
            "foil_token_ids": [foils for _, foils in metrics],
            "metric": "answer_logit_minus_maximum_registered_foil_logit",
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
    if compiled["call_summary"]["manifest_sha256"] != CALL_MANIFEST_SHA256:
        raise CapabilityCompileError("physical call manifest changed")
    if any(call["split"] != "FIT" or call["guard"] != "capability_only"
           or call["arm_role"] != "native" for call in compiled["call_manifest"]):
        raise CapabilityCompileError("non-FIT or non-native call entered capability invocation")
    metric_manifest = _metric_manifest(frozen_rows, compiled["call_manifest"])
    compiled.update({
        "capability_scope": "FIT_native_only_no_reader_localization_or_selection",
        "task21_authority_sha256": TASK21_AUTHORITY_SHA256,
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
    return len(rows) == 168 and all(
        type(row) is dict
        and set(row) == {
            "call_id", "row_id", "side", "transform_id", "answer_logit", "max_foil_logit"
        }
        and type(row["call_id"]) is str
        and type(row["row_id"]) is str
        and row["side"] in ("base", "donor")
        and row["transform_id"] in ("A1", "A2", "P", "C")
        and type(row["answer_logit"]) in (int, float)
        and type(row["max_foil_logit"]) in (int, float)
        and row["answer_logit"] == row["answer_logit"]
        and row["max_foil_logit"] == row["max_foil_logit"]
        and abs(row["answer_logit"]) < 1e308
        and abs(row["max_foil_logit"]) < 1e308
        for row in rows
    )


def evaluate_native_capability(rows):
    """Pure frozen capability gate; it has no component or reader vocabulary."""
    base = [row for row in rows if row["side"] == "base"]
    donor = [row for row in rows if row["side"] == "donor"]
    cells = [
        [row for row in rows if row["side"] == side and row["transform_id"] == transform]
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    return (
        len(base) == 84 and len(donor) == 84
        and all(len(cell) == 21 for cell in cells)
        and sum(row["answer_logit"] > row["max_foil_logit"] for row in base) / 84 >= 0.90
        and sum(row["answer_logit"] > row["max_foil_logit"] for row in donor) / 84 >= 0.90
        and all(
            sum(row["answer_logit"] > row["max_foil_logit"] for row in cell) / 21 >= 0.85
            for cell in cells
        )
        and sum(row["answer_logit"] - row["max_foil_logit"] for row in base) / 84 > 0.0
        and sum(row["answer_logit"] - row["max_foil_logit"] for row in donor) / 84 > 0.0
    )


def project_capability(rows):
    """Pure order-independent capability summary; no localization output exists."""
    base = [row for row in rows if row["side"] == "base"]
    donor = [row for row in rows if row["side"] == "donor"]
    names = [
        side + ":" + transform
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    cells = [
        [
            row for row in rows
            if row["side"] == side and row["transform_id"] == transform
        ]
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    accuracies = [
        sum(row["answer_logit"] > row["max_foil_logit"] for row in cell) / len(cell)
        for cell in cells
    ]
    cell_accuracies = {names[index]: accuracies[index] for index in range(8)}
    return {
        "base_accuracy": sum(
            row["answer_logit"] > row["max_foil_logit"] for row in base
        ) / len(base),
        "donor_accuracy": sum(
            row["answer_logit"] > row["max_foil_logit"] for row in donor
        ) / len(donor),
        "minimum_cell_accuracy": min(accuracies),
        "base_mean_margin": sum(
            row["answer_logit"] - row["max_foil_logit"] for row in base
        ) / len(base),
        "donor_mean_margin": sum(
            row["answer_logit"] - row["max_foil_logit"] for row in donor
        ) / len(donor),
        "cell_accuracies": cell_accuracies,
        "capability_pass": True,
    }


def _expected_primitive_keys(compiled: Mapping[str, object]) -> set[tuple[str, str, str, str]]:
    output: set[tuple[str, str, str, str]] = set()
    for call in compiled["metric_manifest"]:
        for row_id, transform in zip(call["row_ids"], call["transform_ids"]):
            output.add((str(call["call_id"]), str(row_id), str(call["side"]), str(transform)))
    return output


def _validate_compiled_contract(compiled: Mapping[str, object]) -> None:
    """Rebind every mutable data surface before evidence can be interpreted."""
    if compiled.get("spec_sha256") != SPEC_SHA256:
        raise CapabilityCompileError("compiled experiment spec changed")
    manifest = compiled.get("call_manifest")
    if framework.canonical_sha256(manifest) != CALL_MANIFEST_SHA256:
        raise CapabilityCompileError("compiled physical call manifest changed")
    if not isinstance(manifest, list):
        raise CapabilityCompileError("compiled physical call manifest is not a list")
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
    if framework.canonical_sha256(compiled) != COMPILED_CONTRACT_SHA256:
        raise CapabilityCompileError("compiled capability contract changed")


def decide_capability(
    compiled: Mapping[str, object], primitives: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Validate exact row/call coverage, then use the artifact-package decision gate."""
    if compiled.get("task21_authority_sha256") != TASK21_AUTHORITY_SHA256 \
            or compiled.get("fit_authority_sha256") != FIT_RECORDS_SHA256 \
            or compiled.get("capability_scope") != \
            "FIT_native_only_no_reader_localization_or_selection":
        raise CapabilityCompileError("compiled capability identity changed")
    _validate_compiled_contract(compiled)
    observed = [dict(row) for row in primitives]
    if any(set(row) != _PRIMITIVE_FIELDS for row in observed):
        raise CapabilityCompileError("primitive evidence contains localization or undeclared fields")
    keys = [(
        str(row["call_id"]), str(row["row_id"]),
        str(row["side"]), str(row["transform_id"]),
    ) for row in observed]
    if len(keys) != len(set(keys)) or set(keys) != _expected_primitive_keys(compiled):
        raise CapabilityCompileError("primitive evidence differs from exact metric/call coverage")
    decision = package.decide_experiment(
        spec=build_spec(),
        compiled=compiled,
        primitives=observed,
        evaluators={
            "task21_metric_evidence": evaluate_metric_evidence,
            "task21_native_capability": evaluate_native_capability,
        },
        projector=project_capability,
    )
    projection = decision["projection"]
    if set(projection) != set(_PROJECTION_FIELDS):
        raise CapabilityCompileError("capability projection surface changed")
    if any("reader" in key or "site" in key or "component" in key for key in projection):
        raise CapabilityCompileError("capability decision attempted localization")
    return decision


def run_managed_dryrun(base_dir: Path = REPO_ROOT) -> dict[str, object]:
    """Capture the approved immutable closure and compile without model calls."""
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
        "schema": "circuit_battery_task21_capability_fit_dryrun_v1",
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
