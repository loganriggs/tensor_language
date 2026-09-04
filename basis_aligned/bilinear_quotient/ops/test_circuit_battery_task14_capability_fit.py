#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial CPU tests for task14's capability-only FIT compiler."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import replace
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import circuit_battery_integration_contract as battery
import circuit_battery_task14 as task14
import circuit_battery_task14_capability_fit as capability
import circuit_experiment_spec as framework
import circuit_managed_entry as managed


OPS = Path(__file__).resolve().parent
DRYRUN_SHA256 = "9ee6bf94676befbb89101a254a902ebd84fb4a53a6bd3e748a0c9e36336e5636"
DRYRUN_LOGICAL_SHA256 = "3af0b62f3f8cdaa3eeab3f24e2e0537616baef33195ade025270489aa47e871d"


def fit_rows() -> list[dict[str, object]]:
    payload = (OPS / "circuit_battery_task14_agreement_fit_authority.json").read_bytes()
    return capability.load_fit_authority_bytes(payload)


def primitives(compiled: dict[str, object], answer: float = 1.0, foil: float = 0.0):
    output = []
    for call in compiled["metric_manifest"]:
        for row_id, incongruent, answer_changes in zip(
            call["row_ids"], call["incongruent"], call["answer_changes"]
        ):
            output.append({
                "call_id": call["call_id"],
                "row_id": row_id,
                "side": call["side"],
                "transform_id": call["transform_id"],
                "incongruent": incongruent,
                "answer_changes": answer_changes,
                "answer_logit": answer,
                "foil_logit": foil,
            })
    return output


def _fail(records, selected, value: float = -0.1) -> None:
    for row in selected:
        row["answer_logit"] = value


def test_exact_fit_wrapper_hash_rows_and_future_artifacts_absent() -> None:
    path = OPS / "circuit_battery_task14_agreement_fit_authority.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == capability.FIT_AUTHORITY_FILE_SHA256
    value = json.loads(path.read_bytes())
    assert set(value) == {
        "groups", "rows", "schema", "split", "split_records_sha256",
        "task14_authority_sha256", "task_id",
    }
    assert value["groups"] == 32
    assert value["split_records_sha256"] == capability.FIT_RECORDS_SHA256
    assert value["task14_authority_sha256"] == capability.TASK14_AUTHORITY_SHA256
    rows = fit_rows()
    assert len(rows) == 128
    assert framework.canonical_sha256(rows) == capability.FIT_RECORDS_SHA256
    assert {row["split"] for row in rows} == {"FIT"}
    assert len({row[f"{side}_text"] for row in rows for side in ("base", "donor")}) == 256
    authority_files = sorted(path.name for path in OPS.glob("circuit_battery_task14*authority*.json"))
    assert authority_files == ["circuit_battery_task14_agreement_fit_authority.json"]
    spec = capability.build_spec()
    authorities = [artifact for artifact in spec.artifacts if artifact.kind == "authority"]
    assert [(artifact.role, artifact.path) for artifact in authorities] == [
        ("fit_authority", capability.FIT_AUTHORITY_PATH)
    ]
    assert all(artifact.kind != "outcome" for artifact in spec.artifacts)
    assert [phase.name for phase in spec.phases] == ["FIT"]
    assert spec.phases[0].forbidden_splits == ("SELECT", "TEST", "OOD")
    assert not any(
        word in artifact.path.lower() for artifact in spec.artifacts
        for word in ("select_authority", "test_authority", "ood_authority")
    )


def test_exact_replacement_review_generator_memo_and_test_bindings() -> None:
    assert capability.REPLACEMENT_REVIEW_COMMIT \
        == "ea7efad782c088ba91a2ce338a9f740563c4e7c1"
    refs = {role: digest for role, _, digest, _, _ in capability.FROZEN_ARTIFACTS}
    assert refs["task14_generator"] \
        == "33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94"
    assert refs["task14_generator_tests"] \
        == "254fe3798efd8a4426f30e054fd8e5646a5bd6635df69815f376311ac2023694"
    assert refs["repaired_design_memo"] \
        == "3cb4556d1ad2c1564f2708028e5d624c4519fbc4d52a38cac27b9d10d8312f68"
    assert refs["replacement_authority_review"] \
        == "7249991dd727f6385d3269cce23b0e5f83c588bcef3488dce33ae19dfd223fd1"


def test_managed_dryrun_cannot_generate_or_open_later_splits(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("task generator entered FIT dryrun")

    monkeypatch.setattr(task14, "build_authority", forbidden)
    monkeypatch.setattr(task14, "_panel", forbidden)
    monkeypatch.setattr(task14, "validate_authority", forbidden)
    report = capability.run_managed_dryrun()
    assert report["authority_roles"] == ["fit_authority"]
    assert report["model_forwards_executed"] == 0
    assert report["model_backwards_executed"] == 0
    assert report["model_updates_executed"] == 0
    assert report["queue_touched"] is False
    assert report["compiled_contract"]["later_split_generation"] is False
    assert report["compiled_contract"]["later_split_artifacts"] == []


def test_checked_in_dryrun_is_exact_deterministic_zero_execution_report() -> None:
    path = OPS.parent / "circuit_battery_task14_capability_fit_v1_dryrun.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == DRYRUN_SHA256
    assert json.loads(path.read_bytes()) == capability.run_managed_dryrun()


def test_dryrun_is_deterministic_across_python_hash_seeds() -> None:
    command = [sys.executable, "-c", (
        "import circuit_battery_task14_capability_fit as c; "
        "import circuit_experiment_spec as f; "
        "print(f.canonical_sha256(c.run_managed_dryrun()))"
    )]
    for hash_seed in ("0", "1", "999"):
        environment = dict(
            os.environ,
            PYTHONHASHSEED=hash_seed,
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH=str(OPS),
            CUDA_VISIBLE_DEVICES="",
            BQLIB_NO_MODEL="1",
        )
        assert subprocess.check_output(command, env=environment, text=True).strip() \
            == DRYRUN_LOGICAL_SHA256


def test_exact_call_order_shapes_metrics_and_literal_price() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    calls = compiled["call_manifest"]
    assert [call["call_id"] for call in calls] == [
        f"FIT:{side}:{transform}:0:native_{side}_{transform}"
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    assert [call["padded_sequence_length"] for call in calls] == [5, 8, 5, 8] * 2
    assert framework.canonical_sha256(calls) == capability.CALL_MANIFEST_SHA256
    assert all(
        call["logical_batch_size"] == 32
        and call["call_kind"] == "native_answer_foil_logits"
        and call["guard"] == "capability_only"
        and call["arm_role"] == "native"
        and [item["name"] for item in call["array_contracts"]]
        == ["answer_logit", "foil_logit"]
        for call in calls
    )
    assert compiled["literal_price"] == {
        "phase": "FIT", "forward_calls": 8, "example_evaluations": 256,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 2048,
    }
    battery.validate_price(compiled, capability.PRICE)
    assert sum(call["logical_batch_size"] for call in calls) == 256
    assert capability.PRICE.evidence_bytes == 256 * 2 * 4

    row_sides = []
    answer_id = task14.ENCODING.encode(" is")[0]
    plural_id = task14.ENCODING.encode(" are")[0]
    for plan in compiled["metric_manifest"]:
        assert len(plan["row_ids"]) == 32
        assert set(plan["target_token_ids"] + plan["foil_token_ids"]) == {answer_id, plural_id}
        assert all(target != foil for target, foil in zip(
            plan["target_token_ids"], plan["foil_token_ids"]
        ))
        assert sum(plan["incongruent"]) == (0 if plan["transform_id"] == "C" else 16)
        assert set(plan["answer_changes"]) == {plan["transform_id"] in ("A1", "A2")}
        row_sides.extend((row_id, plan["side"]) for row_id in plan["row_ids"])
        counts = Counter(plan["target_token_ids"])
        assert counts == ({plural_id: 32} if plan["transform_id"] == "C"
                          else {answer_id: 16, plural_id: 16})
    assert len(row_sides) == len(set(row_sides)) == 256


def test_future_row_authority_byte_and_wrapper_metadata_attacks_fail(monkeypatch) -> None:
    rows = fit_rows()
    planted = deepcopy(rows[0])
    planted["split"] = "SELECT"
    with pytest.raises(capability.CapabilityCompileError, match="future split"):
        capability.compile_fit_invocation(rows + [planted])
    rows = fit_rows()
    rows[0]["base_ids"] = list(rows[0]["base_ids"])
    rows[0]["base_ids"][0] += 1
    with pytest.raises(capability.CapabilityCompileError, match="authority changed"):
        capability.compile_fit_invocation(rows)

    path = OPS / "circuit_battery_task14_agreement_fit_authority.json"
    value = json.loads(path.read_bytes())
    value["groups"] = 31
    attacked = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    monkeypatch.setattr(capability, "FIT_AUTHORITY_FILE_SHA256", hashlib.sha256(attacked).hexdigest())
    with pytest.raises(capability.CapabilityCompileError, match="metadata changed"):
        capability.load_fit_authority_bytes(attacked)


@pytest.mark.parametrize("attack", ("prediction_position", "coherent_prompt_transplant"))
def test_local_semantic_validation_survives_rebased_record_digest(
    monkeypatch, attack: str,
) -> None:
    rows = fit_rows()
    row = rows[0]
    if attack == "prediction_position":
        row["base_prediction_position"] = 0
    else:
        row["base_text"] = row["donor_text"]
        row["base_ids"] = list(row["donor_ids"])
    monkeypatch.setattr(capability, "FIT_RECORDS_SHA256", framework.canonical_sha256(rows))
    match = "prediction position" if attack == "prediction_position" else "prompts|intervention coordinates"
    with pytest.raises(capability.CapabilityCompileError, match=match):
        capability._validate_fit_rows(rows)


@pytest.mark.parametrize(
    "role", ("task14_generator", "task14_generator_tests", "preregistration",
             "replacement_authority_review", "fit_authority"),
)
def test_managed_closure_rejects_any_captured_source_or_authority_mutation(
    tmp_path: Path, role: str,
) -> None:
    spec = capability.build_spec()
    by_role = {artifact.role: artifact for artifact in spec.artifacts}
    for artifact in spec.artifacts:
        source = capability.REPO_ROOT / artifact.path
        destination = tmp_path / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    target = tmp_path / by_role[role].path
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(managed.ManagedEntryError, match="frozen artifact changed"):
        capability.run_managed_dryrun(tmp_path)


def test_call_metric_primitive_and_price_attacks_fail_closed() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    evidence = primitives(compiled)
    attacked = deepcopy(compiled)
    attacked["call_manifest"][0]["row_ids"][0] = "a" * 64
    attacked["call_summary"] = framework.summarize_call_manifest(attacked["call_manifest"])
    attacked["metric_manifest"][0]["row_ids"][0] = "a" * 64
    attacked["metric_manifest_sha256"] = framework.canonical_sha256(attacked["metric_manifest"])
    with pytest.raises(capability.CapabilityCompileError, match="call manifest"):
        capability.decide_capability(attacked, evidence)

    attacked = deepcopy(compiled)
    attacked["metric_manifest"][0]["foil_token_ids"][0] \
        = attacked["metric_manifest"][0]["target_token_ids"][0]
    attacked["metric_manifest_sha256"] = framework.canonical_sha256(attacked["metric_manifest"])
    with pytest.raises(capability.CapabilityCompileError, match="metric manifest"):
        capability.decide_capability(attacked, evidence)

    attacked = deepcopy(compiled)
    attacked["literal_price"] = dict(attacked["literal_price"])
    attacked["literal_price"]["example_evaluations"] = 255
    with pytest.raises(capability.CapabilityCompileError, match="literal price"):
        capability.decide_capability(attacked, evidence)
    wrong = replace(capability.PRICE, example_evaluations=255)
    with pytest.raises(battery.BatteryContractError):
        battery.validate_price(compiled, wrong)


def test_capability_pass_and_fail_are_complements_with_all_null_abort() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    passing = primitives(compiled)
    decision = capability.decide_capability(compiled, passing)
    assert decision["terminal"] == "ok"
    assert decision["projection"]["capability_pass"] is True
    failing = primitives(compiled)
    c_cell = [
        row for row in failing if row["side"] == "base" and row["transform_id"] == "C"
    ]
    _fail(failing, c_cell[:9])
    assert capability.evaluate_native_capability(failing) is False
    stopped = capability.decide_capability(compiled, failing)
    assert stopped["terminal"] == "hard_abort"
    assert stopped["predicate_results"]["native_capability_gate"] is False
    assert set(stopped["projection"]) == set(capability._PROJECTION_FIELDS)
    assert all(value is None for value in stopped["projection"].values())


def test_registered_cell_incongruent_c_and_zero_margin_boundaries() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())

    rows = primitives(compiled)
    cell = [row for row in rows if row["side"] == "base" and row["transform_id"] == "A1"]
    congruent = [row for row in cell if not row["incongruent"]]
    _fail(rows, congruent[:4])
    assert capability.evaluate_native_capability(rows)  # 28/32
    _fail(rows, congruent[4:5])
    assert not capability.evaluate_native_capability(rows)  # 27/32

    rows = primitives(compiled)
    cell = [row for row in rows if row["side"] == "base" and row["transform_id"] == "A1"]
    incongruent = [row for row in cell if row["incongruent"]]
    _fail(rows, incongruent[:2])
    assert capability.evaluate_native_capability(rows)  # 14/16
    _fail(rows, incongruent[2:3])
    assert not capability.evaluate_native_capability(rows)  # 13/16

    rows = primitives(compiled)
    c_cell = [row for row in rows if row["side"] == "base" and row["transform_id"] == "C"]
    _fail(rows, c_cell[:8])
    assert capability.evaluate_native_capability(rows)  # 24/32
    _fail(rows, c_cell[8:9])
    assert not capability.evaluate_native_capability(rows)  # 23/32

    rows = primitives(compiled)
    cell = [row for row in rows if row["side"] == "base" and row["transform_id"] == "A1"]
    congruent = [row for row in cell if not row["incongruent"]]
    _fail(rows, congruent[:4], value=-7.0)
    assert sum(row["answer_logit"] - row["foil_logit"] for row in cell) == 0.0
    assert not capability.evaluate_native_capability(rows)


def test_registered_pooled_side_boundary_is_independently_binding() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    rows = primitives(compiled)
    for transform in ("A1", "A2", "P"):
        cell = [
            row for row in rows
            if row["side"] == "base" and row["transform_id"] == transform
            and not row["incongruent"]
        ]
        _fail(rows, cell[:4])
    c_cell = [row for row in rows if row["side"] == "base" and row["transform_id"] == "C"]
    _fail(rows, c_cell[:7])
    assert sum(row["answer_logit"] > row["foil_logit"] for row in rows if row["side"] == "base") \
        == 109
    assert capability.evaluate_native_capability(rows)
    _fail(rows, c_cell[7:8])
    assert sum(row["answer_logit"] > row["foil_logit"] for row in rows if row["side"] == "base") \
        == 108
    assert not capability.evaluate_native_capability(rows)


def test_relation_localization_coverage_and_forbidden_import_attacks() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    rows = primitives(compiled)
    rows[0]["answer_changes"] = not rows[0]["answer_changes"]
    with pytest.raises(capability.CapabilityCompileError, match="coverage"):
        capability.decide_capability(compiled, rows)
    planted = primitives(compiled)
    planted[0]["selected_reader"] = "mlp8"
    with pytest.raises(capability.CapabilityCompileError, match="localization"):
        capability.decide_capability(compiled, planted)
    with pytest.raises(capability.CapabilityCompileError, match="coverage"):
        capability.decide_capability(compiled, primitives(compiled)[:-1])

    source = (OPS / "circuit_battery_task14_capability_fit.py").read_text()
    imported = {
        alias.name for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assert "torch" not in imported
    assert "run_science" not in source and "enqueue" not in source
    assert not set(capability._PROJECTION_FIELDS) & {
        "selected_reader", "selected_writer", "selected_site", "selected_component",
    }
