#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial CPU tests for task 21's capability-only FIT compiler."""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path

import pytest

import circuit_battery_integration_contract as battery
import circuit_battery_task21 as task21
import circuit_battery_task21_capability_fit as capability
import circuit_experiment_spec as framework
import circuit_managed_entry as managed


OPS = Path(__file__).resolve().parent


def fit_rows() -> list[dict[str, object]]:
    payload = (OPS / "circuit_battery_task21_copy_fit_authority.json").read_bytes()
    return capability.load_fit_authority_bytes(payload)


def primitives(compiled: dict[str, object], answer: float = 1.0, foil: float = 0.0):
    output = []
    for call in compiled["metric_manifest"]:
        for row_id, transform in zip(call["row_ids"], call["transform_ids"]):
            output.append({
                "call_id": call["call_id"], "row_id": row_id, "side": call["side"],
                "transform_id": transform, "answer_logit": answer, "max_foil_logit": foil,
            })
    return output


def test_exact_fit_wrapper_parent_hash_and_future_artifacts_absent() -> None:
    path = OPS / "circuit_battery_task21_copy_fit_authority.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == capability.FIT_AUTHORITY_FILE_SHA256
    rows = fit_rows()
    assert len(rows) == 84
    assert framework.canonical_sha256(rows) == capability.FIT_RECORDS_SHA256
    assert {row["split"] for row in rows} == {"FIT"}
    assert capability.TASK21_AUTHORITY_SHA256 \
        == "191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b"
    spec = capability.build_spec()
    authorities = [artifact for artifact in spec.artifacts if artifact.kind == "authority"]
    assert [(artifact.role, artifact.path) for artifact in authorities] == [
        ("fit_authority", capability.FIT_AUTHORITY_PATH)
    ]
    assert all(artifact.kind != "outcome" for artifact in spec.artifacts)
    assert [phase.name for phase in spec.phases] == ["FIT"]
    assert spec.phases[0].forbidden_splits == ("SELECT", "TEST", "OOD")
    assert not any(word in artifact.path.lower() for artifact in spec.artifacts
                   for word in ("select_authority", "test_authority", "ood_authority"))


def test_managed_dryrun_cannot_generate_or_open_later_splits(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("task generator entered FIT dryrun")
    monkeypatch.setattr(task21, "build_authority", forbidden)
    monkeypatch.setattr(task21, "_panel", forbidden)
    monkeypatch.setattr(task21, "validate_authority", forbidden)
    report = capability.run_managed_dryrun()
    assert report["authority_roles"] == ["fit_authority"]
    assert report["model_forwards_executed"] == 0
    assert report["model_backwards_executed"] == 0
    assert report["model_updates_executed"] == 0
    assert report["queue_touched"] is False
    assert report["compiled_contract"]["later_split_generation"] is False
    assert report["compiled_contract"]["later_split_artifacts"] == []


def test_checked_in_dryrun_is_exact_deterministic_zero_execution_report() -> None:
    path = OPS.parent / "circuit_battery_task21_capability_fit_v1_dryrun.json"
    assert hashlib.sha256(path.read_bytes()).hexdigest() \
        == "7f508a6daa6d322672a386316cb72d4adcd2738e001809eceaf4e62656aae408"
    assert json.loads(path.read_bytes()) == capability.run_managed_dryrun()


def test_exact_calls_metric_balance_and_literal_price() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    calls = compiled["call_manifest"]
    assert [call["call_id"] for call in calls] == [
        *(f"FIT:base:{index}:native_base" for index in range(4)),
        *(f"FIT:donor:{index}:native_donor" for index in range(4)),
    ]
    assert framework.canonical_sha256(calls) == capability.CALL_MANIFEST_SHA256
    assert all(
        call["logical_batch_size"] == 21
        and call["padded_sequence_length"] == 8
        and call["call_kind"] == "native_answer_foil_logits"
        and call["guard"] == "capability_only"
        and call["arm_role"] == "native"
        for call in calls
    )
    assert compiled["literal_price"] == {
        "phase": "FIT", "forward_calls": 8, "example_evaluations": 168,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 1344,
    }
    battery.validate_price(compiled, capability.PRICE)
    assert sum(call["logical_batch_size"] for call in calls) == 168
    assert capability.PRICE.evidence_bytes == 168 * 2 * 4

    target_counts = defaultdict(Counter)
    foil_counts = defaultdict(Counter)
    for plan in compiled["metric_manifest"]:
        for transform, target, foils in zip(
                plan["transform_ids"], plan["target_token_ids"], plan["foil_token_ids"]):
            cell = (plan["side"], transform)
            assert target not in foils and foils
            target_counts[cell][target] += 1
            foil_counts[cell].update(foils)
    assert set(target_counts) == {
        (side, transform) for side in ("base", "donor")
        for transform in ("A1", "A2", "P", "C")
    }
    assert all(len(counts) == 21 and set(counts.values()) == {1}
               for counts in target_counts.values())
    for (_, transform), counts in foil_counts.items():
        assert len(counts) == 21
        assert set(counts.values()) == ({2} if transform == "C" else {3})


def test_future_row_authority_byte_and_wrapper_metadata_attacks_fail(tmp_path: Path) -> None:
    rows = fit_rows()
    planted = deepcopy(rows[0]); planted["split"] = "SELECT"
    with pytest.raises(capability.CapabilityCompileError, match="future split"):
        capability.compile_fit_invocation(rows + [planted])
    rows = fit_rows(); rows[0]["base_ids"] = list(rows[0]["base_ids"]); rows[0]["base_ids"][0] += 1
    with pytest.raises(capability.CapabilityCompileError, match="authority changed"):
        capability.compile_fit_invocation(rows)

    spec = capability.build_spec()
    for artifact in spec.artifacts:
        source = capability.REPO_ROOT / artifact.path
        destination = tmp_path / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    authority = tmp_path / capability.FIT_AUTHORITY_PATH
    authority.write_bytes(authority.read_bytes().replace(b'"split_records_sha256":', b'"bad_records_sha256":', 1))
    with pytest.raises(managed.ManagedEntryError, match="frozen artifact changed"):
        capability.run_managed_dryrun(tmp_path)


def test_call_metric_primitive_and_price_attacks_fail_closed() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    evidence = primitives(compiled)
    attacked = deepcopy(compiled)
    old_row = attacked["call_manifest"][0]["row_ids"][0]
    attacked["call_manifest"][0]["row_ids"][0] = "a" * 64
    attacked["call_summary"] = framework.summarize_call_manifest(attacked["call_manifest"])
    attacked["metric_manifest"][0]["row_ids"][0] = "a" * 64
    attacked["metric_manifest_sha256"] = framework.canonical_sha256(attacked["metric_manifest"])
    next(row for row in evidence if row["row_id"] == old_row and row["side"] == "base")["row_id"] = "a" * 64
    with pytest.raises(capability.CapabilityCompileError, match="call manifest"):
        capability.decide_capability(attacked, evidence)

    attacked = deepcopy(compiled); attacked["literal_price"] = dict(attacked["literal_price"])
    attacked["literal_price"]["example_evaluations"] = 167
    with pytest.raises(capability.CapabilityCompileError, match="literal price"):
        capability.decide_capability(attacked, primitives(compiled))
    wrong = replace(capability.PRICE, example_evaluations=167)
    with pytest.raises(battery.BatteryContractError):
        battery.validate_price(compiled, wrong)


def test_capability_pass_and_fail_are_exact_complements_with_null_abort() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    passing = primitives(compiled)
    decision = capability.decide_capability(compiled, passing)
    assert decision["terminal"] == "ok"
    assert decision["projection"]["capability_pass"] is True
    failing = primitives(compiled)
    for row in failing:
        if row["side"] == "base" and row["transform_id"] == "A1":
            row["answer_logit"] = -1.0
    stopped = capability.decide_capability(compiled, failing)
    assert stopped["terminal"] == "hard_abort"
    assert stopped["predicate_results"]["native_capability_gate"] is False
    assert set(stopped["projection"]) == set(capability._PROJECTION_FIELDS)
    assert all(value is None for value in stopped["projection"].values())


def test_cell_and_side_bars_have_registered_integer_boundaries() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    rows = primitives(compiled)
    cell = [row for row in rows if row["side"] == "base" and row["transform_id"] == "A1"]
    for row in cell[:3]:
        row["answer_logit"] = -1.0
    assert capability.evaluate_native_capability(rows)  # 18/21 > .85 and 81/84 > .90
    cell[3]["answer_logit"] = -1.0
    assert not capability.evaluate_native_capability(rows)  # 17/21 < .85
    rows = primitives(compiled)
    for transform in ("A1", "A2", "P", "C"):
        cell = [row for row in rows if row["side"] == "base" and row["transform_id"] == transform]
        failures = 3 if transform == "A1" else 2
        for row in cell[:failures]:
            row["answer_logit"] = -1.0
    assert not capability.evaluate_native_capability(rows)  # 75/84 < .90, every cell >=18/21


def test_localization_fields_incomplete_coverage_and_forbidden_imports() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    planted = primitives(compiled); planted[0]["selected_reader"] = "mlp8"
    with pytest.raises(capability.CapabilityCompileError, match="localization"):
        capability.decide_capability(compiled, planted)
    with pytest.raises(capability.CapabilityCompileError, match="coverage"):
        capability.decide_capability(compiled, primitives(compiled)[:-1])

    source = (OPS / "circuit_battery_task21_capability_fit.py").read_text()
    imported = {
        alias.name for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names
    }
    assert "torch" not in imported
    assert "circuit_battery_tasks" not in imported
    assert "run_science" not in source and "enqueue" not in source
    assert "localization" not in set(capability._PROJECTION_FIELDS)
