#!/usr/bin/env python3
# BQLANE: cpu
"""Adversarial CPU tests for task 17's capability-only FIT compilation."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import ast
import hashlib
from pathlib import Path

import pytest

import circuit_battery_integration_contract as battery
import circuit_battery_task17 as task17
import circuit_battery_task17_capability_fit as capability
import circuit_experiment_spec as framework
import circuit_managed_entry as managed


OPS = Path(__file__).resolve().parent


def fit_rows() -> list[dict[str, object]]:
    return capability.load_fit_authority_bytes((OPS / "circuit_battery_task17_fit_authority.json").read_bytes())


def primitives(compiled: dict[str, object], *, answer: float = 1.0, foil: float = 0.0) \
        -> list[dict[str, object]]:
    output = []
    for call in compiled["metric_manifest"]:
        for row_id, transform in zip(call["row_ids"], call["transform_ids"]):
            output.append({
                "call_id": call["call_id"],
                "row_id": row_id,
                "side": call["side"],
                "transform_id": transform,
                "answer_logit": answer,
                "max_foil_logit": foil,
            })
    return output


def test_exact_fit_authority_parent_hash_and_no_future_artifact() -> None:
    authority_path = OPS / "circuit_battery_task17_fit_authority.json"
    assert hashlib.sha256(authority_path.read_bytes()).hexdigest() \
        == capability.FIT_AUTHORITY_FILE_SHA256
    rows = fit_rows()
    assert len(rows) == 96
    assert framework.canonical_sha256(rows) == capability.FIT_RECORDS_SHA256
    assert {row["split"] for row in rows} == {"FIT"}
    assert capability.TASK17_AUTHORITY_SHA256 \
        == "16307b8bb9273d56f7c3d09cd629fca78fa1db7f110278e959b6ee301cfb7571"

    spec = capability.build_spec()
    authorities = [artifact for artifact in spec.artifacts if artifact.kind == "authority"]
    assert [(artifact.role, artifact.path) for artifact in authorities] == [
        ("fit_authority", capability.FIT_AUTHORITY_PATH)
    ]
    assert all(artifact.kind != "outcome" for artifact in spec.artifacts)
    assert [phase.name for phase in spec.phases] == ["FIT"]
    assert spec.phases[0].forbidden_splits == ("SELECT", "TEST", "OOD")


def test_managed_dryrun_does_not_generate_later_splits(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a task generator was called by the FIT invocation")

    monkeypatch.setattr(task17, "build_authority", forbidden)
    monkeypatch.setattr(task17, "_panel", forbidden)
    monkeypatch.setattr(task17, "validate_authority", forbidden)
    report = capability.run_managed_dryrun()
    assert report["authority_roles"] == ["fit_authority"]
    assert report["model_forwards_executed"] == 0
    assert report["model_backwards_executed"] == 0
    assert report["model_updates_executed"] == 0
    assert report["queue_touched"] is False
    compiled = report["compiled_contract"]
    assert compiled["later_split_generation"] is False
    assert compiled["later_split_artifacts"] == []
    assert {call["split"] for call in compiled["call_manifest"]} == {"FIT"}


def test_exact_physical_manifest_native_metric_and_literal_price() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    calls = compiled["call_manifest"]
    assert [call["call_id"] for call in calls] == [
        *(f"FIT:base:{index}:native_base" for index in range(4)),
        *(f"FIT:donor:{index}:native_donor" for index in range(4)),
    ]
    assert framework.canonical_sha256(calls) == capability.CALL_MANIFEST_SHA256
    assert all(
        call["logical_batch_size"] == 24
        and call["padded_sequence_length"] == 13
        and call["call_kind"] == "native_answer_foil_logits"
        and call["guard"] == "capability_only"
        and call["arm_role"] == "native"
        and call["arm_direction"] == "undirected"
        for call in calls
    )
    assert compiled["literal_price"] == {
        "phase": "FIT", "forward_calls": 8, "example_evaluations": 192,
        "backward_calls": 0, "model_updates": 0, "evidence_bytes": 1536,
    }
    battery.validate_price(compiled, capability.PRICE)
    assert sum(call["logical_batch_size"] for call in calls) == 192
    assert capability.PRICE.evidence_bytes == 192 * 2 * 4

    plans = compiled["metric_manifest"]
    assert len(plans) == 8
    assert sum(len(plan["row_ids"]) for plan in plans) == 192
    for plan in plans:
        assert len(plan["target_token_ids"]) == len(plan["foil_token_ids"]) == 24
        assert all(target not in foils and foils
                   for target, foils in zip(plan["target_token_ids"], plan["foil_token_ids"]))
        assert plan["metric"] == "answer_logit_minus_maximum_registered_foil_logit"
        assert plan["strict_correct_rule"] == "margin_gt_zero"


def test_planted_future_split_leakage_fails_before_hash_or_calls() -> None:
    rows = fit_rows()
    planted = deepcopy(rows[0])
    planted["split"] = "SELECT"
    with pytest.raises(capability.CapabilityCompileError, match="future split"):
        capability.compile_fit_invocation(rows + [planted])


def test_authority_mutation_and_managed_byte_mutation_fail_closed(tmp_path: Path) -> None:
    rows = fit_rows()
    rows[0]["base_ids"] = list(rows[0]["base_ids"])
    rows[0]["base_ids"][0] += 1
    with pytest.raises(capability.CapabilityCompileError, match="authority changed"):
        capability.compile_fit_invocation(rows)

    spec = capability.build_spec()
    for artifact in spec.artifacts:
        source = capability.REPO_ROOT / artifact.path
        destination = tmp_path / artifact.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    authority = tmp_path / capability.FIT_AUTHORITY_PATH
    authority.write_bytes(authority.read_bytes().replace(b'"groups":24', b'"groups":25', 1))
    with pytest.raises(managed.ManagedEntryError, match="frozen artifact changed"):
        capability.run_managed_dryrun(tmp_path)


def test_call_and_price_mismatches_abort() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    bad_call = deepcopy(compiled)
    bad_call["call_manifest"][0]["row_ids"] = list(bad_call["call_manifest"][0]["row_ids"])
    bad_call["call_manifest"][0]["row_ids"][0] = "planted-wrong-row"
    with pytest.raises(capability.CapabilityCompileError, match="call manifest"):
        capability.decide_capability(bad_call, primitives(compiled))

    wrong_price = replace(capability.PRICE, example_evaluations=191)
    with pytest.raises(battery.BatteryContractError, match="example-evaluation price"):
        battery.validate_price(compiled, wrong_price)
    with pytest.raises(battery.BatteryContractError, match="measured phase price"):
        battery.validate_price_receipt(capability.PRICE, wrong_price)


def test_jointly_mutated_call_metric_and_primitive_still_abort() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    evidence = primitives(compiled)
    attacked = deepcopy(compiled)
    old_row = attacked["call_manifest"][0]["row_ids"][0]
    new_row = "a" * 64
    attacked["call_manifest"][0]["row_ids"][0] = new_row
    attacked["call_summary"] = framework.summarize_call_manifest(attacked["call_manifest"])
    attacked["metric_manifest"][0]["row_ids"][0] = new_row
    attacked["metric_manifest_sha256"] = framework.canonical_sha256(attacked["metric_manifest"])
    matching = next(row for row in evidence if row["row_id"] == old_row and row["side"] == "base")
    matching["row_id"] = new_row
    with pytest.raises(capability.CapabilityCompileError, match="call manifest"):
        capability.decide_capability(attacked, evidence)


def test_compiled_literal_price_field_mutation_aborts() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    compiled["literal_price"] = dict(compiled["literal_price"])
    compiled["literal_price"]["example_evaluations"] = 191
    with pytest.raises(capability.CapabilityCompileError, match="literal price"):
        capability.decide_capability(compiled, primitives(compiled))


def test_capability_fail_hard_stops_without_reader_selection() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    passing = primitives(compiled)
    decision = capability.decide_capability(compiled, passing)
    assert decision["terminal"] == "ok"
    assert decision["projection"]["capability_pass"] is True
    assert decision["projection"]["base_accuracy"] == 1.0
    assert decision["projection"]["donor_accuracy"] == 1.0

    failing = primitives(compiled)
    for row in failing:
        if row["side"] == "base" and row["transform_id"] == "A1":
            row["answer_logit"] = -1.0
    stopped = capability.decide_capability(compiled, failing)
    assert stopped["terminal"] == "hard_abort"
    assert stopped["predicate_results"] == {
        "metric_evidence_contract": True,
        "native_capability_gate": False,
    }
    assert set(stopped["projection"]) == set(capability._PROJECTION_FIELDS)
    assert all(value is None for value in stopped["projection"].values())
    assert not any(
        word in key
        for key in stopped["projection"]
        for word in ("reader", "site", "component", "writer", "selection")
    )
    assert not any(
        word in key
        for key in compiled
        for word in ("reader", "site", "component", "writer", "selection")
    )


def test_planted_localization_field_and_incomplete_coverage_are_rejected() -> None:
    compiled = capability.compile_fit_invocation(fit_rows())
    planted = primitives(compiled)
    planted[0]["selected_reader"] = "mlp8"
    with pytest.raises(capability.CapabilityCompileError, match="localization"):
        capability.decide_capability(compiled, planted)
    with pytest.raises(capability.CapabilityCompileError, match="coverage"):
        capability.decide_capability(compiled, primitives(compiled)[:-1])


def test_owned_compiler_has_no_forbidden_runtime_imports_or_science_entry() -> None:
    source = (OPS / "circuit_battery_task17_capability_fit.py").read_text()
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imported
    assert "circuit_battery_tasks" not in imported
    assert not any("r593" in name.lower() for name in imported)
    assert "run_science" not in source
    assert "enqueue" not in source
