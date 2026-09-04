#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free and adversarial tests for the task-17 capability producer."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import io
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

import circuit_artifact_package as package
import circuit_battery_task17_capability_fit as capability
import circuit_battery_task17_capability_fit_producer as producer
import circuit_experiment_spec as framework


OPS = Path(__file__).resolve().parent
AUTHORITY = OPS / "circuit_battery_task17_fit_authority.json"
SOURCE = OPS / "circuit_battery_task17_capability_fit_producer.py"


def captured() -> dict[str, bytes]:
    return {"fit_authority": AUTHORITY.read_bytes()}


def contract():
    rows, compiled = producer.compile_from_captured(captured())
    return rows, compiled


def passing_evaluator(records=None):
    seen = [] if records is None else records

    def evaluate(call, sequences, targets, foils):
        seen.append({
            "call": deepcopy(call), "sequences": sequences.copy(),
            "targets": targets.copy(), "foils": deepcopy(foils),
        })
        return np.full(24, 2.0, dtype="<f4"), np.full(24, -1.0, dtype="<f4")

    return evaluate


def package_paths(root: Path) -> package.PackagePaths:
    namespace = producer.NAMESPACE
    return package.PackagePaths(
        root=root,
        result=root / f"{namespace}_results.json",
        receipt=root / f"{namespace}_receipt.json",
        evidence=root / f"{namespace}_evidence",
        namespace=namespace,
    )


@dataclass(frozen=True)
class FakeCheckpoint:
    revision: str = producer.MODEL_REVISION
    config_sha256: str = producer.CHECKPOINT_CONFIG_SHA256
    weights_sha256: str = producer.CHECKPOINT_WEIGHTS_SHA256
    weights_bytes: int = producer.CHECKPOINT_WEIGHTS_BYTES
    tokenizer_vocab: int = 50_257
    logit_vocab: int = 50_304


def test_dryrun_is_exact_model_free_and_capability_fail_is_valid() -> None:
    report = producer.run_dryrun(captured())
    assert report["compiled_contract_sha256"] == capability.COMPILED_CONTRACT_SHA256
    assert report["call_manifest_sha256"] == capability.CALL_MANIFEST_SHA256
    assert report["completed_calls"] == 8
    assert report["example_evaluations"] == 192
    assert report["raw_numeric_evidence_bytes"] == 1536
    assert report["evidence_file_count"] == 24
    assert report["passing_fixture_terminal"] == "ok"
    assert report["failing_fixture_terminal"] == "hard_abort"
    assert report["failing_fixture_projection_all_null"] is True
    assert report["model_loaded"] is False
    assert report["model_forwards"] == report["model_backwards"] == 0
    assert report["forbidden_phases_opened"] == []


def test_every_row_side_is_physically_evaluated_without_prompt_deduplication() -> None:
    rows, compiled = contract()
    seen = []
    evidence, primitives, completed = producer.execute_call_manifest(
        rows, compiled, passing_evaluator(seen)
    )
    assert len(seen) == len(completed) == 8
    assert sum(item["sequences"].shape[0] for item in seen) == 192
    unique = {tuple(row) for item in seen for row in item["sequences"].tolist()}
    assert len(unique) == 144 < 192
    assert len(primitives) == 192
    assert framework.canonical_sha256(completed) == capability.CALL_MANIFEST_SHA256
    assert producer.evidence_numeric_bytes(evidence) == 1536


def test_evidence_is_only_exact_requests_and_two_float32_arrays_per_call() -> None:
    rows, compiled = contract()
    evidence, _, _ = producer.execute_call_manifest(
        rows, compiled, passing_evaluator()
    )
    assert len(evidence) == 24
    assert sum(name.endswith("/call.json") for name in evidence) == 8
    assert sum(name.endswith("/answer_logit.npy") for name in evidence) == 8
    assert sum(name.endswith("/max_foil_logit.npy") for name in evidence) == 8
    assert not any(any(word in name for word in (
        "activation", "logits.npy", "hidden", "reader", "component"
    )) for name in evidence)
    for name, payload in evidence.items():
        if name.endswith(".npy"):
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            assert array.shape == (24,)
            assert array.dtype == np.float32
            assert array.flags.c_contiguous
        else:
            request = json.loads(payload)
            assert request in compiled["call_manifest"]


@pytest.mark.parametrize("attack", ("shape", "dtype", "nonfinite"))
def test_malformed_model_arrays_fail_before_a_decision(attack: str) -> None:
    rows, compiled = contract()

    def malformed(_call, _sequences, _targets, _foils):
        answer = np.zeros(24, dtype="<f4")
        maximum = np.zeros(24, dtype="<f4")
        if attack == "shape":
            answer = answer[:-1]
        elif attack == "dtype":
            answer = answer.astype("<f8")
        else:
            answer[0] = np.nan
        return answer, maximum

    with pytest.raises(producer.ProducerError, match=r"float32\[24\]"):
        producer.execute_call_manifest(rows, compiled, malformed)


def test_call_metric_position_rebinding_is_rejected() -> None:
    rows, compiled = contract()
    attacked = deepcopy(compiled)
    attacked["metric_manifest"][0]["row_ids"] = list(
        attacked["metric_manifest"][0]["row_ids"]
    )
    attacked["metric_manifest"][0]["row_ids"][0], attacked["metric_manifest"][0]["row_ids"][1] = (
        attacked["metric_manifest"][0]["row_ids"][1], attacked["metric_manifest"][0]["row_ids"][0]
    )
    with pytest.raises(producer.ProducerError, match="not aligned"):
        producer.execute_call_manifest(rows, attacked, passing_evaluator())


def test_future_phase_role_is_rejected_before_compilation() -> None:
    attack = captured()
    attack["select_authority"] = b"planted"
    with pytest.raises(producer.ProducerError, match="later-phase"):
        producer.compile_from_captured(attack)


def test_nested_undeclared_analysis_key_is_rejected() -> None:
    planted = {
        "decision": {
            "terminal": "ok",
            "projection": {"selected_reader": "mlp8"},
        }
    }
    with pytest.raises(producer.ProducerError, match="undeclared analysis"):
        producer.validate_result_surface(planted)


def test_occupied_final_namespace_stops_before_evaluator(tmp_path: Path) -> None:
    paths = package_paths(tmp_path)
    paths.result.write_text("occupied")
    with pytest.raises(producer.ProducerError, match="occupied"):
        producer.require_unused_namespaces(paths)


def test_canary_gate_rejects_failure_and_changed_fingerprint(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({
        "pa": True, "pb": True, "pc": True, "score_rank": 1.0,
        "l1_cost": 1.0, "ratio_5_6": 1.0, "ratio_14_15": 1.0,
    }))
    second.write_text(json.dumps({
        "canary1": True, "atlases": True, "fingerprint_stable_vs_previous": True,
        "ALL": True, "fingerprint": {
            "composition": producer.CANARY2_COMPOSITION,
            "sha": producer.CANARY2_FINGERPRINT_SHA256,
        },
    }))
    assert producer.validate_canaries(first, second)["canary2_pass"] is True
    value = json.loads(second.read_text())
    value["fingerprint"]["sha"] = "0" * 64
    second.write_text(json.dumps(value))
    with pytest.raises(producer.ProducerError, match="fingerprint"):
        producer.validate_canaries(first, second)
    first.write_text(first.read_text().replace('"pa": true', '"pa": false'))
    with pytest.raises(producer.ProducerError, match="canary 1"):
        producer.validate_canaries(first, second)


@pytest.mark.parametrize("capability_pass", (True, False))
def test_fake_runtime_publishes_one_atomic_bound_package(
    tmp_path: Path, monkeypatch, capability_pass: bool,
) -> None:
    paths = package_paths(tmp_path)
    fake_facade = ModuleType("bilin18_observed_model_facade")
    fake_facade.CONFIG_SHA256 = producer.CHECKPOINT_CONFIG_SHA256
    fake_facade.WEIGHTS_SHA256 = producer.CHECKPOINT_WEIGHTS_SHA256
    fake_facade.MODEL_REVISION = producer.MODEL_REVISION
    fake_facade.load_bilin18 = lambda **_kwargs: (object(), FakeCheckpoint())
    monkeypatch.setitem(sys.modules, "bilin18_observed_model_facade", fake_facade)
    monkeypatch.setattr(producer, "runtime_receipt", lambda: dict(producer.EXPECTED_RUNTIME))
    monkeypatch.setattr(producer, "validate_canaries", lambda: {"canary1_pass": True})

    def evaluator(_model):
        def evaluate(_call, _sequences, _targets, _foils):
            answer = np.ones(24, dtype="<f4")
            maximum = np.zeros(24, dtype="<f4")
            if not capability_pass:
                answer.fill(-1.0)
            return answer, maximum
        return evaluate

    monkeypatch.setattr(producer, "model_evaluator", evaluator)
    clock = iter((10.0, 11.25)).__next__
    receipt = producer.run_science(captured(), paths=paths, clock=clock)
    assert receipt["forward_calls"] == 8
    assert receipt["example_evaluations"] == 192
    assert receipt["raw_numeric_evidence_bytes"] == 1536
    assert receipt["terminal"] == ("ok" if capability_pass else "hard_abort")
    result = package.validate_complete_package(paths)
    assert result["decision"]["terminal"] == receipt["terminal"]
    if not capability_pass:
        assert all(value is None for value in result["decision"]["projection"].values())
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted((
        paths.result.name, paths.receipt.name, paths.evidence.name,
    ))
    with pytest.raises(producer.ProducerError, match="occupied"):
        producer.run_science(captured(), paths=paths, clock=clock)


def test_source_has_no_old_battery_or_future_generator_or_training_path() -> None:
    source = SOURCE.read_text()
    assert "circuit_battery.py" not in source
    assert "circuit_battery_tasks" not in source
    assert "build_authority(" not in source
    assert "optimizer" not in source.lower()
    assert ".backward(" not in source
    assert "load_bilin18" in source
    assert "30.0 * torch.tanh(logits / 30.0)" in source
