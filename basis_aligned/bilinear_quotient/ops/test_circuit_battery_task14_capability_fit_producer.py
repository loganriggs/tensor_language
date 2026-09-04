#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free/adversarial tests for the task14 capability producer."""

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
import circuit_battery_task14_capability_fit as capability
import circuit_battery_task14_capability_fit_producer as producer
import circuit_experiment_spec as framework


OPS = Path(__file__).resolve().parent
AUTHORITY = OPS / "circuit_battery_task14_agreement_fit_authority.json"
SOURCE = OPS / "circuit_battery_task14_capability_fit_producer.py"


def captured():
    return {"fit_authority": AUTHORITY.read_bytes()}


def contract():
    return producer.compile_from_captured(captured())


def passing(seen=None):
    records = [] if seen is None else seen

    def evaluate(call, sequences, targets, foils):
        records.append((deepcopy(call), sequences.copy(), targets.copy(), foils.copy()))
        return np.ones(32, dtype="<f4"), np.zeros(32, dtype="<f4")

    return evaluate


def paths(root):
    return package.PackagePaths(
        root=root,
        result=root / f"{producer.NAMESPACE}_results.json",
        receipt=root / f"{producer.NAMESPACE}_receipt.json",
        evidence=root / f"{producer.NAMESPACE}_evidence",
        namespace=producer.NAMESPACE,
    )


def stage(value):
    return package.stage_package(
        value,
        evidence_files={"calls/0000_fixture/answer_logit.npy": b"fixture"},
        result={"schema": "fixture-result-v1"},
    )


@dataclass(frozen=True)
class FakeCheckpoint:
    revision: str = producer.MODEL_REVISION
    config_sha256: str = producer.CHECKPOINT_CONFIG_SHA256
    weights_sha256: str = producer.CHECKPOINT_WEIGHTS_SHA256
    weights_bytes: int = producer.CHECKPOINT_WEIGHTS_BYTES
    tokenizer_vocab: int = 50_257
    logit_vocab: int = 50_304


def test_exact_model_free_dryrun_and_distinct_failures():
    report = producer.run_dryrun(captured())
    assert report["compiled_contract_sha256"] == capability.COMPILED_CONTRACT_SHA256
    assert report["call_manifest_sha256"] == capability.CALL_MANIFEST_SHA256
    assert report["metric_manifest_sha256"] == capability.METRIC_MANIFEST_SHA256
    assert (report["completed_calls"], report["example_evaluations"]) == (8, 256)
    assert report["raw_numeric_evidence_bytes"] == 2048
    assert report["evidence_file_count"] == 24
    assert report["passing_fixture_terminal"] == "ok"
    assert report["scientific_fail_terminal"] == "hard_abort"
    assert report["instrument_fail_terminal"] == "hard_abort"
    assert report["scientific_fail_projection_all_null"] is True
    assert report["instrument_fail_projection_all_null"] is True
    assert report["instrument_and_science_predicates_distinct"] is True
    assert report["model_loaded"] is report["gpu_accessed"] is False
    assert report["model_forwards"] == report["model_backwards"] == 0
    assert report["publication_attempted"] is report["queue_touched"] is False


def test_exact_call_order_lengths_rows_tokens_labels_and_evidence():
    rows, compiled = contract()
    seen = []
    evidence, primitives, completed = producer.execute_call_manifest(
        rows, compiled, passing(seen)
    )
    assert [c["call_id"] for c in completed] == [
        f"FIT:{side}:{transform}:0:native_{side}_{transform}"
        for side in ("base", "donor") for transform in ("A1", "A2", "P", "C")
    ]
    assert [item[1].shape for item in seen] == [
        (32, length) for length in (5, 8, 5, 8, 5, 8, 5, 8)
    ]
    assert all(item[2].shape == item[3].shape == (32,) for item in seen)
    assert len(primitives) == len({(r["row_id"], r["side"]) for r in primitives}) == 256
    assert all(set(row) == {
        "call_id", "row_id", "side", "transform_id", "incongruent",
        "answer_changes", "answer_logit", "foil_logit",
    } for row in primitives)
    assert framework.canonical_sha256(completed) == capability.CALL_MANIFEST_SHA256
    assert len(evidence) == 24 and producer.evidence_numeric_bytes(evidence) == 2048
    assert sum(name.endswith("/call.json") for name in evidence) == 8
    assert sum(name.endswith("/answer_logit.npy") for name in evidence) == 8
    assert sum(name.endswith("/foil_logit.npy") for name in evidence) == 8
    for name, payload in evidence.items():
        if name.endswith(".npy"):
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            assert array.shape == (32,) and array.dtype == np.float32
            assert array.flags.c_contiguous
        else:
            assert json.loads(payload) in compiled["call_manifest"]


@pytest.mark.parametrize("attack", ("shape", "dtype", "noncontiguous", "nonfinite"))
def test_malformed_arrays_fail_before_scoring(attack):
    rows, compiled = contract()

    def bad(_call, _sequences, _targets, _foils):
        value = np.zeros(32, dtype="<f4")
        if attack == "shape":
            value = value[:-1]
        elif attack == "dtype":
            value = value.astype("<f8")
        elif attack == "noncontiguous":
            value = np.zeros(64, dtype="<f4")[::2]
        else:
            value[0] = np.nan
        return value, np.zeros(32, dtype="<f4")

    with pytest.raises(producer.ProducerError, match=r"float32\[32\]"):
        producer.execute_call_manifest(rows, compiled, bad)


@pytest.mark.parametrize("field", ("row_ids", "prediction_positions", "incongruent", "answer_changes"))
def test_call_metric_rebinding_rejected(field):
    rows, compiled = contract()
    attack = deepcopy(compiled)
    if field == "row_ids":
        values = attack["metric_manifest"][0][field]
        values[0], values[1] = values[1], values[0]
    elif field == "prediction_positions":
        attack["metric_manifest"][0][field][0] = 0
    else:
        attack["metric_manifest"][0][field][0] = not attack["metric_manifest"][0][field][0]
    with pytest.raises(producer.ProducerError):
        producer.execute_call_manifest(rows, attack, passing())


def test_future_phase_and_missing_authority_reject():
    with pytest.raises(producer.ProducerError, match="did not capture"):
        producer.compile_from_captured({})
    attack = captured(); attack["ood_authority"] = b"planted"
    with pytest.raises(producer.ProducerError, match="later-phase"):
        producer.compile_from_captured(attack)


def test_nested_localization_surface_rejected():
    with pytest.raises(producer.ProducerError, match="undeclared analysis"):
        producer.validate_result_surface({
            "decision": {"terminal": "ok", "projection": {"selected_reader": "x"}}
        })


@pytest.mark.parametrize("field", ("result", "receipt", "evidence"))
def test_dangling_symlink_occupies_each_final_namespace(tmp_path, field):
    value = paths(tmp_path)
    destination = getattr(value, field)
    destination.symlink_to(tmp_path / "missing")
    assert destination.is_symlink() and not destination.exists()
    with pytest.raises(producer.ProducerError, match="occupied"):
        producer.require_unused_namespaces(value)


@pytest.mark.parametrize("label", ("evidence", "result", "receipt"))
def test_late_race_never_overwritten_and_retryable(tmp_path, label):
    value = paths(tmp_path); staged = stage(value); raced = getattr(value, label)

    def plant(current, _source, destination):
        if current == label:
            destination.symlink_to(tmp_path / "external-missing")

    with pytest.raises(FileExistsError):
        producer.publish_task14_package(staged, value, before_move=plant)
    assert raced.is_symlink()
    for other in ("evidence", "result", "receipt"):
        if other != label:
            assert not producer.path_entry_exists(getattr(value, other))
    raced.unlink()
    producer.publish_task14_package(staged, value)
    assert package.validate_complete_package(value)["schema"] == "fixture-result-v1"


@pytest.mark.parametrize("label", ("evidence", "result", "receipt"))
def test_crash_rolls_back_owned_entries_and_receipt_is_last(tmp_path, label):
    value = paths(tmp_path); staged = stage(value); order = []

    def crash(point):
        order.append(point)
        if point == f"published:{label}":
            raise RuntimeError("planted")

    with pytest.raises(RuntimeError, match="planted"):
        producer.publish_task14_package(staged, value, crash=crash)
    assert not any(producer.path_entry_exists(getattr(value, x)) for x in (
        "evidence", "result", "receipt"
    ))
    producer.publish_task14_package(staged, value)
    assert package.validate_complete_package(value)["schema"] == "fixture-result-v1"


def test_successful_publication_orders_receipt_last(tmp_path):
    value = paths(tmp_path); staged = stage(value); order = []

    def observe(label, _source, _destination):
        order.append(label)
        if label != "receipt":
            assert not producer.path_entry_exists(value.receipt)

    producer.publish_task14_package(staged, value, before_move=observe)
    assert order == ["evidence", "result", "receipt"]


@pytest.mark.parametrize("capability_pass", (True, False))
def test_fake_runtime_science_publishes_exact_terminal(tmp_path, monkeypatch, capability_pass):
    value = paths(tmp_path)
    fake_facade = ModuleType("bilin18_observed_model_facade")
    fake_facade.CONFIG_SHA256 = producer.CHECKPOINT_CONFIG_SHA256
    fake_facade.WEIGHTS_SHA256 = producer.CHECKPOINT_WEIGHTS_SHA256
    fake_facade.WEIGHTS_BYTES = producer.CHECKPOINT_WEIGHTS_BYTES
    fake_facade.MODEL_REVISION = producer.MODEL_REVISION
    fake_facade.validate_snapshot = lambda **_kwargs: FakeCheckpoint()
    fake_facade.validate_production_model = lambda _model: None
    fake_fastload = ModuleType("fastload")
    fake_fastload.load_model_fast = lambda: object()
    fake_torch = ModuleType("torch"); fake_torch.float32 = object()
    monkeypatch.setitem(sys.modules, "bilin18_observed_model_facade", fake_facade)
    monkeypatch.setitem(sys.modules, "fastload", fake_fastload)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(producer, "runtime_receipt", lambda: dict(producer.EXPECTED_RUNTIME))
    monkeypatch.setattr(producer, "validate_canaries", lambda: {"canary1_pass": True})

    class FakeModel:
        def to(self, **_kwargs): return self

    fake_fastload.load_model_fast = lambda: FakeModel()

    def make_evaluator(_model):
        def evaluate(_call, _sequences, _targets, _foils):
            answer = np.full(32, 1.0 if capability_pass else -1.0, dtype="<f4")
            return answer, np.zeros(32, dtype="<f4")
        return evaluate

    monkeypatch.setattr(producer, "model_evaluator", make_evaluator)
    clock = iter((10.0, 11.0)).__next__
    receipt = producer.run_science(captured(), paths=value, clock=clock)
    assert receipt["terminal"] == ("ok" if capability_pass else "hard_abort")
    assert receipt["forward_calls"] == 8 and receipt["example_evaluations"] == 256
    assert receipt["raw_numeric_evidence_bytes"] == 2048
    result = package.validate_complete_package(value)
    if not capability_pass:
        assert all(x is None for x in result["decision"]["projection"].values())
    with pytest.raises(producer.ProducerError, match="occupied"):
        producer.run_science(captured(), paths=value)


def test_checkpoint_replay_change_aborts_before_publication(tmp_path, monkeypatch):
    value = paths(tmp_path)
    fake_facade = ModuleType("bilin18_observed_model_facade")
    fake_facade.CONFIG_SHA256 = producer.CHECKPOINT_CONFIG_SHA256
    fake_facade.WEIGHTS_SHA256 = producer.CHECKPOINT_WEIGHTS_SHA256
    fake_facade.WEIGHTS_BYTES = producer.CHECKPOINT_WEIGHTS_BYTES
    fake_facade.MODEL_REVISION = producer.MODEL_REVISION
    receipts = iter((FakeCheckpoint(), FakeCheckpoint(weights_bytes=1)))
    fake_facade.validate_snapshot = lambda **_kwargs: next(receipts)
    fake_facade.validate_production_model = lambda _model: None
    fake_fastload = ModuleType("fastload")
    class FakeModel:
        def to(self, **_kwargs): return self
    fake_fastload.load_model_fast = lambda: FakeModel()
    fake_torch = ModuleType("torch"); fake_torch.float32 = object()
    monkeypatch.setitem(sys.modules, "bilin18_observed_model_facade", fake_facade)
    monkeypatch.setitem(sys.modules, "fastload", fake_fastload)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(producer, "runtime_receipt", lambda: dict(producer.EXPECTED_RUNTIME))
    monkeypatch.setattr(producer, "validate_canaries", lambda: {"canary1_pass": True})
    monkeypatch.setattr(producer, "model_evaluator", lambda _model: passing())
    with pytest.raises(producer.ProducerError, match="checkpoint changed"):
        producer.run_science(captured(), paths=value)
    assert not any(producer.path_entry_exists(getattr(value, name)) for name in (
        "result", "receipt", "evidence"
    ))


def test_source_has_no_training_future_generator_or_old_battery():
    source = SOURCE.read_text()
    assert "circuit_battery.py" not in source
    assert "build_authority(" not in source
    assert ".backward(" not in source and "optimizer" not in source.lower()
    assert "load_model_fast" in source
    assert "30.0 * torch.tanh(logits / 30.0)" in source
