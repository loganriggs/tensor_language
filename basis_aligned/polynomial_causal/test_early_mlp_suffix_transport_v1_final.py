from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import early_mlp_suffix_transport_v1_final as final
import early_mlp_suffix_transport_v1_lifecycle as lifecycle
import early_mlp_suffix_transport_v1_runtime as runtime
import early_mlp_suffix_transport_v1_statistics as statistics


def _touch(path: Path, contents: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)


def _fake_bank(value):
    if not isinstance(value, dict) or set(value) != {"tensor", "payload_sha256"} or (
        not torch.is_tensor(value["tensor"])
    ):
        raise RuntimeError("synthetic canonical bank schema changed")
    identity = runtime.logical_identity_sha256({
        "tensor": {"tensor_sha256": runtime.tensor_identity_sha256(value["tensor"])}
    })
    if value["payload_sha256"] != identity:
        raise RuntimeError("synthetic canonical bank payload hash changed")
    return {
        "payload_sha256": identity,
        "teacher_calibration": {"calibration_passed": True},
    }


def _bank() -> dict:
    tensor = torch.arange(4, dtype=torch.float32)
    return {
        "tensor": tensor,
        "payload_sha256": runtime.logical_identity_sha256({
            "tensor": {"tensor_sha256": runtime.tensor_identity_sha256(tensor)}
        }),
    }


def _stage_inputs(paths: lifecycle.ArtifactPaths) -> None:
    final_rows = torch.arange(66, dtype=torch.long).view(2, 33)
    final_cache = paths.root / "synthetic-final-rows.pt"
    torch.save(final_rows, final_cache)
    paths.rows_receipt.write_text(json.dumps({
        "entries": {
            lifecycle.ROLE_NAMES[2]: {
                "cache_path": str(final_cache),
                "cache_file_sha256": lifecycle.file_sha256(final_cache),
                "shape_full": list(final_rows.shape),
                "tensor_full_raw_sha256": lifecycle.tensor_sha256(final_rows),
            }
        }
    }))
    for path in (
        paths.rows_manifest, paths.fit_ledger, paths.fit_manifest, paths.fit_receipt,
    ):
        _touch(path)


def _source_closure() -> dict:
    return {"source_commit": "c" * 40, "source_hashes": {"source.py": "d" * 64}}


def _protected(paths: lifecycle.ArtifactPaths) -> dict:
    return lifecycle.protected_snapshot(final._required_program_snapshot_paths(paths))


def _patch_neutral_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lifecycle, "verify_source_closure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(lifecycle, "_validate_rows_receipt", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(final.programs, "validate_canonical_program_bank_payload", _fake_bank)


def test_program_bank_publication_is_create_only_and_semantically_reloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = lifecycle.ArtifactPaths(tmp_path)
    _stage_inputs(paths)
    _patch_neutral_lifecycle(monkeypatch)
    order = []
    original_torch = lifecycle.atomic_create_torch
    original_json = lifecycle.atomic_create_json

    def write_torch(value, path):
        order.append(Path(path).name)
        return original_torch(value, path)

    def write_json(value, path):
        order.append(Path(path).name)
        return original_json(value, path)

    monkeypatch.setattr(lifecycle, "atomic_create_torch", write_torch)
    monkeypatch.setattr(lifecycle, "atomic_create_json", write_json)
    lock = tmp_path / "run.lock"
    with lifecycle.exclusive_run_claim(lock) as nonce:
        receipt = final.publish_program_bank(
            _bank(), source_closure=_source_closure(), protected_before=_protected(paths),
            lock_nonce=nonce, paths=paths, lock_path=lock,
        )
        assert order == [paths.programs.name, paths.programs_receipt.name]
        assert receipt["programs"] == lifecycle.artifact_binding(paths.programs)
        unlock, payload, validated = final.load_program_bank(paths=paths)
        assert unlock == receipt
        assert torch.equal(payload["tensor"], torch.arange(4, dtype=torch.float32))
        assert validated["payload_sha256"] == _bank()["payload_sha256"]
        with pytest.raises(RuntimeError, match="ordering"):
            final.publish_program_bank(
                _bank(), source_closure=_source_closure(),
                protected_before=_protected(paths), lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )


def test_program_publisher_refuses_semantically_corrupt_reload_before_unlock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths = lifecycle.ArtifactPaths(tmp_path)
    _stage_inputs(paths)
    _patch_neutral_lifecycle(monkeypatch)
    original = lifecycle.atomic_create_torch

    def corrupt_after_write(value, path):
        original(value, path)
        changed = torch.load(path, map_location="cpu", weights_only=True)
        changed["tensor"][0] += 1
        torch.save(changed, path)

    monkeypatch.setattr(lifecycle, "atomic_create_torch", corrupt_after_write)
    lock = tmp_path / "run.lock"
    with lifecycle.exclusive_run_claim(lock) as nonce:
        with pytest.raises(RuntimeError, match="payload hash"):
            final.publish_program_bank(
                _bank(), source_closure=_source_closure(),
                protected_before=_protected(paths), lock_nonce=nonce,
                paths=paths, lock_path=lock,
            )
    assert paths.programs.is_file()
    assert not paths.programs_receipt.exists()


def _response(student: float, identity: str) -> dict:
    teacher = torch.ones(4, dtype=torch.float64)
    student_sum = torch.full((4,), student * student, dtype=torch.float64)
    dot = torch.full((4,), student, dtype=torch.float64)
    return {
        "error_sum": student_sum + teacher - 2 * dot,
        "teacher_sum": teacher,
        "student_sum": student_sum,
        "dot_sum": dot,
        "unit_identity": identity,
    }


def _transport(*, observational: bool = True) -> dict:
    identity = "e" * 64
    weights = torch.ones(32, 4, dtype=torch.float64)
    return statistics.transport_route_decision(
        code_baseline=_response(0.2, identity),
        code_candidate=_response(0.8, identity),
        logit_baseline=_response(0.2, identity),
        logit_candidate=_response(0.8, identity),
        logit_nulls=[_response(0.3, identity) for _ in range(20)],
        weights=weights, calibration_passed=True,
        observational_gates={
            name: observational for name in statistics.TRANSPORT_OBSERVATIONAL_GATES
        },
    )


def _execution() -> dict:
    return {
        "final_role_loads": 1,
        "final_evaluation_callbacks": 1,
        "outer_model_returned": True,
        "hooks_restored": True,
        "hooks_inert": True,
        "component_tree_unchanged": True,
        "student_poison_closed": True,
        "programs_reloaded_semantically": True,
        "common_support_complete": True,
        "observational_action_call_ledger_sha256": "3" * 64,
        "observational_student_outer_forwards": 68 * 48,
        "gauge_replays": 8,
        "gauge_max_abs_drift": 1e-7,
        "svd_max_abs_drift": 1e-7,
        "difference_in_differences_max_abs_drift": 1e-7,
        "row_count": 192,
        "scored_tokens_per_row": 192,
        "scored_token_count": 192 * 192,
    }


def _objective(value: bool = False) -> dict[str, bool]:
    return {name: value for name in final.OBJECTIVE_GATES}


def _program_unlock(
    paths: lifecycle.ArtifactPaths, protected: dict, source: dict,
) -> None:
    lifecycle.atomic_create_torch(_bank(), paths.programs)
    lifecycle.atomic_create_json({
        "schema_version": 1,
        "status": "frozen_programs_before_final",
        "authority": "early_mlp_suffix_transport_v1_programs_unlock",
        "authorized_for_final_scoring": True,
        "rows_receipt": lifecycle.artifact_binding(paths.rows_receipt),
        "programs": lifecycle.artifact_binding(paths.programs),
        "source_commit": source["source_commit"],
        "source_hashes": source["source_hashes"],
        "protected_before": protected,
    }, paths.programs_receipt)


def _terminal_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> tuple[lifecycle.ArtifactPaths, Path, str, dict]:
    paths = lifecycle.ArtifactPaths(tmp_path)
    _stage_inputs(paths)
    _patch_neutral_lifecycle(monkeypatch)
    source = _source_closure()
    protected = _protected(paths)
    _program_unlock(paths, protected, source)
    lock = tmp_path / "run.lock"
    claim = lifecycle.exclusive_run_claim(lock)
    nonce = claim.__enter__()
    lifecycle.write_final_attempt(
        paths=paths, source_closure=source, protected_before=protected,
        lock_nonce=nonce, lock_path=lock,
    )
    monkeypatch.setattr(lifecycle, "_FINAL_ROLE_LOADS", 1)
    bindings, _, _ = final.terminal_bindings(paths=paths)
    result = final.build_final_result(
        bindings=bindings, execution_closure=_execution(),
        objective_gates=_objective(False), transport_route=_transport(),
        numerical_payload={"raw_sufficient_statistics_sha256": "f" * 64,
                           "small_tensor": torch.arange(3)},
        expected_calibration=True,
    )
    # Keep the context manager alive until the test explicitly closes it.
    result["_test_claim"] = claim
    return paths, lock, nonce, result


def _take_claim(result: dict):
    claim = result.pop("_test_claim")
    return claim


def test_terminal_result_writes_result_manifest_then_authority_last(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths, lock, nonce, result = _terminal_setup(monkeypatch, tmp_path)
    claim = _take_claim(result)
    order = []
    original_torch = lifecycle.atomic_create_torch
    original_json = lifecycle.atomic_create_json

    def write_torch(value, path):
        order.append(Path(path).name)
        return original_torch(value, path)

    def write_json(value, path):
        order.append(Path(path).name)
        return original_json(value, path)

    monkeypatch.setattr(lifecycle, "atomic_create_torch", write_torch)
    monkeypatch.setattr(lifecycle, "atomic_create_json", write_json)
    try:
        authority = final.publish_terminal_result(
            result, lock_nonce=nonce, paths=paths, lock_path=lock,
        )
    finally:
        claim.__exit__(None, None, None)
    assert order == [paths.final_result.name, paths.final_manifest.name,
                     paths.final_authority.name]
    assert authority["status"] == "authoritative_local_route_positive"
    assert authority["objective_route_passes"] is False
    assert authority["transport_route_passes"] is True
    assert authority["authorized_for_global_ledger_credit"] is False
    assert json.loads(paths.final_manifest.read_text())["authority"] == "none"
    reloaded = torch.load(paths.final_result, map_location="cpu", weights_only=True)
    final.validate_final_result_payload(
        reloaded, expected_bindings=result["bindings"], expected_calibration=True,
    )


def test_semantic_validator_rejects_overpromotion_rank_and_integrity_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths, _, _, result = _terminal_setup(monkeypatch, tmp_path)
    claim = _take_claim(result)
    try:
        changed = deepcopy(result)
        changed["outcome_class"] = "objective_and_transport_local_positive"
        changed["payload_sha256"] = runtime.logical_identity_sha256(
            final._semantic_identity({k: v for k, v in changed.items() if k != "payload_sha256"})
        )
        with pytest.raises(RuntimeError, match="overstates"):
            final.validate_final_result_payload(changed, expected_calibration=True)

        changed = deepcopy(result)
        changed["transport_route"]["finite_null_rank"] = 2
        changed["payload_sha256"] = runtime.logical_identity_sha256(
            final._semantic_identity({k: v for k, v in changed.items() if k != "payload_sha256"})
        )
        with pytest.raises(RuntimeError, match="rank was not recomputed"):
            final.validate_final_result_payload(changed, expected_calibration=True)

        changed = deepcopy(result)
        changed["ledger_credit"]["whole_model"] = True
        changed["payload_sha256"] = runtime.logical_identity_sha256(
            final._semantic_identity({k: v for k, v in changed.items() if k != "payload_sha256"})
        )
        with pytest.raises(RuntimeError, match="global ledger"):
            final.validate_final_result_payload(changed, expected_calibration=True)

        changed = deepcopy(result)
        changed["execution_closure"]["gauge_max_abs_drift"] = 2.1e-6
        changed["payload_sha256"] = runtime.logical_identity_sha256(
            final._semantic_identity({k: v for k, v in changed.items() if k != "payload_sha256"})
        )
        with pytest.raises(RuntimeError, match="tolerance"):
            final.validate_final_result_payload(changed, expected_calibration=True)
    finally:
        claim.__exit__(None, None, None)
    assert not any(path.exists() for path in (
        paths.final_result, paths.final_manifest, paths.final_authority,
    ))


def test_corrupt_serialized_result_gets_failure_not_outcome_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths, lock, nonce, result = _terminal_setup(monkeypatch, tmp_path)
    claim = _take_claim(result)
    original = lifecycle.atomic_create_torch

    def corrupt_after_write(value, path):
        original(value, path)
        changed = torch.load(path, map_location="cpu", weights_only=True)
        changed["transport_route"]["finite_null_rank"] = 2
        changed["payload_sha256"] = runtime.logical_identity_sha256(
            final._semantic_identity({k: v for k, v in changed.items() if k != "payload_sha256"})
        )
        torch.save(changed, path)

    monkeypatch.setattr(lifecycle, "atomic_create_torch", corrupt_after_write)
    try:
        with pytest.raises(RuntimeError, match="rank was not recomputed"):
            final.publish_terminal_result(
                result, lock_nonce=nonce, paths=paths, lock_path=lock,
            )
    finally:
        claim.__exit__(None, None, None)
    assert paths.final_result.is_file()
    assert paths.integrity_failure.is_file()
    assert not paths.final_manifest.exists() and not paths.final_authority.exists()
    failure = json.loads(paths.integrity_failure.read_text())
    assert failure["authority"] == "none"
    assert failure["preserved_outputs"]["final_result"] == lifecycle.artifact_binding(
        paths.final_result
    )


def test_protected_drift_after_manifest_prevents_last_write_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    paths, lock, nonce, result = _terminal_setup(monkeypatch, tmp_path)
    claim = _take_claim(result)
    original = lifecycle.atomic_create_json

    def drift_after_manifest(value, path):
        original(value, path)
        if Path(path) == paths.final_manifest:
            paths.fit_receipt.write_text("drift")

    monkeypatch.setattr(lifecycle, "atomic_create_json", drift_after_manifest)
    try:
        with pytest.raises(RuntimeError, match="drifted"):
            final.publish_terminal_result(
                result, lock_nonce=nonce, paths=paths, lock_path=lock,
            )
    finally:
        claim.__exit__(None, None, None)
    assert paths.final_result.is_file() and paths.final_manifest.is_file()
    assert paths.integrity_failure.is_file()
    assert not paths.final_authority.exists()


def test_module_is_semantic_only_and_has_no_execution_entrypoint() -> None:
    source = Path(final.__file__).read_text()
    assert "def main(" not in source
    assert "if __name__" not in source
    assert "jacclust" not in source
    assert "huggingface" not in source.lower()
    assert "load_roles(" not in source
