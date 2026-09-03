"""CPU tests for rung-522's atomic frame archive and pre-TEST manifest."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    path = OPS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GUARD = _load("attention8_selective_shared_projector_rung522_state_guard")
ARCHIVE = _load("attention8_selective_shared_projector_rung522_archive")


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _frame(offset: int = 0) -> torch.Tensor:
    frame = torch.zeros(1152, 4, dtype=torch.float32)
    frame[offset : offset + 4] = torch.eye(4, dtype=torch.float32)
    return frame


def _artifacts() -> list[ARCHIVE.FrameArtifact]:
    result = []
    for spec in GUARD.EXPECTED_FRAME_SPECS.values():
        # Three of five all-three frames share a second projector. Their group
        # is the unique geometry-only medoid group; seed 52202 wins its tie.
        offset = 4 if spec.family == "all_three" and spec.seed >= 52202 else 0
        frame = _frame(offset)
        result.append(ARCHIVE.FrameArtifact(
            spec=spec,
            frame=frame,
            tensor_sha256=ARCHIVE.tensor_sha256(frame),
            scheduler_sha256=_hash("scheduler:" + spec.frame_id),
            fit_record_sha256=_hash("fit:" + spec.frame_id),
            health_record_sha256=_hash("health:" + spec.frame_id),
        ))
    return result


def _null_hashes() -> dict[int, str]:
    return {seed: _hash(f"null:{seed}") for seed in GUARD.PERMUTATION_SEEDS}


def _haar_hashes() -> dict[int, str]:
    return {seed: _hash(f"haar:{seed}") for seed in ARCHIVE.HAAR_SEEDS}


def _ledger(**changes) -> ARCHIVE.CallLedgerSnapshot:
    values = {
        "optimization_forward_events": 20_600,
        "optimization_backward_events": 20_600,
        "inference_forward_events": 638,
        "removal_inference_forward_events": 0,
    }
    values.update(changes)
    return ARCHIVE.CallLedgerSnapshot(**values)


def _write_manifest(tmp_path: Path, archive_path: Path, **changes):
    values = {
        "path": tmp_path / "pretest_manifest.json",
        "archive_path": archive_path,
        "null_hashes": _null_hashes(),
        "validation_decisions": {
            "provisional_gates_passed": True,
            "selection_uses": "health_eligibility_and_projector_geometry_only",
        },
        "validation_provisional_gates_passed": True,
        "haar_hashes": _haar_hashes(),
        "eligible_all_three_frame_ids": tuple(
            spec.frame_id for spec in GUARD.EXPECTED_FRAME_SPECS.values()
            if spec.family == "all_three"
        ),
        "fit_mu_q": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
        "fit_mu_q_source_split": "FIT",
        "call_ledger": _ledger(),
        "fingerprint_definition_sha256": _hash("fingerprint-definition"),
        "test_sweep_plan_sha256": _hash("test-sweep-plan"),
    }
    values.update(changes)
    return ARCHIVE.write_pretest_manifest(**values)


def test_archive_atomically_round_trips_exactly_103_validated_frames(tmp_path):
    path = tmp_path / "frames.pt"
    receipt = ARCHIVE.write_frame_archive(path, _artifacts())
    assert path.is_file()
    assert len(receipt.frames) == len(receipt.records) == 103
    assert set(receipt.frames) == set(GUARD.EXPECTED_FRAME_SPECS)
    assert receipt.file_sha256 == ARCHIVE._sha256_file(path)
    assert not list(tmp_path.glob("*.tmp.*"))
    loaded = ARCHIVE.load_frame_archive(path)
    assert loaded.content_sha256 == receipt.content_sha256
    for frame_id, record in loaded.records.items():
        assert record.spec == GUARD.EXPECTED_FRAME_SPECS[frame_id]
        assert record.tensor_sha256 == ARCHIVE.tensor_sha256(loaded.frames[frame_id])


def test_archive_refuses_overwrite_duplicates_and_incomplete_census(tmp_path):
    path = tmp_path / "frames.pt"
    artifacts = _artifacts()
    ARCHIVE.write_frame_archive(path, artifacts)
    with pytest.raises(FileExistsError, match="overwrite"):
        ARCHIVE.write_frame_archive(path, artifacts)
    duplicate = artifacts[:-1] + [artifacts[0]]
    with pytest.raises(ARCHIVE.ArchiveViolation, match="duplicate"):
        ARCHIVE.write_frame_archive(tmp_path / "duplicate.pt", duplicate)
    with pytest.raises(ARCHIVE.ArchiveViolation, match="exactly 103"):
        ARCHIVE.write_frame_archive(tmp_path / "short.pt", artifacts[:-1])


def test_archive_rejects_tensor_hash_spec_scheduler_and_geometry_mismatch(tmp_path):
    artifacts = _artifacts()
    with pytest.raises(ARCHIVE.ArchiveViolation, match="claimed tensor hash"):
        ARCHIVE.write_frame_archive(
            tmp_path / "bad_hash.pt",
            [replace(artifacts[0], tensor_sha256=_hash("wrong"))] + artifacts[1:],
        )
    with pytest.raises(ARCHIVE.ArchiveViolation, match="scheduler_sha256"):
        ARCHIVE.write_frame_archive(
            tmp_path / "bad_scheduler.pt",
            [replace(artifacts[0], scheduler_sha256="bad")] + artifacts[1:],
        )
    bad_spec = replace(
        artifacts[0].spec,
        health_targets=artifacts[0].spec.health_targets + (GUARD.RESERVED_FOURTH_TARGET,),
    )
    with pytest.raises(ARCHIVE.ArchiveViolation, match="specification differs"):
        ARCHIVE.write_frame_archive(
            tmp_path / "bad_spec.pt",
            [replace(artifacts[0], spec=bad_spec)] + artifacts[1:],
        )
    nonorthogonal = artifacts[0].frame.clone()
    nonorthogonal[0, 0] = 2
    with pytest.raises(ARCHIVE.ArchiveViolation, match="orthonormality"):
        ARCHIVE.write_frame_archive(
            tmp_path / "bad_geometry.pt",
            [replace(
                artifacts[0],
                frame=nonorthogonal,
                tensor_sha256=ARCHIVE.tensor_sha256(nonorthogonal),
            )] + artifacts[1:],
        )


def test_loaded_archive_detects_tampered_tensor_bytes(tmp_path):
    good = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(good, _artifacts())
    payload = torch.load(good, map_location="cpu", weights_only=False)
    frame_id = next(iter(payload["frames"]))
    payload["frames"][frame_id][0, 0] += 0.25
    tampered = tmp_path / "tampered.pt"
    torch.save(payload, tampered)
    with pytest.raises(ARCHIVE.ArchiveViolation, match="orthonormality|tensor bytes"):
        ARCHIVE.load_frame_archive(tampered)


def test_manifest_binds_every_pretest_artifact_and_returns_guard_freeze(tmp_path):
    archive_path = tmp_path / "frames.pt"
    archive = ARCHIVE.write_frame_archive(archive_path, _artifacts())
    receipt = _write_manifest(tmp_path, archive_path)
    assert isinstance(receipt.pretest_freeze, GUARD.PretestFreeze)
    assert receipt.selected_all_three_frame_id == "all_three:52202"
    assert receipt.selected_all_three_seed == 52202
    assert receipt.file_sha256 == ARCHIVE._sha256_file(receipt.path)
    payload = json.loads(receipt.path.read_text())
    assert payload["schema"] == ARCHIVE.MANIFEST_SCHEMA
    assert payload["test_opened"] is False
    assert payload["post_test_fitting_allowed"] is False
    assert payload["frame_archive"]["file_sha256"] == archive.file_sha256
    assert len(payload["label_null_sha256"]) == 16
    assert len(payload["haar_sha256"]) == 20
    assert payload["fit_mu_q"]["source_split"] == "FIT"
    assert payload["fit_mu_q"]["selected_frame_id"] == "all_three:52202"
    assert payload["registered_contract_sha256"] == GUARD.registered_contract_sha256()
    assert payload["pretest_freeze"]["selected_final_frame_id"] == "all_three:52202"
    receipt.protocol_state.open_test_once()
    receipt.protocol_state.authorize_split_access("TEST")


def test_geometry_medoid_uses_projectors_and_lower_seed_tiebreak(tmp_path):
    archive = ARCHIVE.write_frame_archive(tmp_path / "frames.pt", _artifacts())
    eligible = tuple(
        frame_id for frame_id, record in archive.records.items()
        if record.spec.family == "all_three"
    )
    selected, decision = ARCHIVE.geometry_only_grassmann_medoid(archive, reversed(eligible))
    assert selected == "all_three:52202"
    assert decision["rule"] == "grassmann_medoid_lower_seed_tiebreak"
    assert decision["selected_seed"] == 52202
    assert decision["selection_targets"] == list(GUARD.FITTED_TARGETS)
    assert len(decision["sha256"]) == 64


@pytest.mark.parametrize(
    "changes,pattern",
    [
        ({"null_hashes": {}}, "label-null hashes must contain exactly"),
        ({"haar_hashes": {}}, "Haar hashes must contain exactly"),
        ({"eligible_all_three_frame_ids": ("real_leave_one_out:r.2.0.2:52200",)},
         "non-all-three"),
        ({"fit_mu_q_source_split": "VALIDATION"}, "FIT only"),
        ({"fit_mu_q": torch.zeros(5, dtype=torch.float32)}, "length four"),
        ({"call_ledger": _ledger(optimization_forward_events=20_599)},
         "exactly 20,600"),
        ({"call_ledger": _ledger(removal_inference_forward_events=1)},
         "removal inference is illegal"),
        ({"validation_provisional_gates_passed": False}, "VALIDATION gates failed"),
        ({"validation_provisional_gates_passed": "yes"}, "must be boolean"),
        ({"fingerprint_definition_sha256": "bad"}, "not a lowercase SHA-256"),
    ],
)
def test_manifest_fails_closed_on_missing_or_changed_inputs(tmp_path, changes, pattern):
    archive_path = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(archive_path, _artifacts())
    with pytest.raises((ARCHIVE.ArchiveViolation, GUARD.ProtocolViolation), match=pattern):
        _write_manifest(tmp_path, archive_path, **changes)
    assert not (tmp_path / "pretest_manifest.json").exists()


def test_manifest_refuses_overwrite(tmp_path):
    archive_path = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(archive_path, _artifacts())
    _write_manifest(tmp_path, archive_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_manifest(tmp_path, archive_path)


def test_archive_module_has_no_model_data_or_cuda_imports():
    source = (OPS / "attention8_selective_shared_projector_rung522_archive.py").read_text()
    assert "bilin18_observed_model_facade" not in source
    assert "census_state_diverse.pt" not in source
    assert "curated_rows.pt" not in source
    assert ".cuda(" not in source
