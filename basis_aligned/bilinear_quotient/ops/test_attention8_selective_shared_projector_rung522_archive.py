"""CPU tests for rung-522's atomic frame archive and pre-TEST manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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


def _scheduler_payload(spec, split: str) -> dict[str, object]:
    mode = (
        "single_target_oracle" if spec.family == "target_oracle"
        else "all_three" if spec.family == "all_three"
        else "two_target"
    )
    if spec.family == "target_oracle":
        target = spec.training_targets[0]
        roles = [
            (f"{target}:member:0", target, "member", 0),
            (f"{target}:member:1", target, "member", 1),
            (f"{target}:control:0", target, "control", 0),
            (f"{target}:control:1", target, "control", 1),
        ]
    else:
        roles = [
            (f"{target}:{kind}:0", target, kind, 0)
            for target in spec.training_targets
            for kind in ("member", "control")
        ]
    base = 100 if split == "validation" else 0
    return {
        "namespace": ARCHIVE.SCHEDULER_NAMESPACE,
        "mode": mode,
        "seed": spec.seed,
        "donor_map_rule": "update_mod_4",
        "roles": [
            {
                "name": name,
                "target": target,
                "kind": kind,
                "replica": replica,
                "permutation": [base + 3 * index, base + 3 * index + 1],
            }
            for index, (name, target, kind, replica) in enumerate(roles)
        ],
    }


def _artifact(spec, frame: torch.Tensor) -> ARCHIVE.FrameArtifact:
    frame_hash = ARCHIVE.tensor_sha256(frame)
    fit_scheduler = _scheduler_payload(spec, "fit")
    validation_scheduler = _scheduler_payload(spec, "validation")
    fit_scheduler_hash = ARCHIVE._sha256_json(fit_scheduler)
    validation_scheduler_hash = ARCHIVE._sha256_json(validation_scheduler)
    fit_rows = {
        role["name"]: role["permutation"][0] for role in fit_scheduler["roles"]
    }
    validation_rows = {
        role["name"]: role["permutation"][0]
        for role in validation_scheduler["roles"]
    }
    history = [2.0 - index / 200.0 for index in range(200)]
    initial_window = sum(history[:20]) / 20
    final_window = sum(history[-20:]) / 20
    coefficient = 0.0 if spec.family == "recovery_only" else 24.0
    spec_payload = json.loads(json.dumps(asdict(spec)))
    fit_record = {
        "frame_id": spec.frame_id,
        "spec": spec_payload,
        "frame_sha256": frame_hash,
        "fit_scheduler_sha256": fit_scheduler_hash,
        "fit_batch_zero_selected_row_ids": fit_rows,
        "coefficient": coefficient,
        "optimizer": {**ARCHIVE.EXPECTED_OPTIMIZER, "control_coefficient": coefficient},
        "loss_history": history,
        "maximizing_targets": [spec.training_targets[0]] * 200,
    }
    orthonormality = float(
        (frame.mT @ frame - torch.eye(4, dtype=torch.float32)).abs().amax()
    )
    distance = ARCHIVE._float32_projector_distance(
        ARCHIVE._initial_frame(spec.seed), frame
    )
    health_record = {
        "frame_id": spec.frame_id,
        "spec": spec_payload,
        "frame_sha256": frame_hash,
        "validation_scheduler_sha256": validation_scheduler_hash,
        "validation_batch_zero_selected_row_ids": validation_rows,
        "healthy": True,
        "failures": [],
        "initial_validation_objective": 2.0,
        "final_validation_objective": 1.0,
        "initial_window_mean": initial_window,
        "final_window_mean": final_window,
        "orthonormality_error": orthonormality,
        "projector_distance_from_initialization": distance,
    }
    return ARCHIVE.FrameArtifact(
        spec=spec,
        frame=frame,
        tensor_sha256=frame_hash,
        fit_scheduler_payload=fit_scheduler,
        validation_scheduler_payload=validation_scheduler,
        fit_record_payload=fit_record,
        health_record_payload=health_record,
    )


def _artifacts() -> list[ARCHIVE.FrameArtifact]:
    result = []
    for spec in GUARD.EXPECTED_FRAME_SPECS.values():
        # Three of five all-three frames share a second projector. Their group
        # is the unique geometry-only medoid group; seed 52202 wins its tie.
        offset = 4 if spec.family == "all_three" and spec.seed >= 52202 else 0
        frame = _frame(offset)
        result.append(_artifact(spec, frame))
    return result


def _null_hashes() -> dict[int, str]:
    return {seed: _hash(f"null:{seed}") for seed in GUARD.PERMUTATION_SEEDS}


def _haar_hashes() -> dict[int, str]:
    return {seed: _hash(f"haar:{seed}") for seed in ARCHIVE.HAAR_SEEDS}


def _ledger(**changes) -> ARCHIVE.CallLedgerSnapshot:
    values = {
        "optimization_forward_events": 20_600,
        "optimization_backward_events": 20_600,
        "inference_forward_events": ARCHIVE.PRETEST_INFERENCE_TOTAL,
        "inference_by_bucket": dict(ARCHIVE.PRETEST_INFERENCE_BY_BUCKET),
        "removal_inference_forward_events": 0,
    }
    values.update(changes)
    return ARCHIVE.CallLedgerSnapshot(**values)


def _write_manifest(tmp_path: Path, archive_path: Path, **changes):
    eligible = tuple(
        spec.frame_id for spec in GUARD.EXPECTED_FRAME_SPECS.values()
        if spec.family == "all_three"
    )
    values = {
        "path": tmp_path / "pretest_manifest.json",
        "archive_path": archive_path,
        "null_hashes": _null_hashes(),
        "validation_decisions": {
            "pretest_passes": True,
            "eligible_all_three_frame_ids": eligible,
            "selection_uses": "health_eligibility_and_projector_geometry_only",
        },
        "validation_provisional_gates_passed": True,
        "haar_hashes": _haar_hashes(),
        "eligible_all_three_frame_ids": eligible,
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
        assert record.fit_scheduler_sha256 == ARCHIVE._sha256_json(
            record.fit_scheduler_payload
        )
        assert record.validation_scheduler_sha256 == ARCHIVE._sha256_json(
            record.validation_scheduler_payload
        )
        assert record.fit_scheduler_sha256 != record.validation_scheduler_sha256
        assert record.fit_record_sha256 == ARCHIVE._sha256_json(record.fit_record_payload)
        assert record.health_record_sha256 == ARCHIVE._sha256_json(
            record.health_record_payload
        )
        assert record.healthy is True
        assert record.health_failures == ()
        assert record.validation_batch_zero_selected_row_ids == {
            role["name"]: role["permutation"][0]
            for role in record.validation_scheduler_payload["roles"]
        }


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
    bad_scheduler = json.loads(json.dumps(artifacts[0].fit_scheduler_payload))
    bad_scheduler["namespace"] = "changed"
    with pytest.raises(ARCHIVE.ArchiveViolation, match="namespace changed"):
        ARCHIVE.write_frame_archive(
            tmp_path / "bad_scheduler.pt",
            [replace(artifacts[0], fit_scheduler_payload=bad_scheduler)] + artifacts[1:],
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


@pytest.mark.parametrize(
    "record_name,field,value,pattern",
    [
        ("fit_record_payload", "frame_id", "wrong", "frame_id/spec"),
        ("fit_record_payload", "frame_sha256", _hash("wrong-frame"), "frame hash"),
        ("fit_record_payload", "fit_scheduler_sha256", _hash("wrong-scheduler"),
         "scheduler hash"),
        ("health_record_payload", "healthy", False, "health state/failures"),
        ("health_record_payload", "initial_window_mean", 999.0,
         "not derived from archived data"),
    ],
)
def test_archive_recomputes_record_hashes_and_derived_health(
    tmp_path, record_name, field, value, pattern
):
    artifacts = _artifacts()
    changed = dict(getattr(artifacts[0], record_name))
    changed[field] = value
    with pytest.raises(ARCHIVE.ArchiveViolation, match=pattern):
        ARCHIVE.write_frame_archive(
            tmp_path / f"bad-{field}.pt",
            [replace(artifacts[0], **{record_name: changed})] + artifacts[1:],
        )


def test_loaded_archive_rejects_tampered_actual_record_payload(tmp_path):
    good = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(good, _artifacts())
    payload = torch.load(good, map_location="cpu", weights_only=False)
    payload["records"][0]["fit_record_payload"]["loss_history"][0] += 1.0
    tampered = tmp_path / "tampered-record.pt"
    torch.save(payload, tampered)
    with pytest.raises(ARCHIVE.ArchiveViolation, match="record hash"):
        ARCHIVE.load_frame_archive(tampered)


def test_rehashing_a_forged_record_cannot_bypass_derived_validation(tmp_path):
    good = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(good, _artifacts())
    payload = torch.load(good, map_location="cpu", weights_only=False)
    record = payload["records"][0]
    record["fit_record_payload"]["loss_history"][0] += 10.0
    record["fit_record_sha256"] = ARCHIVE._sha256_json(record["fit_record_payload"])
    record["record_sha256"] = ARCHIVE._sha256_json({
        key: value for key, value in record.items() if key != "record_sha256"
    })
    payload["content_sha256"] = ARCHIVE._sha256_json(payload["records"])
    forged = tmp_path / "forged-record.pt"
    torch.save(payload, forged)
    with pytest.raises(ARCHIVE.ArchiveViolation, match="not derived from archived data"):
        ARCHIVE.load_frame_archive(forged)


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
    assert len(payload["frame_records"]) == 103
    assert len(payload["frame_health_states"]) == 103
    assert all(value["healthy"] for value in payload["frame_health_states"].values())
    assert payload["call_ledger"]["inference_forward_events"] == 5_029
    assert payload["call_ledger"]["inference_by_bucket"] == (
        ARCHIVE.PRETEST_INFERENCE_BY_BUCKET
    )
    receipt.protocol_state.open_test_once()
    receipt.protocol_state.authorize_split_access("TEST")


def test_validation_decision_is_authoritative_and_may_exclude_healthy_all_three(tmp_path):
    archive_path = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(archive_path, _artifacts())
    eligible = tuple(f"all_three:{seed}" for seed in range(52_201, 52_205))
    receipt = _write_manifest(
        tmp_path,
        archive_path,
        validation_decisions={
            "pretest_passes": True,
            "eligible_all_three_frame_ids": eligible,
            "all_three_frames": "full A-style decision payload retained",
        },
        eligible_all_three_frame_ids=eligible,
    )
    assert receipt.pretest_freeze.eligible_all_three_frame_ids == eligible
    assert receipt.selected_all_three_frame_id == "all_three:52202"


def test_validation_decision_dataclass_is_canonicalized_and_caller_must_match(tmp_path):
    @dataclass(frozen=True)
    class Decision:
        eligible_all_three_frame_ids: tuple[str, ...]
        pretest_passes: bool

    archive_path = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(archive_path, _artifacts())
    eligible = tuple(f"all_three:{seed}" for seed in range(52_202, 52_205))
    receipt = _write_manifest(
        tmp_path,
        archive_path,
        validation_decisions=Decision(eligible, True),
        eligible_all_three_frame_ids=eligible,
    )
    stored = json.loads(receipt.path.read_text())["validation_decisions"]
    assert stored == {
        "eligible_all_three_frame_ids": list(eligible),
        "pretest_passes": True,
    }


def test_validation_decision_cannot_mark_an_archived_unhealthy_frame_eligible(tmp_path):
    artifacts = _artifacts()
    index = next(
        index for index, artifact in enumerate(artifacts)
        if artifact.spec.frame_id == "all_three:52200"
    )
    health = dict(artifacts[index].health_record_payload)
    health.update({
        "healthy": False,
        "failures": ["validation_not_better_than_initialization"],
        "final_validation_objective": health["initial_validation_objective"],
    })
    artifacts[index] = replace(artifacts[index], health_record_payload=health)
    archive_path = tmp_path / "frames.pt"
    ARCHIVE.write_frame_archive(archive_path, artifacts)
    eligible = tuple(f"all_three:{seed}" for seed in range(52_200, 52_205))
    with pytest.raises(ARCHIVE.ArchiveViolation, match="unhealthy all-three"):
        _write_manifest(
            tmp_path,
            archive_path,
            validation_decisions={
                "pretest_passes": True,
                "eligible_all_three_frame_ids": eligible,
            },
            eligible_all_three_frame_ids=eligible,
        )


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
         "disagree with validation decision"),
        ({"fit_mu_q_source_split": "VALIDATION"}, "FIT only"),
        ({"fit_mu_q": torch.zeros(5, dtype=torch.float32)}, "length four"),
        ({"call_ledger": _ledger(optimization_forward_events=20_599)},
         "exactly 20,600"),
        ({"call_ledger": _ledger(removal_inference_forward_events=1)},
         "removal inference is illegal"),
        ({"call_ledger": _ledger(inference_forward_events=5_028)},
         "exactly 5,029"),
        ({"call_ledger": _ledger(inference_by_bucket={
            **ARCHIVE.PRETEST_INFERENCE_BY_BUCKET,
            "native_capture": 130,
            "native_replay": 132,
        })}, "bucket ledger changed"),
        ({
            "validation_provisional_gates_passed": False,
            "validation_decisions": {
                "pretest_passes": False,
                "eligible_all_three_frame_ids": ["all_three:52200"],
            },
        }, "pretest_passes=False"),
        ({
            "validation_decisions": {
                "pretest_passes": False,
                "eligible_all_three_frame_ids": ["all_three:52200"],
            },
        }, "separate pass flag disagrees"),
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
