"""CPU tests for the independent rung-523 terminal auditor."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load("attention8_projector_optimizer_repair_rung523_terminal_audit")
REPAIR = _load("attention8_projector_optimizer_repair_rung523_math")
ARCHIVE = _load("attention8_selective_shared_projector_rung522_archive")


def _fixture():
    frames = {}
    artifact_records = {}
    diagnostics = {}
    cells = {}
    for arm_index, arm in enumerate(REPAIR.PROSPECTIVE_ARMS):
        records = []
        diagnostics[arm] = {}
        for fit_index in range(15):
            frame_id = f"real_leave_one_out:r.2.{fit_index % 3}.{fit_index // 5}:{52200 + fit_index % 5}"
            key = f"{arm}:{frame_id}"
            frame = torch.zeros(1152, 4)
            frame[arm_index, 0] = fit_index + 1
            frames[key] = frame
            record = {
                "arm": arm,
                "frame_id": frame_id,
                "omitted_target": "r.2.0.2",
                "seed": 52200 + fit_index % 5,
                "scale_mode": "fixed_target_map" if arm.startswith("fixed") else "row_specific",
                "learning_rate": 0.003 if arm != REPAIR.FIXED_HIGH_LR else 0.03,
                "frame_sha256": ARCHIVE.tensor_sha256(frame),
                "fit_scheduler_sha256": "a" * 64,
                "validation_scheduler_sha256": "b" * 64,
                "loss_history": tuple(2.0 - 0.005 * update for update in range(200)),
                "maximizing_targets": tuple("r.2.1.1" for _ in range(200)),
                "initial_common_validation": 2.0,
                "final_common_validation": 1.0,
                "orthonormality_error": 1e-7,
                "projector_distance": 1.0,
                "finite_gradients": True,
                "model_gradients_absent": True,
            }
            records.append(record)
            diagnostics[arm][frame_id] = {"digest": "c" * 64}
        # Make every arm fail exactly one fit's common validation rule.
        records[0]["final_common_validation"] = 2.1
        artifact_records[arm] = records
        cells[arm] = REPAIR.score_candidate_cell([AUDIT._health(record) for record in records])
    artifact = {
        "schema": "attention8-projector-optimizer-repair-rung523-frames-v1",
        "frames": frames,
        "records": artifact_records,
        "diagnostics": diagnostics,
    }
    decision = REPAIR.adoption_decision(cells)
    result = {
        "rung": 523,
        "status": "raw_adam_through_qr_closed",
        "test_accessible": False,
        "test_opened": False,
        "omitted_targets_evaluated": False,
        "claim_level": "FIT/VALIDATION-only optimizer calibration; no circuit evidence",
        "seal": {
            "requested_splits": ["FIT", "VALIDATION", "VALIDATION"],
            "test_split_requested": False,
            "scientific_a_through_d_scored": False,
        },
        "execution_price": {**AUDIT.EXPECTED_LEDGER, "runtime_seconds": 1.0},
        "frame_artifact": {
            "file_sha256": "d" * 64,
            "frame_count": 45,
        },
        "candidate_records": AUDIT._json_value(artifact_records),
        "candidate_diagnostics": diagnostics,
        "prospective_cells": cells,
        "decision": decision,
        "registered_predictions": {
            "pred_a_fixed_scale_high_lr_passes": False,
            "pred_b_row_specific_low_lr_passes": False,
            "pred_c_fixed_scale_low_lr_passes": False,
        },
    }
    return result, artifact


def test_terminal_closure_recomputes_frames_scores_and_decision():
    result, artifact = _fixture()
    audit = AUDIT.audit_terminal_result(result, artifact, artifact_file_sha256="d" * 64)
    assert audit["passes"]
    assert audit["frame_count"] == 45
    assert audit["decision"]["adopted_arm"] is None
    assert set(audit["arm_summaries"]) == set(REPAIR.PROSPECTIVE_ARMS)


def test_opened_test_is_rejected():
    result, artifact = _fixture()
    result["test_opened"] = True
    with pytest.raises(ValueError, match="test_opened"):
        AUDIT.audit_terminal_result(result, artifact, artifact_file_sha256="d" * 64)


def test_tampered_frame_is_rejected():
    result, artifact = _fixture()
    key = next(iter(artifact["frames"]))
    artifact["frames"][key][100, 1] = 1.0
    with pytest.raises(ValueError, match="frame hash differs"):
        AUDIT.audit_terminal_result(result, artifact, artifact_file_sha256="d" * 64)


def test_tampered_score_is_rejected():
    result, artifact = _fixture()
    result["prospective_cells"][REPAIR.FIXED_LOW_LR]["passing_fit_count"] = 15
    with pytest.raises(ValueError, match="stored score differs"):
        AUDIT.audit_terminal_result(result, artifact, artifact_file_sha256="d" * 64)
