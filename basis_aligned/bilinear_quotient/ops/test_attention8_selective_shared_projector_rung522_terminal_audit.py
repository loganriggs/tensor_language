"""CPU tests for the independent rung-522 terminal auditor."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


OPS = Path(__file__).parent
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load("attention8_selective_shared_projector_rung522_terminal_audit")


def _fixture():
    records = {}
    health = {}
    for family, count in (
        ("real_leave_one_out", 15),
        ("recovery_only", 15),
        ("target_oracle", 20),
        ("label_null", 48),
        ("all_three", 5),
    ):
        for index in range(count):
            frame_id = f"{family}:{index}"
            record = SimpleNamespace(
                spec=SimpleNamespace(family=family),
                healthy=False,
                health_failures=("validation_not_better_than_initialization",),
                tensor_sha256=f"{index:064x}"[-64:],
            )
            records[frame_id] = record
            health[frame_id] = {
                "healthy": False,
                "failures": ["validation_not_better_than_initialization"],
                "frame_sha256": record.tensor_sha256,
            }
    loaded = SimpleNamespace(
        file_sha256="a" * 64,
        content_sha256="b" * 64,
        records=records,
    )
    ledger = {
        "optimization_forwards": 20_600,
        "optimization_backwards": 20_600,
        "inference_forwards": AUDIT.EXPECTED_INFERENCE,
        "removal_forwards": 0,
        "inference_by_bucket": AUDIT.EXPECTED_BUCKETS,
    }
    result = {
        "rung": 522,
        "status": "terminal_pretest_validation_failure",
        "test_opened": False,
        "test_closed": False,
        "pretest_manifest_created": False,
        "predictions": {"a": False, "b": False, "c": None, "d": None},
        "provisional_validation_decision": {"pretest_passes": False},
        "pretest_call_ledger": ledger,
        "execution_price": {**ledger, "runtime_seconds": 1.0},
        "frame_archive": {
            "file_sha256": loaded.file_sha256,
            "content_sha256": loaded.content_sha256,
            "frame_count": 103,
        },
        "frame_health": health,
    }
    return result, loaded


def test_expected_pretest_failure_matches_archive_and_ledger():
    result, loaded = _fixture()
    audit = AUDIT.audit_terminal_result(result, loaded)
    assert audit["passes"]
    assert audit["healthy_fit_count"] == 0
    assert audit["family_health"]["real_leave_one_out"]["total"] == 15
    assert audit["exact_call_ledger"]["inference_forwards"] == 5_029


def test_test_stage_fields_are_rejected():
    result, loaded = _fixture()
    result["test_outputs"] = {}
    with pytest.raises(ValueError, match="TEST-stage fields"):
        AUDIT.audit_terminal_result(result, loaded)


def test_healthy_real_fit_is_rejected_at_expected_invalid_boundary():
    result, loaded = _fixture()
    frame_id = "real_leave_one_out:0"
    loaded.records[frame_id].healthy = True
    result["frame_health"][frame_id]["healthy"] = True
    with pytest.raises(ValueError, match="a real fit is healthy"):
        AUDIT.audit_terminal_result(result, loaded)
