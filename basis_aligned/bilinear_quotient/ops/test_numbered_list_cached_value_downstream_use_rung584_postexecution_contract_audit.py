"""Post-execution contract attacks for the exact committed R584 producer.

These tests do not run the model.  The positive test mechanically reconstructs
the landed result with the independent R588 auditor.  The strict xfails record
producer-boundary defects that cannot be repaired retrospectively without a new
result namespace and a fresh independent review.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest


OPS = Path(__file__).resolve().parent
ROOT = OPS.parent
RUNNER = OPS / "numbered_list_cached_value_downstream_use_rung584.py"
AUDITOR = OPS / "audit_numbered_list_cached_value_downstream_use_rung588.py"
RESULT = ROOT / "numbered_list_cached_value_downstream_use_rung584_results.json"

RUNNER_SHA256 = "50609756d97de2f13f717774f13d72b1c743f38a172375e9b08efc2b055336c7"
RESULT_SHA256 = "7980753636fab422ed6c609a1afd054f99ed7f903e2bb3e61eddf0617316fdf6"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


r584 = _load("r584_postexecution_contract_target", RUNNER)
r588 = _load("r588_postexecution_reconstructor", AUDITOR)


@pytest.fixture(scope="module")
def landed_result():
    assert r588.sha256(RUNNER) == RUNNER_SHA256
    assert r588.sha256(RESULT) == RESULT_SHA256
    return r588.strict_loads(RESULT.read_bytes(), "R584 landed result")


def _producer_validate(result):
    rows = r584.load_authority()
    return r584.validate_scientific_result(
        result,
        rows,
        expected_forwards=result["model_forwards"],
        expected_provenance=result["input_sha256"],
    )


def test_landed_result_mechanically_reconstructs_under_r588(landed_result):
    audit = r588.audit_payload(landed_result)
    assert audit["audit_verdict"] == "held_independent_audit"
    assert audit["audit_failures"] == []
    assert audit["independently_recomputed_decision"] == landed_result["decision"]
    assert audit["independently_recomputed_opened_splits"] == landed_result["evaluated_splits"]
    assert audit["independently_recomputed_model_forwards"] == landed_result["model_forwards"]
    assert audit["raw_counts"] == {
        "fit_capture": 576,
        "fit_real_arms": 12,
        "fit_null_arms": 0,
        "select_real_arms": 0,
        "select_null_arms": 0,
    }
    assert audit["bootstrap_cell_count"] == 432


@pytest.mark.xfail(
    strict=True,
    reason="R584 validates capture membership but does not derive reports or terminal fields",
)
def test_producer_rejects_report_and_terminal_mutation(landed_result):
    changed = dict(landed_result)
    changed["fit_reports"] = {}
    changed["decision"] = "downstream_use_component_held"
    changed["next_step"] = "invented_postexecution_action"
    with pytest.raises((RuntimeError, r584.result_contract.ContractError)):
        _producer_validate(changed)


@pytest.mark.xfail(
    strict=True,
    reason="R584 does not recompute exactness or reject an invalid instrument before publication",
)
def test_producer_rejects_unretained_exactness_failure_as_scientific_result(landed_result):
    changed = dict(landed_result)
    changed["fit_capture_raw"] = list(landed_result["fit_capture_raw"])
    changed["fit_capture_raw"][0] = copy.deepcopy(changed["fit_capture_raw"][0])
    changed["fit_capture_raw"][0]["native_replay_relative_squared_error_by_row"] = {
        "source_present": 1.0,
        "source_deleted": 1.0,
        "maximum": 1.0,
    }
    changed["fit_exactness"] = {
        key: 1.0 for key in landed_result["fit_exactness"]
    }
    with pytest.raises((RuntimeError, r584.result_contract.ContractError)):
        _producer_validate(changed)


@pytest.mark.xfail(
    strict=True,
    reason="R584 publishes one result with write_text and has no atomic result/receipt package",
)
def test_producer_has_atomic_mutually_bound_result_and_receipt_namespace():
    source = RUNNER.read_text()
    assert hasattr(r584, "RESULT_RECEIPT")
    assert "OUT.write_text" not in source
    assert "os.replace" in source


def test_exact_committed_result_is_strict_finite_json(landed_result):
    # strict_loads rejects duplicate keys, NaN, and infinity.  Re-encoding with
    # allow_nan=False also proves every nested value is standard JSON.
    json.dumps(landed_result, allow_nan=False)
