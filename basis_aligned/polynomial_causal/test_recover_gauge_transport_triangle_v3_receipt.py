import copy
import json

import pytest
import torch

import recover_gauge_transport_triangle_v3_receipt as recovery


def _production_payloads():
    return (
        json.loads(recovery.RESULT.read_text()),
        torch.load(recovery.STATE, map_location="cpu", weights_only=True),
        json.loads(recovery.V2_FAILURE.read_text()),
        json.loads(recovery.V2_AUTHORITY.read_text()),
    )


def test_complete_partial_result_and_state_validate():
    recovery.validate_payloads(*_production_payloads())


def test_changed_scientific_decision_is_rejected():
    result, state, failure, authority = _production_payloads()
    changed = copy.deepcopy(result)
    changed["decisions"]["projected_u14_sufficient"] = True
    with pytest.raises(RuntimeError, match="result semantics changed"):
        recovery.validate_payloads(changed, state, failure, authority)


def test_create_only_receipt_refuses_overwrite(tmp_path):
    path = tmp_path / "receipt.json"
    recovery.create_only_json(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        recovery.create_only_json(path, {"status": "second"})
