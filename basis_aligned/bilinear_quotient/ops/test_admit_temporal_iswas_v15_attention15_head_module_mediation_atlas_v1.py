import json

import pytest

import admit_temporal_iswas_v15_attention15_head_module_mediation_atlas_v1 as gate


def payload(predictions=None):
    values = {key: True for key in gate.REQUIRED}
    values.update(predictions or {})
    return json.dumps({
        "candidate_id": "temporal_auxiliary.iswas_v15_construction_oracle_attention15_dependency_factorial_v1",
        "predictions": values,
    }, sort_keys=True).encode()


def prior():
    return json.dumps({
        "candidate_id": "temporal_auxiliary.iswas_v15_attention15_head_module_mediation_atlas_v1"
    }).encode()


def test_all_required_gates_admit(monkeypatch):
    runner = b"reviewed runner"
    monkeypatch.setattr(gate, "EXPECTED_RUNNER_SHA256", gate.digest(runner))
    result = gate.build_manifest(payload(), runner, prior())
    assert result["admitted"] is True
    assert result["disposition"] == "admit_head_module_atlas"
    assert result["model_forwards"] == 0


@pytest.mark.parametrize("failed", gate.REQUIRED)
def test_each_required_gate_closes_branch(monkeypatch, failed):
    runner = b"reviewed runner"
    monkeypatch.setattr(gate, "EXPECTED_RUNNER_SHA256", gate.digest(runner))
    result = gate.build_manifest(payload({failed: False}), runner, prior())
    assert result["admitted"] is False
    assert result["disposition"] == "close_weight_reader_branch"


def test_missing_gate_does_not_admit(monkeypatch):
    runner = b"reviewed runner"
    monkeypatch.setattr(gate, "EXPECTED_RUNNER_SHA256", gate.digest(runner))
    body = json.loads(payload())
    del body["predictions"][gate.REQUIRED[1]]
    result = gate.build_manifest(json.dumps(body).encode(), runner, prior())
    assert result["admitted"] is False


def test_changed_runner_fails_closed():
    with pytest.raises(gate.AdmissionError):
        gate.build_manifest(payload(), b"changed", prior())
