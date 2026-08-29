from pathlib import Path

import pytest
import torch

import mlp2_error_rayleigh_collector_core as core
import run_mlp2_error_rayleigh_v1_collect as collect


def test_arm_and_call_census_is_exact():
    assert collect.PROGRAM_NAMES == ("FULL512", "CONTINUE512", "ROBUST512")
    assert collect.BACKGROUND_NAMES == ("NATIVE", "C512")
    calls = collect.expected_calls()
    assert calls["outer_forwards"] == 688
    assert calls["native_mlp2_calls"] == 640
    assert calls["direct_program_calls"] == 48
    assert calls["offline_program_calls"] == 48
    assert calls["c512_calls"] == 344


def test_role_namespaces_are_disjoint_and_heldout_has_unlock_gate(tmp_path, monkeypatch):
    design, heldout = collect.role_paths("DESIGN"), collect.role_paths("HELDOUT")
    assert set(design.values()).isdisjoint(heldout.values())
    monkeypatch.setattr(collect, "PREDICTOR_RECEIPT", tmp_path / "absent.json")
    monkeypatch.setattr(collect, "role_paths", lambda _role: {
        "authority": tmp_path / "authority", "ledger": tmp_path / "ledger",
        "receipt": tmp_path / "receipt", "failure": tmp_path / "failure",
        "lock": tmp_path / "lock",
    })
    with pytest.raises(RuntimeError, match="remains locked"):
        collect.run("HELDOUT")


def test_ledger_schema_accepts_exact_replay_and_rejects_nonexact():
    features = torch.ones(3, 2, 3, 32, len(core.FEATURE_NAMES), dtype=torch.float64)
    finite = torch.zeros(3, 2, 32, len(core.FINITE_NAMES), dtype=torch.float64)
    finite[..., 5:] = 1
    value = {
        "schema": "mlp2_error_rayleigh_v1_role_ledger", "role": "DESIGN",
        "features": features, "finite": finite,
        "axes": {"programs": list(collect.PROGRAM_NAMES),
                 "backgrounds": list(collect.BACKGROUND_NAMES),
                 "controls": list(core.CONTROL_NAMES),
                 "features": list(core.FEATURE_NAMES),
                 "finite": list(core.FINITE_NAMES), "documents": 32},
        "control_hashes": {}, "calls": collect.expected_calls(),
        "authority_sha256": "a", "checkpoint": {},
    }
    assert collect.validate_ledger(value, "a", "DESIGN") is value
    value["finite"] = finite.clone(); value["finite"][0, 0, 0, 5] = 0
    with pytest.raises(RuntimeError, match="ledger tensors"):
        collect.validate_ledger(value, "a", "DESIGN")


def test_source_closure_contains_direct_science_and_tests():
    for path in (collect.PREREG, collect.ADDENDUM, collect.RUNNER, collect.TEST,
                 collect.CORE, collect.CORE_TEST,
                 collect.HERE / "mlp2_error_rayleigh_metrics.py",
                 collect.HERE / "test_mlp2_error_rayleigh_metrics.py"):
        assert collect.SOURCE_PATHS.count(path) == 1


def test_row_receipt_contract_rejects_role_leak():
    value = {
        "schema": "mlp2_error_rayleigh_v1_rows",
        "status": "fresh_roles_frozen_before_any_model_or_training_access",
        "selection": {"start_document_index": 121000, "documents_per_role": 32,
                      "token_length": 257, "scored_slice": [64, 256]},
        "roles": {
            "DESIGN": {"authorized_for_training": True, "authorized_for_evaluation": False},
            "HELDOUT": {"authorized_for_training": False, "authorized_for_evaluation": True},
        },
        "outcome_access": {"model_loaded": False, "training_run": False},
        "entries": {"DESIGN": {}, "HELDOUT": {}},
        "provenance": {"DESIGN": [{}] * 32, "HELDOUT": [{}] * 32},
        "disjointness": {"all": True},
    }
    # Entry validation deliberately reaches the path only after every semantic role gate.
    value["roles"]["HELDOUT"]["authorized_for_training"] = True
    with pytest.raises(RuntimeError, match="semantics changed"):
        collect.validate_row_receipt(value)
