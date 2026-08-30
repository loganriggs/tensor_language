from pathlib import Path
import json
from types import SimpleNamespace

import pytest
import torch

import mlp2_error_rayleigh_collector_core as core
import run_mlp2_error_rayleigh_v1_collect as collect


def valid_control_hashes(role="DESIGN"):
    output = {}
    for pi, program in enumerate(collect.PROGRAM_NAMES):
        for bi, background in enumerate(collect.BACKGROUND_NAMES):
            output[f"{program}|{background}"] = {
                "seed": collect.control_seed(role, pi, bi),
                "bindings": {name: "a" * 64 for name in (
                    "mlp2_state", "native_write", "candidate_write",
                )},
                "errors": {name: "b" * 64 for name in core.CONTROL_NAMES},
            }
    return output


def mocked_transaction(tmp_path, monkeypatch, *, checkpoint="checkpoint"):
    paths = {
        name: tmp_path / f"{name}.artifact" for name in (
            "authority", "ledger", "receipt", "failure", "lock",
        )
    }
    rows = torch.zeros(32, 257, dtype=torch.long)
    row_path = tmp_path / "rows.pt"; torch.save(rows, row_path)
    row_receipt_path = tmp_path / "rows.json"
    row_receipt = {"entries": {"DESIGN": {
        "path": str(row_path), "file_sha256": collect.file_sha256(row_path),
        "tensor_sha256": collect.row_life.base.tensor_sha256(rows),
    }}}
    row_receipt_path.write_text(json.dumps(row_receipt))
    checkpoint_value = {"name": "checkpoint"}
    parent = {
        "parents": {}, "program_integrity": {},
        "row_receipt_sha256": collect.file_sha256(row_receipt_path),
        "checkpoint": checkpoint_value,
    }
    features = torch.ones(3, 2, 3, 32, len(core.FEATURE_NAMES), dtype=torch.float64)
    finite = torch.zeros(3, 2, 32, len(core.FINITE_NAMES), dtype=torch.float64)
    finite[..., 5:] = 1
    monkeypatch.setattr(collect, "role_paths", lambda _role: paths)
    monkeypatch.setattr(collect, "ROWS_RECEIPT", row_receipt_path)
    monkeypatch.setattr(collect, "source_hashes", lambda _commit: {})
    monkeypatch.setattr(collect, "validate_audit", lambda _sources: (
        {"reviewer": "mock-independent-audit"}, "c" * 64,
    ))
    monkeypatch.setattr(collect, "validate_row_receipt", lambda value: value)
    monkeypatch.setattr(collect, "parent_snapshot", lambda: parent)
    monkeypatch.setattr(collect, "protected_snapshot", lambda authority: authority["parent_snapshot"])
    monkeypatch.setattr(collect.row_life.base, "acquire_claim", lambda _path: object())
    monkeypatch.setattr(collect.row_life.base, "require_claim", lambda _claim, _path: None)
    monkeypatch.setattr(collect.row_life.base, "release_claim", lambda _claim, _path: None)
    monkeypatch.setattr(collect.facade, "load_bilin18", lambda **_kwargs: (
        object(), SimpleNamespace(name=checkpoint),
    ))
    monkeypatch.setattr(collect, "load_programs", lambda _device: {})
    monkeypatch.setattr(collect, "c512_tensors", lambda _device: {})
    monkeypatch.setattr(collect, "collect", lambda *_args: (
        features, finite, valid_control_hashes(), collect.expected_calls(),
    ))
    return paths, parent


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


def test_control_seeds_are_role_separated_and_coordinate_bound():
    design = {collect.control_seed("DESIGN", pi, bi) for pi in range(3) for bi in range(2)}
    heldout = {collect.control_seed("HELDOUT", pi, bi) for pi in range(3) for bi in range(2)}
    assert len(design) == len(heldout) == 6
    assert design.isdisjoint(heldout)
    with pytest.raises(ValueError, match="coordinates"):
        collect.control_seed("DESIGN", 3, 0)


def test_ledger_schema_accepts_exact_replay_and_rejects_nonexact():
    features = torch.ones(3, 2, 3, 32, len(core.FEATURE_NAMES), dtype=torch.float64)
    finite = torch.zeros(3, 2, 32, len(core.FINITE_NAMES), dtype=torch.float64)
    finite[..., 5:] = 1
    control_hashes = valid_control_hashes()
    value = {
        "schema": "mlp2_error_rayleigh_v1_role_ledger", "role": "DESIGN",
        "features": features, "finite": finite,
        "axes": {"programs": list(collect.PROGRAM_NAMES),
                 "backgrounds": list(collect.BACKGROUND_NAMES),
                 "controls": list(core.CONTROL_NAMES),
                 "features": list(core.FEATURE_NAMES),
                 "finite": list(core.FINITE_NAMES), "documents": 32},
        "control_hashes": control_hashes, "calls": collect.expected_calls(),
        "authority_sha256": "a", "checkpoint": {},
    }
    assert collect.validate_ledger(value, "a", "DESIGN") is value
    value["finite"] = finite.clone(); value["finite"][0, 0, 0, 5] = 0
    with pytest.raises(RuntimeError, match="ledger tensors"):
        collect.validate_ledger(value, "a", "DESIGN")
    value["finite"] = finite
    value["control_hashes"]["FULL512|NATIVE"]["errors"]["ACTUAL"] = "g" * 64
    with pytest.raises(RuntimeError, match="control-hash schema"):
        collect.validate_ledger(value, "a", "DESIGN")


def test_source_closure_contains_direct_science_and_tests():
    for path in (collect.PREREG, collect.ADDENDUM, collect.RUNNER, collect.TEST,
                 collect.CORE, collect.CORE_TEST,
                 collect.HERE / "mlp2_error_rayleigh_metrics.py",
                 collect.HERE / "test_mlp2_error_rayleigh_metrics.py",
                 collect.HERE / "mlp2_error_rayleigh_predictor.py",
                 collect.HERE / "test_mlp2_error_rayleigh_predictor.py",
                 collect.HERE / "prepare_mlp2_trajectory_robust_r512_v1_eval_rows.py",
                 collect.HERE / "prepare_mlp0_c512_mlp2_full512_composition_v2_rows.py",
                 collect.HERE / "prepare_mlp0_c512_mlp2_full512_composition_v1_rows.py",
                 collect.HERE / "mlp2_cmr_v1_physical_program.py"):
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


def test_forward_capture_calls_attention_once_and_edits_complete_mlp2_write(monkeypatch):
    class Block:
        def __init__(self, site):
            self.site = site
            self.attn = lambda state, first: (state + site, torch.tensor(float(site)))
            self.mlp = lambda state: state + 100 + site

    def fake_forward(_model, tokens, attention, mlp):
        state = torch.zeros(*tokens.shape, 4, dtype=torch.bfloat16)
        first = torch.tensor(0.0)
        for site in range(18):
            block = Block(site)
            event_a = collect.facade.AttentionEvent(site, block, state, tokens, first)
            write, first = attention(event_a)
            event_m = collect.facade.EarlyMLPEvent(site, block, state, write, tokens, ())
            mlp(event_m)
        return torch.zeros(*tokens.shape, 9)

    monkeypatch.setattr(collect.facade, "forward_with_dispatch", fake_forward)
    monkeypatch.setattr(collect.base, "c512_write", lambda event, tensors: event.state + 7)
    calls = {}
    tokens = torch.zeros(2, 5, dtype=torch.long)
    candidate = torch.full((2, 5, 4), 23, dtype=torch.bfloat16)
    out = collect.forward_capture(
        object(), tokens, "C512", "ACTUAL", {}, candidate=candidate,
        alpha=1.0, calls=calls,
    )
    assert out["attention5"].shape == candidate.shape
    assert out["attention6"].shape == candidate.shape
    assert calls["attention_calls"] == 18
    assert calls["c512_calls"] == 1
    assert calls["native_mlp2_calls"] == 1
    assert calls["injected_calls"] == 1


def test_mocked_success_transaction_publishes_authority_ledger_receipt_in_order(
        tmp_path, monkeypatch):
    paths, _ = mocked_transaction(tmp_path, monkeypatch)
    order = []
    original_json, original_torch = collect.base.atomic_json, collect.base.atomic_torch

    def atomic_json(path, value, *, pre_link_check=None):
        original_json(path, value, pre_link_check=pre_link_check); order.append(path)

    def atomic_torch(path, value, *, pre_link_check=None):
        original_torch(path, value, pre_link_check=pre_link_check); order.append(path)

    monkeypatch.setattr(collect.base, "atomic_json", atomic_json)
    monkeypatch.setattr(collect.base, "atomic_torch", atomic_torch)
    collect.run("DESIGN")
    assert order == [paths["authority"], paths["ledger"], paths["receipt"]]
    assert paths["receipt"].is_file() and not paths["failure"].exists()


def test_mocked_checkpoint_mismatch_is_terminal_before_collection(tmp_path, monkeypatch):
    paths, _ = mocked_transaction(tmp_path, monkeypatch, checkpoint="wrong-checkpoint")
    with pytest.raises(RuntimeError, match="differs from frozen authority"):
        collect.run("DESIGN")
    failure = json.loads(paths["failure"].read_text())
    assert failure["model_or_response_may_have_opened"] is True
    assert not paths["ledger"].exists() and not paths["receipt"].exists()


def test_mocked_protected_drift_after_authority_is_preserved_as_failure(tmp_path, monkeypatch):
    paths, _ = mocked_transaction(tmp_path, monkeypatch)
    calls = {"count": 0}

    def drifting(authority):
        calls["count"] += 1
        if calls["count"] >= 3:
            raise RuntimeError("protected parent drift")
        return authority["parent_snapshot"]

    monkeypatch.setattr(collect, "protected_snapshot", drifting)
    with pytest.raises(RuntimeError, match="protected parent drift"):
        collect.run("DESIGN")
    failure = json.loads(paths["failure"].read_text())
    assert failure["protected_observation"]["status"] == "replay_error"
    assert not paths["receipt"].exists()


def test_mocked_late_rival_terminal_blocks_receipt(tmp_path, monkeypatch):
    paths, _ = mocked_transaction(tmp_path, monkeypatch)
    calls = {"count": 0}
    original = collect.verify_protected

    def rival(expected, authority, claim, role_paths):
        calls["count"] += 1
        original(expected, authority, claim, role_paths)
        if calls["count"] == 3:
            role_paths["failure"].write_text("{}")

    monkeypatch.setattr(collect, "verify_protected", rival)
    with pytest.raises(RuntimeError, match="terminal raced receipt"):
        collect.run("DESIGN")
    assert paths["failure"].is_file() and not paths["receipt"].exists()


def test_mocked_lock_replacement_and_authority_mutation_fail_closed(tmp_path, monkeypatch):
    paths, _ = mocked_transaction(tmp_path, monkeypatch)
    calls = {"count": 0}

    def replaced(_claim, _path):
        calls["count"] += 1
        if calls["count"] >= 3:
            raise RuntimeError("lock replaced")

    monkeypatch.setattr(collect.row_life.base, "require_claim", replaced)
    with pytest.raises(RuntimeError, match="lock replaced"):
        collect.run("DESIGN")
    assert paths["authority"].is_file() and not paths["receipt"].exists()

    # A fresh namespace exercises semantic mutation immediately after authority link.
    other = tmp_path / "other"; other.mkdir()
    paths, _ = mocked_transaction(other, monkeypatch)
    original_json = collect.base.atomic_json

    def mutate_authority(path, value, *, pre_link_check=None):
        original_json(path, value, pre_link_check=pre_link_check)
        if path == paths["authority"]:
            changed = json.loads(path.read_text()); changed["status"] = "mutated"
            path.write_text(json.dumps(changed))

    monkeypatch.setattr(collect.base, "atomic_json", mutate_authority)
    with pytest.raises(RuntimeError, match="authority changed before role access"):
        collect.run("DESIGN")
    assert not paths["receipt"].exists()


def test_committed_scorer_source_map_rejects_empty_and_truncated():
    with pytest.raises(RuntimeError, match="exact canonical set"):
        collect.committed_hash_map("irrelevant", {})
    one = {str(collect.SOURCE_PATHS[0].relative_to(collect.ROOT)): "a" * 64}
    with pytest.raises(RuntimeError, match="exact canonical set"):
        collect.committed_hash_map("irrelevant", one)


def test_mutation_during_program_load_cannot_reach_collect(tmp_path, monkeypatch):
    paths, _ = mocked_transaction(tmp_path, monkeypatch)
    reached = {"collect": False}

    def mutate_during_load(_device):
        changed = json.loads(paths["authority"].read_text())
        changed["status"] = "mutated_during_program_load"
        paths["authority"].write_text(json.dumps(changed))
        return {}

    def forbidden_collect(*_args):
        reached["collect"] = True
        raise AssertionError("collect must remain unopened")

    monkeypatch.setattr(collect, "load_programs", mutate_during_load)
    monkeypatch.setattr(collect, "collect", forbidden_collect)
    with pytest.raises(RuntimeError, match="JSON parent hash changed|final collection boundary"):
        collect.run("DESIGN")
    assert reached["collect"] is False
    assert not paths["ledger"].exists() and not paths["receipt"].exists()


def test_rival_during_final_authority_reload_cannot_reach_collect(tmp_path, monkeypatch):
    paths, _ = mocked_transaction(tmp_path, monkeypatch)
    reached = {"collect": False}; calls = {"authority": 0}
    original_stable = collect.stable_json

    def racing_stable(path, expected=None):
        value = original_stable(path, expected)
        if path == paths["authority"]:
            calls["authority"] += 1
            if calls["authority"] == 3:
                paths["failure"].write_text("{}")
        return value

    def forbidden_collect(*_args):
        reached["collect"] = True
        raise AssertionError("collect must remain unopened")

    monkeypatch.setattr(collect, "stable_json", racing_stable)
    monkeypatch.setattr(collect, "collect", forbidden_collect)
    with pytest.raises(RuntimeError, match="terminal raced final authority reload"):
        collect.run("DESIGN")
    assert reached["collect"] is False
    assert paths["failure"].is_file() and not paths["ledger"].exists()
