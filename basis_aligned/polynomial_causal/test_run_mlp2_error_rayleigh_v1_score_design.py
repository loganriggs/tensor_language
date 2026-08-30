import json

import pytest
import torch

import mlp2_error_rayleigh_collector_core as core
import mlp2_error_rayleigh_predictor as predictor
import run_mlp2_error_rayleigh_v1_score_design as score


def mocked_scorer_transaction(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in (
        "authority", "bundle", "receipt", "failure", "lock",
    )}
    design = {
        "authority": tmp_path / "design_authority.json",
        "ledger": tmp_path / "design_ledger.pt",
        "receipt": tmp_path / "design_receipt.json",
    }
    design_authority = {"parent_snapshot": {"checkpoint": {"name": "checkpoint"}}}
    score.base.atomic_json(design["authority"], design_authority)
    features = torch.ones(3, 2, 3, 32, len(core.FEATURE_NAMES), dtype=torch.float64)
    finite = torch.zeros(3, 2, 32, len(core.FINITE_NAMES), dtype=torch.float64)
    finite[..., 5:] = 1
    ledger = {"features": features, "finite": finite}
    score.base.atomic_torch(design["ledger"], ledger)
    design_authority_sha = score.file_sha256(design["authority"])
    ledger_sha = score.file_sha256(design["ledger"])
    design_receipt = {
        "schema": "mlp2_error_rayleigh_v1_collector_receipt",
        "status": "role_measurements_complete_receipt_last", "role": "DESIGN",
        "authority_sha256": design_authority_sha, "ledger_sha256": ledger_sha,
        "runtime_s": 1.0, "model_responses_opened": True,
        "heldout_predictor_was_frozen": False,
    }
    score.base.atomic_json(design["receipt"], design_receipt)

    monkeypatch.setattr(score, "AUTHORITY", paths["authority"])
    monkeypatch.setattr(score, "BUNDLE", paths["bundle"])
    monkeypatch.setattr(score, "RECEIPT", paths["receipt"])
    monkeypatch.setattr(score, "FAILURE", paths["failure"])
    monkeypatch.setattr(score, "LOCK", paths["lock"])
    monkeypatch.setattr(score, "DESIGN", design)
    monkeypatch.setattr(score, "audited_source_commit", lambda: "audited-commit")
    monkeypatch.setattr(score, "validate_spent_v3_scorer", lambda: score.V3_FAILURE_SHA)
    monkeypatch.setattr(score, "source_hashes", lambda _commit: {})
    monkeypatch.setattr(score, "validate_audit", lambda _sources: (
        {"reviewer": "mock-independent-auditor", "audited_source_commit": "audited-commit"},
        "a" * 64,
    ))
    monkeypatch.setattr(score, "validate_design_authority", lambda value, _sha: value)
    monkeypatch.setattr(score.collector, "protected_snapshot", lambda _authority: {})
    monkeypatch.setattr(score.collector, "validate_ledger", lambda value, *_args: value)
    monkeypatch.setattr(score.row_life.base, "acquire_claim", lambda _path: object())
    monkeypatch.setattr(score.row_life.base, "require_claim", lambda _claim, _path: None)
    monkeypatch.setattr(score.row_life.base, "release_claim", lambda _claim, _path: None)

    width_models = {}
    for family, names in predictor.FAMILIES.items():
        width = len(names)
        width_models[family] = {
            "ridge": {"selected": 0.01,
                      "clustered_lodo_mse": {penalty: 1.0 for penalty in predictor.RIDGE_GRID}},
            "mean": torch.zeros(width, dtype=torch.float64),
            "scale": torch.ones(width, dtype=torch.float64),
            "coefficients": torch.ones(width + 1, dtype=torch.float64),
            "design_prediction": torch.zeros(32, 3, dtype=torch.float64),
        }
    fit = {
        "target": torch.zeros(32, 3, dtype=torch.float64), "models": width_models,
        "null_predictions": {control: {
            family: torch.zeros(32, 3, dtype=torch.float64) for family in predictor.FAMILIES
        } for control in ("DERANGED", "COV_RANDOM")},
    }
    monkeypatch.setattr(score.predictor, "fit_design", lambda *_args: fit)
    return paths, design


def test_serialized_predictor_bundle_has_exact_deployable_schema():
    models = {}
    for family, names in predictor.FAMILIES.items():
        width = len(names)
        models[family] = {
            "ridge": {"selected": 0.01,
                      "clustered_lodo_mse": {penalty: 1.0 for penalty in predictor.RIDGE_GRID}},
            "mean": torch.zeros(width, dtype=torch.float64),
            "scale": torch.ones(width, dtype=torch.float64),
            "coefficients": torch.ones(width + 1, dtype=torch.float64),
            "design_prediction": torch.zeros(32, 3, dtype=torch.float64),
        }
    fit = {
        "target": torch.zeros(32, 3, dtype=torch.float64), "models": models,
        "null_predictions": {control: {
            family: torch.zeros(32, 3, dtype=torch.float64) for family in predictor.FAMILIES
        } for control in ("DERANGED", "COV_RANDOM")},
    }
    value = score.serialize_fit(fit)
    assert score.validate_bundle(value) is value
    assert value["program_identity_feature"] is False
    assert value["directional_amplitude_reduction"] == "arithmetic_mean_h16_h8"


def test_source_closure_contains_scorer_predictor_and_collector_contracts():
    for path in (score.RUNNER, score.TEST, score.HERE / "mlp2_error_rayleigh_predictor.py",
                 score.HERE / "test_mlp2_error_rayleigh_predictor.py",
                 score.collector.RUNNER, score.collector.TEST, score.collector.ADDENDUM):
        assert score.SOURCE_PATHS.count(path) == 1
    assert score.SOURCE_PATHS.count(score.RECOVERY_AMENDMENT) == 1
    assert set(score.collector.SOURCE_PATHS).issubset(score.SOURCE_PATHS)


def test_v4_namespace_is_fresh_and_v3_terminal_is_bound():
    assert "v4_design_predictor" in score.AUTHORITY.name
    assert "v4_design_predictor" in score.BUNDLE.name
    assert "v4_design_predictor" in score.RECEIPT.name
    assert score.validate_spent_v3_scorer() == score.V3_FAILURE_SHA
    assert all(not path.exists() for path in score.V3_ABSENT_PATHS)


def test_receipt_shape_matches_heldout_unlock_exactly():
    required = {
        "schema", "status", "design_ledger_sha256", "design_receipt_sha256",
        "predictor_authority_sha256", "scorer_audit_sha256",
        "predictor_bundle_sha256", "heldout_unlocked",
    }
    assert required == {
        "schema", "status", "design_ledger_sha256", "design_receipt_sha256",
        "predictor_authority_sha256", "scorer_audit_sha256",
        "predictor_bundle_sha256", "heldout_unlocked",
    }
    assert "v4_design_predictor" in score.RECEIPT.name


def test_success_publishes_authority_before_any_design_tensor_access(tmp_path, monkeypatch):
    paths, design = mocked_scorer_transaction(tmp_path, monkeypatch)
    accesses = []
    original_stable_torch = score.base.stable_torch

    def guarded_stable_torch(path, expected=None):
        if path == design["ledger"]:
            assert paths["authority"].is_file(), "DESIGN tensor opened before scorer authority"
            accesses.append("design-ledger")
        return original_stable_torch(path, expected)

    monkeypatch.setattr(score.base, "stable_torch", guarded_stable_torch)
    score.run()
    assert accesses
    assert paths["authority"].is_file()
    assert paths["bundle"].is_file()
    assert paths["receipt"].is_file()
    assert not paths["failure"].exists()


def test_preauthority_metadata_failure_never_opens_design_tensor(tmp_path, monkeypatch):
    paths, design = mocked_scorer_transaction(tmp_path, monkeypatch)
    opened = {"value": False}

    def forbidden_stable_torch(path, expected=None):
        if path == design["ledger"]:
            opened["value"] = True
            raise AssertionError("tensor access forbidden")
        return original(path, expected)

    original = score.base.stable_torch
    monkeypatch.setattr(score, "stable_file_sha", lambda *_args, **_kwargs: (
        (_ for _ in ()).throw(RuntimeError("metadata drift"))
    ))
    monkeypatch.setattr(score.base, "stable_torch", forbidden_stable_torch)
    with pytest.raises(RuntimeError, match="metadata drift"):
        score.run()
    assert opened["value"] is False
    assert not paths["authority"].exists()
    failure = json.loads(paths["failure"].read_text())
    assert failure["design_ledger_may_have_opened"] is False


def test_same_source_map_different_audit_commit_swap_never_opens_design_tensor(
        tmp_path, monkeypatch):
    paths, design = mocked_scorer_transaction(tmp_path, monkeypatch)
    original_torch = score.base.stable_torch
    design_loads = {"count": 0}

    def swapped_audit(_sources):
        return {
            "reviewer": "mock-independent-auditor",
            "audited_source_commit": "different-final-audit-commit",
        }, "a" * 64

    def record_torch(path, expected=None):
        if path == design["ledger"]:
            design_loads["count"] += 1
        return original_torch(path, expected)

    monkeypatch.setattr(score, "validate_audit", swapped_audit)
    monkeypatch.setattr(score.base, "stable_torch", record_torch)
    with pytest.raises(RuntimeError, match="audit commit swapped after selection"):
        score.run()
    assert design_loads["count"] == 0
    assert not paths["authority"].exists()
    assert paths["failure"].is_file()
    assert json.loads(paths["failure"].read_text())["design_ledger_may_have_opened"] is False


def test_protected_drift_after_authority_is_publishable(tmp_path, monkeypatch):
    paths, _ = mocked_scorer_transaction(tmp_path, monkeypatch)

    def drifting(authority):
        raise RuntimeError("post-authority protected drift")

    monkeypatch.setattr(score, "protected_snapshot", drifting)
    with pytest.raises(RuntimeError, match="post-authority protected drift"):
        score.run()
    failure = json.loads(paths["failure"].read_text())
    assert paths["authority"].is_file() and not paths["receipt"].exists()
    assert failure["design_ledger_may_have_opened"] is True
    assert failure["protected_observation"]["status"] == "replay_error"


def test_authority_mutation_before_ledger_access_fails_without_tensor_open(tmp_path, monkeypatch):
    paths, design = mocked_scorer_transaction(tmp_path, monkeypatch)
    original_json = score.base.atomic_json
    original_torch = score.base.stable_torch
    opened = {"value": False}

    def mutate_authority(path, value, *, pre_link_check=None):
        original_json(path, value, pre_link_check=pre_link_check)
        if path == paths["authority"]:
            changed = json.loads(path.read_text()); changed["status"] = "mutated"
            path.write_text(json.dumps(changed))

    def record_torch(path, expected=None):
        if path == design["ledger"]:
            opened["value"] = True
        return original_torch(path, expected)

    monkeypatch.setattr(score.base, "atomic_json", mutate_authority)
    monkeypatch.setattr(score.base, "stable_torch", record_torch)
    with pytest.raises(RuntimeError, match="authority changed before ledger access|JSON parent hash changed"):
        score.run()
    assert opened["value"] is False
    assert paths["failure"].is_file() and not paths["receipt"].exists()


def test_authority_drift_at_final_preopen_boundary_never_opens_design_tensor(
        tmp_path, monkeypatch):
    paths, design = mocked_scorer_transaction(tmp_path, monkeypatch)
    original_json = score.base.stable_json
    original_torch = score.base.stable_torch
    authority_reads = {"count": 0}
    design_loads = {"count": 0}

    def drift_after_first_postlink_replay(path, expected=None):
        value = original_json(path, expected)
        if path == paths["authority"]:
            authority_reads["count"] += 1
            if authority_reads["count"] == 1:
                changed = json.loads(path.read_text())
                changed["status"] = "raced-after-first-replay"
                path.write_text(json.dumps(changed))
        return value

    def record_torch(path, expected=None):
        if path == design["ledger"]:
            design_loads["count"] += 1
        return original_torch(path, expected)

    monkeypatch.setattr(score.base, "stable_json", drift_after_first_postlink_replay)
    monkeypatch.setattr(score.base, "stable_torch", record_torch)
    with pytest.raises(RuntimeError, match="pre-open boundary|JSON parent hash changed"):
        score.run()
    assert authority_reads["count"] >= 2
    assert design_loads["count"] == 0
    assert not paths["receipt"].exists()


def test_rival_terminal_at_final_preopen_boundary_never_opens_design_tensor(
        tmp_path, monkeypatch):
    paths, design = mocked_scorer_transaction(tmp_path, monkeypatch)
    original_json = score.base.stable_json
    original_torch = score.base.stable_torch
    authority_reads = {"count": 0}
    design_loads = {"count": 0}

    def insert_rival_after_first_postlink_replay(path, expected=None):
        value = original_json(path, expected)
        if path == paths["authority"]:
            authority_reads["count"] += 1
            if authority_reads["count"] == 1:
                paths["failure"].write_text("{}")
        return value

    def record_torch(path, expected=None):
        if path == design["ledger"]:
            design_loads["count"] += 1
        return original_torch(path, expected)

    monkeypatch.setattr(score.base, "stable_json", insert_rival_after_first_postlink_replay)
    monkeypatch.setattr(score.base, "stable_torch", record_torch)
    with pytest.raises(RuntimeError, match="terminal appeared at pre-open boundary"):
        score.run()
    assert authority_reads["count"] >= 2
    assert design_loads["count"] == 0
    assert not paths["receipt"].exists()


def test_late_rival_terminal_blocks_receipt(tmp_path, monkeypatch):
    paths, _ = mocked_scorer_transaction(tmp_path, monkeypatch)
    original_torch = score.base.atomic_torch

    def rival_after_bundle(path, value, *, pre_link_check=None):
        original_torch(path, value, pre_link_check=pre_link_check)
        if path == paths["bundle"]:
            paths["failure"].write_text("{}")

    monkeypatch.setattr(score.base, "atomic_torch", rival_after_bundle)
    with pytest.raises(RuntimeError, match="terminal raced receipt"):
        score.run()
    assert paths["failure"].is_file() and not paths["receipt"].exists()


def test_lock_replacement_fails_closed(tmp_path, monkeypatch):
    paths, design = mocked_scorer_transaction(tmp_path, monkeypatch)
    calls = {"count": 0}
    design_loads = {"count": 0}
    original_torch = score.base.stable_torch

    def replaced(_claim, _path):
        calls["count"] += 1
        if calls["count"] >= 3:
            raise RuntimeError("lock replaced")

    def record_torch(path, expected=None):
        if path == design["ledger"]:
            design_loads["count"] += 1
        return original_torch(path, expected)

    monkeypatch.setattr(score.row_life.base, "require_claim", replaced)
    monkeypatch.setattr(score.base, "stable_torch", record_torch)
    with pytest.raises(RuntimeError, match="lock replaced"):
        score.run()
    assert design_loads["count"] == 0
    assert not paths["receipt"].exists()
