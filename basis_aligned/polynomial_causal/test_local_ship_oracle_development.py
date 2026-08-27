import json
import hashlib

import pytest
import torch

import local_ship_oracle_development as DEV


def current_payload():
    return torch.load(DEV.CORPUS, map_location="cpu", weights_only=True)


def test_whole_document_split_sizes_and_no_leakage():
    splits = DEV.allocate_whole_document_splits(current_payload())
    assert {role: len(row["rows"]) for role, row in splits.items()} == {
        "ship_fit": 480, "basis": 96, "discovery": 192,
        "heldout": 192, "spare": 40, "covariance": 96,
    }
    role_documents = {
        role: set(row["document_ids"])
        for role, row in splits.items() if role != "covariance"
    }
    roles = list(role_documents)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            assert role_documents[left].isdisjoint(role_documents[right])
    assert torch.equal(splits["covariance"]["rows"], splits["ship_fit"]["rows"][:96])
    causal = ("ship_fit", "basis", "discovery", "heldout")
    for index, left in enumerate(causal):
        left_rows = splits[left]["rows"]
        left_full = {DEV.tensor_sha256(row) for row in left_rows}
        left_prefix = {tuple(row[:32].tolist()) for row in left_rows}
        for right in causal[index + 1:]:
            assert left_full.isdisjoint(DEV.tensor_sha256(row) for row in splits[right]["rows"])
            assert left_prefix.isdisjoint(tuple(row[:32].tolist()) for row in splits[right]["rows"])


def test_development_decisions_can_never_create_training_licenses():
    result = {"site_decisions": {}, "paired_gains": {}}
    for site in range(3):
        key = str(site)
        result["site_decisions"][key] = {
            "full_oracle_ci95_lower_gt_zero": True,
            "content_positive_both_splits": True,
        }
        heldout = {"content": {"global": {"mean": 1.0}}}
        heldout.update({f"null_{index:02d}": {"global": {"mean": 0.0}}
                        for index in range(20)})
        result["paired_gains"][key] = {"heldout": heldout}

    candidates = DEV.exact_development_decisions(
        result, lambda content, nulls: {"passes_5pct": content > max(nulls)}
    )
    assert candidates == [0, 1, 2]
    assert result["development_candidate_sites"] == [0, 1, 2]
    assert result["training_license_sites"] == []


def test_manifest_is_explicitly_nonauthoritative():
    splits = DEV.allocate_whole_document_splits(current_payload())
    manifest = DEV.manifest_for_splits(splits)
    assert manifest["authority"] == "none"
    assert manifest["authorized_for_scored_experiments"] is False
    assert manifest["training_license_sites"] == []
    assert "no FineWeb" in manifest["scope_guardrail"]
    assert json.dumps(manifest)


def test_preregistration_has_a_distinct_immutable_path():
    assert DEV.DEV_PREREG != DEV.DEV_MANIFEST
    assert DEV.DEV_PREREG != DEV.DEV_RESULT
    assert "preregistration" in DEV.DEV_PREREG.name


def test_immutable_preregistration_bytes_survive_manifest_updates(tmp_path):
    prereg = tmp_path / "prereg.json"
    manifest = tmp_path / "manifest.json"
    DEV.write_json_atomic({"status": "preregistered"}, prereg)
    before = prereg.read_bytes()
    before_hash = hashlib.sha256(before).hexdigest()
    DEV.write_json_atomic({"status": "running", "preregistration_sha256": before_hash}, manifest)
    DEV.write_json_atomic({"status": "completed", "preregistration_sha256": before_hash}, manifest)
    assert prereg.read_bytes() == before
    assert hashlib.sha256(prereg.read_bytes()).hexdigest() == before_hash


def test_failure_marker_invalidates_result_and_license(monkeypatch, tmp_path):
    result = tmp_path / "result.json"
    manifest = tmp_path / "manifest.json"
    monkeypatch.setattr(DEV, "DEV_RESULT", result)
    monkeypatch.setattr(DEV, "DEV_MANIFEST", manifest)
    DEV.write_json_atomic({"status": "completed_exploratory_only"}, manifest)
    DEV.write_json_atomic({
        "config": {"status": "completed", "authorized_for_scored_experiments": True},
        "training_license_sites": [0],
    }, result)
    DEV.mark_failed("invalid_canonical_contamination", RuntimeError("changed"), {"x": "y"})
    marked_manifest = json.loads(manifest.read_text())
    marked_result = json.loads(result.read_text())
    assert marked_manifest["status"] == "invalid_canonical_contamination"
    assert marked_manifest["training_license_sites"] == []
    assert marked_result["config"]["authorized_for_scored_experiments"] is False
    assert marked_result["training_license_sites"] == []
    assert marked_result["invalidated_by_guard"] is True


def test_atomic_claim_refuses_concurrent_launch(monkeypatch, tmp_path):
    lock = tmp_path / "claim"
    lock.mkdir()
    monkeypatch.setattr(DEV, "DEV_LOCK", lock)
    monkeypatch.setattr(DEV, "DEV_RESULT", tmp_path / "result")
    monkeypatch.setattr(DEV, "DEV_PREREG", tmp_path / "prereg")
    monkeypatch.setattr(DEV, "DEV_MANIFEST", tmp_path / "manifest")
    monkeypatch.setattr(DEV, "DEV_STATE", tmp_path / "state")
    monkeypatch.setattr(DEV, "DEV_ORACLE_STATE", tmp_path / "oracle")
    with pytest.raises(RuntimeError, match="already claimed"):
        DEV.main()


def test_nonfinite_exact_null_inputs_fail_closed():
    result = {"site_decisions": {}, "paired_gains": {}}
    for site in range(3):
        key = str(site)
        result["site_decisions"][key] = {
            "full_oracle_ci95_lower_gt_zero": True,
            "content_positive_both_splits": True,
        }
        heldout = {"content": {"global": {"mean": float("nan") if site == 0 else 1.0}}}
        heldout.update({f"null_{index:02d}": {"global": {"mean": 0.0}}
                        for index in range(20)})
        result["paired_gains"][key] = {"heldout": heldout}
    with pytest.raises(RuntimeError, match="nonfinite"):
        DEV.exact_development_decisions(result, lambda *_: {"passes_5pct": True})
