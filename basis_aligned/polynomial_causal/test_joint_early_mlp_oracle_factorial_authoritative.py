import json
import math

import pytest
import torch

from factorial_causal_attribution import powerset
import joint_early_mlp_oracle_factorial as joint
import joint_early_mlp_oracle_factorial_authoritative as AUTH


def _row_cube(terms, rows=12):
    gains = {
        arm: sum(value for term, value in terms.items() if set(term).issubset(arm))
        for arm in powerset(joint.EARLY_MLP_GROUPS)
    }
    baseline = [3.0 + 0.01 * index for index in range(rows)]
    return {
        arm: [value - gains[arm] for value in baseline]
        for arm in powerset(joint.EARLY_MLP_GROUPS)
    }


def _bootstrap_row(lower=0.01, upper=0.2):
    return {"point_estimate": 0.1, "bootstrap_mean": 0.1, "ci95": [lower, upper]}


def test_document_cluster_bootstrap_is_deterministic_and_matches_point_contrasts():
    terms = {
        (): 0.0,
        ("mlp0",): 0.1,
        ("mlp1",): 0.2,
        ("mlp2",): -0.1,
        ("mlp0", "mlp1"): 0.25,
        ("mlp0", "mlp2"): 0.02,
        ("mlp1", "mlp2"): 0.03,
        ("mlp0", "mlp1", "mlp2"): 0.10,
    }
    rows = _row_cube(terms)
    documents = ["a"] * 4 + ["b"] * 3 + ["c"] * 2 + ["d"] * 3
    first = AUTH.paired_document_cluster_bootstrap(
        rows, documents, draws=100, seed=17
    )
    second = AUTH.paired_document_cluster_bootstrap(
        rows, documents, draws=100, seed=17
    )
    analysis = joint.analyze_full_live_subset_rows(rows)
    assert first == second
    assert first["unique_document_count"] == 4
    assert first["cluster_size_range"] == [2, 4]
    assert first["arm_gain"]["mlp0+mlp1+mlp2"]["point_estimate"] == pytest.approx(
        analysis["joint_gain"]
    )
    assert first["mlp2_conditional_marginal_after_mlp0_mlp1"][
        "point_estimate"
    ] == pytest.approx(analysis["mlp2_conditional_marginal_after_mlp0_mlp1"])
    best = max(analysis["gain_by_arm"][group] for group in joint.EARLY_MLP_GROUPS)
    assert first["joint_minus_best_singleton"]["point_estimate"] == pytest.approx(
        analysis["joint_gain"] - best
    )


def test_cluster_bootstrap_retains_row_weighted_not_equal_document_estimand():
    arms = powerset(joint.EARLY_MLP_GROUPS)
    baseline = [3.0, 3.0, 3.0]
    rows = {arm: list(baseline) for arm in arms}
    rows[("mlp0",)] = [2.0, 2.0, 3.0]
    result = AUTH.paired_document_cluster_bootstrap(
        rows, ["long", "long", "short"], draws=200, seed=5
    )
    assert result["arm_gain"]["mlp0"]["point_estimate"] == pytest.approx(2 / 3)
    assert result["arm_gain"]["mlp0"]["bootstrap_mean"] != pytest.approx(0.5)


def test_registered_decisions_capture_robust_joint_sign_flip_and_nonadditivity():
    row = {
        "gain_by_arm": {"mlp0": 0.1, "mlp1": 0.2, "mlp2": -0.1},
        "joint_gain": 0.5,
        "joint_minus_singleton_sum": 0.3,
        "mlp2_conditional_marginal_after_mlp0_mlp1": 0.12,
        "interaction_l1_fraction_of_joint_gain": 0.4,
    }
    bootstrap = {
        "joint_gain": _bootstrap_row(0.2, 0.7),
        "joint_minus_best_singleton": _bootstrap_row(0.1, 0.4),
        "mlp2_singleton_gain": _bootstrap_row(-0.2, -0.03),
        "mlp2_conditional_marginal_after_mlp0_mlp1": _bootstrap_row(0.03, 0.2),
        "mlp2_sign_flip_contrast": _bootstrap_row(0.08, 0.3),
        "joint_minus_singleton_sum": _bootstrap_row(0.1, 0.5),
    }
    decisions = AUTH.score_decisions(
        {"discovery": row, "heldout": row},
        {"discovery": bootstrap, "heldout": bootstrap},
    )
    assert all(decisions.values())


def test_registered_decisions_do_not_hide_failed_uncertainty_forms():
    row = {
        "gain_by_arm": {"mlp0": 0.1, "mlp1": 0.2, "mlp2": -0.1},
        "joint_gain": 0.5,
        "joint_minus_singleton_sum": 0.3,
        "mlp2_conditional_marginal_after_mlp0_mlp1": 0.12,
        "interaction_l1_fraction_of_joint_gain": 0.4,
    }
    bootstrap = {
        "joint_gain": _bootstrap_row(-0.01, 0.7),
        "joint_minus_best_singleton": _bootstrap_row(-0.01, 0.4),
        "mlp2_singleton_gain": _bootstrap_row(-0.2, 0.01),
        "mlp2_conditional_marginal_after_mlp0_mlp1": _bootstrap_row(-0.02, 0.2),
        "mlp2_sign_flip_contrast": _bootstrap_row(-0.03, 0.3),
        "joint_minus_singleton_sum": _bootstrap_row(-0.04, 0.5),
    }
    decisions = AUTH.score_decisions(
        {"discovery": row, "heldout": row},
        {"discovery": bootstrap, "heldout": bootstrap},
    )
    assert decisions["pred_a_joint_exceeds_best_singleton_heldout"] is True
    assert decisions["pred_a_joint_heldout_ci95_lower_gt_zero"] is False
    assert decisions["pred_b_mlp2_singleton_heldout_ci95_upper_lt_zero"] is False
    assert decisions["pred_b_sign_flip_contrast_heldout_ci95_lower_gt_zero"] is False
    assert decisions[
        "pred_c_joint_minus_singleton_sum_heldout_ci95_lower_gt_zero"
    ] is False


def test_bootstrap_and_decision_inputs_fail_closed():
    rows = _row_cube({(): 0.0})
    rows.pop(("mlp2",))
    with pytest.raises(ValueError, match="complete registered cube"):
        AUTH.paired_document_cluster_bootstrap(rows, ["a"] * 12, draws=10)
    rows = _row_cube({(): 0.0})
    with pytest.raises(ValueError, match="align one-to-one"):
        AUTH.paired_document_cluster_bootstrap(rows, ["a"], draws=10)

    row = {
        "gain_by_arm": {"mlp0": 0.1, "mlp1": math.nan, "mlp2": -0.1},
        "joint_gain": 0.5,
        "joint_minus_singleton_sum": 0.3,
        "mlp2_conditional_marginal_after_mlp0_mlp1": 0.12,
        "interaction_l1_fraction_of_joint_gain": 0.4,
    }
    bootstrap = {
        name: _bootstrap_row() for name in (
            "joint_gain", "joint_minus_best_singleton", "mlp2_singleton_gain",
            "mlp2_conditional_marginal_after_mlp0_mlp1", "mlp2_sign_flip_contrast",
            "joint_minus_singleton_sum",
        )
    }
    with pytest.raises(ValueError, match="must be finite"):
        AUTH.score_decisions(
            {"discovery": row, "heldout": row},
            {"discovery": bootstrap, "heldout": bootstrap},
        )


def test_real_receipt_provenance_prefixes_counts_and_disjointness_validate():
    receipt, rows = AUTH.row_prep.validate_receipt()
    document_ids, receipts = AUTH.validate_document_provenance(receipt, rows)
    assert {role: len(set(ids)) for role, ids in document_ids.items()} == (
        AUTH.UNIQUE_DOCUMENT_COUNTS
    )
    assert receipts["discovery"]["tensor_prefix257_raw_sha256"] == (
        AUTH.PREFIX257_SHA256["discovery"]
    )
    roles = tuple(document_ids)
    for index, left in enumerate(roles):
        for right in roles[index + 1:]:
            assert set(document_ids[left]).isdisjoint(document_ids[right])


def test_prefix_or_provenance_mutation_fails_closed():
    receipt, rows = AUTH.row_prep.validate_receipt()
    changed = dict(rows)
    changed[(192, 7000)] = rows[(192, 7000)].clone()
    changed[(192, 7000)][0, 0] += 1
    with pytest.raises(RuntimeError, match="prefix hash changed"):
        AUTH.validate_document_provenance(receipt, changed)
    bad_receipt = json.loads(json.dumps(receipt))
    bad_receipt["document_provenance"]["sets"]["n192_skip7000"] = []
    with pytest.raises(RuntimeError, match="row-aligned provenance"):
        AUTH.validate_document_provenance(bad_receipt, rows)


def test_finalization_is_the_only_step_that_authorizes_result(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    authority_path = tmp_path / "authority.json"
    result_path.write_text(json.dumps({
        "status": "scored_pending_integrity",
        "authorized_for_scored_experiments": False,
        "ship_realization_sha256": "b" * 64,
    }))
    manifest_path.write_text(json.dumps({
        "status": "scored_pending_integrity",
        "authorized_for_scored_experiments": False,
        "result_sha256": AUTH.file_sha256(result_path),
        "source_commit": "c" * 40,
    }))
    monkeypatch.setattr(AUTH, "RESULT", result_path)
    monkeypatch.setattr(AUTH, "MANIFEST", manifest_path)
    monkeypatch.setattr(AUTH, "AUTHORITY_RECEIPT", authority_path)
    monkeypatch.setattr(
        AUTH, "frozen_lifecycle_receipt",
        lambda _receipt=None: {"validated": True, "artifact_sha256": "a" * 64},
    )
    monkeypatch.setattr(AUTH.row_prep, "RECEIPT", tmp_path / "receipt.json")
    AUTH.row_prep.RECEIPT.write_text("{}")
    AUTH.finalize_success({"protected": "same"})
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    authority = json.loads(authority_path.read_text())
    assert result["authorized_for_scored_experiments"] is False
    assert manifest["authorized_for_scored_experiments"] is False
    assert authority["authorized_for_scored_experiments"] is True
    assert authority["result_sha256"] == AUTH.file_sha256(result_path)
    assert authority["manifest_sha256"] == AUTH.file_sha256(manifest_path)
    assert manifest["result_sha256"] == AUTH.file_sha256(result_path)


def test_finalization_write_failure_never_publishes_authority(monkeypatch, tmp_path):
    result_path = tmp_path / "result.json"
    manifest_path = tmp_path / "manifest.json"
    authority_path = tmp_path / "authority.json"
    result_path.write_text(json.dumps({
        "status": "scored_pending_integrity",
        "authorized_for_scored_experiments": False,
        "ship_realization_sha256": "b" * 64,
    }))
    manifest_path.write_text(json.dumps({
        "status": "scored_pending_integrity",
        "authorized_for_scored_experiments": False,
        "result_sha256": AUTH.file_sha256(result_path),
        "source_commit": "c" * 40,
    }))
    monkeypatch.setattr(AUTH, "RESULT", result_path)
    monkeypatch.setattr(AUTH, "MANIFEST", manifest_path)
    monkeypatch.setattr(AUTH, "AUTHORITY_RECEIPT", authority_path)
    monkeypatch.setattr(
        AUTH, "frozen_lifecycle_receipt",
        lambda _receipt=None: {"validated": True, "artifact_sha256": "a" * 64},
    )
    monkeypatch.setattr(AUTH.row_prep, "RECEIPT", tmp_path / "receipt.json")
    AUTH.row_prep.RECEIPT.write_text("{}")
    original_write = AUTH.write_json_atomic

    def fail_on_manifest(value, path):
        if path == manifest_path:
            raise OSError("injected manifest write failure")
        original_write(value, path)

    monkeypatch.setattr(AUTH, "write_json_atomic", fail_on_manifest)
    with pytest.raises(OSError, match="injected"):
        AUTH.finalize_success({"protected": "same"})
    assert not authority_path.exists()
    assert json.loads(result_path.read_text())["authorized_for_scored_experiments"] is False


def test_namespace_and_authority_guards_are_distinct():
    assert AUTH.RESULT.name.endswith("authoritative_v3_results.json")
    assert AUTH.MANIFEST.name.endswith("authoritative_v3_manifest.json")
    assert AUTH.AUTHORITY_RECEIPT.name.endswith("authoritative_v3_authority.json")
    assert AUTH.RESULT not in AUTH.PROTECTED_EXISTING
    assert AUTH.FROZEN_STATE == AUTH.frozen.FROZEN_STATE
    assert AUTH.FROZEN_MANIFEST == AUTH.frozen.FROZEN_MANIFEST
    assert AUTH.file_sha256(AUTH.PREREG) == AUTH.PREREG_SHA256
