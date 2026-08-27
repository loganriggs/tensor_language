import math

import pytest
import torch

import joint_early_mlp_oracle_factorial_development as DEV


def test_realization_component_hashes_match_device_independent_tensor_trees():
    saved = {
        "ship": {"x": torch.arange(4)},
        "corr": {"on": True, "b": torch.ones(2)},
        "attention": {0: {"q": torch.eye(2)}},
        "all_attention": [0, 1],
    }
    expected = DEV.realization_component_hashes(saved)
    assert DEV.verify_realization(saved, expected, expected) == expected
    bad = dict(expected)
    bad["ship"] = "wrong"
    with pytest.raises(RuntimeError, match="does not match"):
        DEV.verify_realization(saved, bad, expected)


def test_registered_decisions_use_heldout_joint_and_conditional_metrics():
    row = {
        "gain_by_arm": {"mlp0": 0.10, "mlp1": 0.15, "mlp2": -0.20},
        "joint_gain": 0.25,
        "mlp2_conditional_marginal_after_mlp0_mlp1": 0.04,
        "interaction_l1_fraction_of_joint_gain": 0.30,
    }
    decisions = DEV.score_decisions({"discovery": row, "heldout": row})
    assert all(decisions.values())
    row["mlp2_conditional_marginal_after_mlp0_mlp1"] = -0.01
    assert DEV.score_decisions({"discovery": row, "heldout": row})[
        "pred_b_mlp2_conditional_marginal_after_mlp0_mlp1_is_positive"
    ] is False


def test_nonfinite_or_missing_saved_components_fail_closed():
    with pytest.raises(RuntimeError, match="missing components"):
        DEV.realization_component_hashes({"ship": {}})
    row = {
        "gain_by_arm": {"mlp0": 0.1, "mlp1": math.nan, "mlp2": 0.0},
        "joint_gain": 0.2,
        "mlp2_conditional_marginal_after_mlp0_mlp1": 0.0,
        "interaction_l1_fraction_of_joint_gain": 0.0,
    }
    with pytest.raises(ValueError, match="must be finite"):
        DEV.score_decisions({"discovery": row, "heldout": row})


def test_joint_namespace_is_distinct_and_protected_snapshot_includes_prior_runs():
    assert DEV.RESULT.name.startswith("joint_early_mlp_oracle_factorial_curated_dev_v2")
    assert DEV.RESULT not in DEV.PROTECTED_EXISTING
    snapshot = DEV.protected_snapshot()
    assert str(DEV.PREREG) in snapshot
    assert str(DEV.SAVED_SHIP) in snapshot
    assert all(str(path) in snapshot for path in DEV.local.CANONICAL_PATHS)


def test_row_split_receipt_is_materialized_without_external_helper():
    splits = DEV.local.allocate_whole_document_splits(
        torch.load(DEV.local.CORPUS, map_location="cpu", weights_only=True)
    )
    receipts = DEV.row_split_receipts(splits)
    assert receipts["discovery"]["shape"] == [192, 257]
    assert receipts["heldout"]["dtype"] == "torch.int64"
    assert len(receipts["ship_fit"]["indices"]) == 480
    assert receipts["covariance"]["tensor_raw_sha256"] == splits["covariance"]["tensor_raw_sha256"]
