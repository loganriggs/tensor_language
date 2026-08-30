import json

import pytest

import causal_response_factorization_v1_validation_analysis as analysis


OWNERS = ["x", "y"]


def _pairs(values):
    """Owner-pair reports for a 2-owner table; None marks an unscored (empty) pair."""

    out = {}
    for i in range(2):
        for j in range(2):
            value = values[i][j]
            out[f"{i}->{j}"] = (
                {"cells": 0, "mse": None, "nrmse_by_training_rms": None, "signed_correlation": None}
                if value is None else
                {"cells": 10, "mse": value ** 2, "nrmse_by_training_rms": value, "signed_correlation": 0.1}
            )
    return out


def _report(pooled, pairs, worst):
    return {
        "pooled": {"cells": 40, "mse": pooled ** 2, "nrmse_by_training_rms": pooled, "signed_correlation": 0.2},
        "owner_pairs": _pairs(pairs),
        "worst_owner_pair_nrmse": worst,
    }


def _row(g, p, seed, P, C, uncond, cal_full, cal_partial):
    return {
        "global_rank": g, "private_rank_each_owner": p, "seed": seed,
        "persistent_values": P, "per_document_values": C,
        "unconditional": _report(uncond, [[uncond, uncond], [uncond, uncond]], uncond),
        "calibrated": {
            "sha256_outcome_blind_blocks": {
                "2": {
                    "status": "scored", "supported_document_fraction": 1.0, "support_gate_passes": True,
                    "calibrated": _report(cal_full, [[cal_full, cal_full], [cal_full, 3.0]], 3.0),
                },
            },
            "training_only_block_d_optimal": {
                "2": {
                    "status": "scored", "supported_document_fraction": 1.0, "support_gate_passes": True,
                    # second owner block fully anchored: pairs 1->* unscored
                    "calibrated": _report(cal_partial, [[cal_partial, cal_partial], [None, None]], -1.0),
                },
            },
        },
    }


def _table():
    return {
        "owner_components": OWNERS, "designs": [
            "sha256_outcome_blind_blocks", "training_only_block_d_optimal",
        ],
        "calibration_arm_budgets": [2], "training_response_rms": 0.5,
        "candidates": [
            _row(1, 0, 1, 100, 1, 0.99, 0.9, 0.3),
            _row(1, 0, 2, 100, 1, 1.01, 0.8, 0.2),
            _row(2, 0, 1, 200, 2, 0.98, 0.7, 0.25),
            _row(2, 0, 2, 200, 2, 0.97, 0.6, 0.35),
        ],
        "candidate_selected": False,
    }


def test_incomplete_owner_coverage_is_detected_and_excluded_from_frontier():
    result = analysis.analyze(_table())
    g1 = result["rank_pairs"][0]
    d_opt = g1["calibrated"]["training_only_block_d_optimal"]["2"]
    assert d_opt["owner_pairs_scored_per_seed"] == [2, 2]
    assert d_opt["complete_owner_coverage_all_seeds"] is False
    assert d_opt["eligible_for_block_balanced_frontier"] is False
    assert d_opt["worst_owner_pair_nrmse_median"] == 0.25
    assert d_opt["per_seed"][0]["scorer_worst_owner_pair_nrmse"] == -1.0
    blind = g1["calibrated"]["sha256_outcome_blind_blocks"]["2"]
    assert blind["complete_owner_coverage_all_seeds"] is True
    assert blind["eligible_for_block_balanced_frontier"] is True
    assert blind["worst_owner_pairs"] == ["1->1", "1->1"]
    frontiers = result["block_balanced_frontiers"]
    assert frontiers["training_only_block_d_optimal"]["2"]["eligible_rank_pairs"] == []
    assert frontiers["sha256_outcome_blind_blocks"]["2"]["eligible_rank_pairs"] == ["g1_p0", "g2_p0"]


def test_frontier_is_nondominated_and_keeps_both_price_coordinates():
    result = analysis.analyze(_table())
    blind = result["block_balanced_frontiers"]["sha256_outcome_blind_blocks"]["2"]
    # g1_p0: cheaper (100,1) but worse mse (median 0.85^2); g2_p0: pricier, better mse; same worst.
    assert blind["nondominated_rank_pairs"] == ["g1_p0", "g2_p0"]
    points = {
        "a": {"persistent_values": 1, "per_document_values": 1,
              "calibrated_pooled_mse_median": 1.0, "worst_owner_pair_nrmse_median": 1.0},
        "b": {"persistent_values": 1, "per_document_values": 1,
              "calibrated_pooled_mse_median": 2.0, "worst_owner_pair_nrmse_median": 1.0},
    }
    assert analysis._frontier(points) == ["a"]


def test_prospective_failure_pattern_and_hierarchy_note():
    result = analysis.analyze(_table())
    pattern = result["prospective_failure_pattern"]
    assert pattern["unconditional_fails_broadly"] is True
    assert pattern["unconditional_pooled_nrmse_median_by_pair"] == {"g1_p0": 1.0, "g2_p0": 0.975}
    assert pattern["hierarchy_support"].startswith("untestable")
    assert result["candidate_selected"] is False
    assert result["unconditional_frontier"]["eligible_rank_pairs"] == ["g1_p0", "g2_p0"]
    json.dumps(result, allow_nan=False)


def test_publish_is_create_only(tmp_path):
    output = tmp_path / "analysis.json"
    analysis.publish_create_only({"a": 1}, output)
    assert json.loads(output.read_text()) == {"a": 1}
    with pytest.raises(RuntimeError, match="already spent"):
        analysis.publish_create_only({"a": 2}, output)
    assert json.loads(output.read_text()) == {"a": 1}
