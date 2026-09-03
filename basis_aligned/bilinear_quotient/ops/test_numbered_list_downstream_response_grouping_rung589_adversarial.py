"""Independent, outcome-aware audit tests for the frozen R589 screen.

These tests do not modify or rerun model science.  They recompute R589's saved
FIT-only statistics from the already-produced R584 sufficient-stat rows.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Iterable

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "ops/numbered_list_downstream_response_grouping_rung589.py"
SOURCE = ROOT / "numbered_list_cached_value_downstream_use_rung584_results.json"
RESULT = ROOT / "numbered_list_downstream_response_grouping_rung589_results.json"
EXPECTED_SOURCE_SHA256 = "7980753636fab422ed6c609a1afd054f99ed7f903e2bb3e61eddf0617316fdf6"


def load_r589():
    spec = importlib.util.spec_from_file_location("r589_adversarial_target", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def strict_json(path: Path):
    def reject_constant(value: str):
        raise ValueError(f"non-standard JSON constant: {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


def independent_response(row: dict) -> float:
    native = row["native"]
    intervened = row["intervened"]
    if row["condition"] == "step_two":
        native_margin = float(native["arithmetic_logit"]) - float(native["structural_logit"])
        intervened_margin = float(intervened["arithmetic_logit"]) - float(intervened["structural_logit"])
        assert float(native["arithmetic_minus_structural"]) == pytest.approx(native_margin, abs=1e-7)
        assert float(intervened["arithmetic_minus_structural"]) == pytest.approx(intervened_margin, abs=1e-7)
    else:
        native_margin = float(native["answer_logit"]) - float(native["max_other_candidate_logit"])
        intervened_margin = float(intervened["answer_logit"]) - float(intervened["max_other_candidate_logit"])
        assert float(native["margin"]) == pytest.approx(native_margin, abs=1e-7)
        assert float(intervened["margin"]) == pytest.approx(intervened_margin, abs=1e-7)
    response = native_margin - intervened_margin
    assert math.isfinite(response)
    return response


def independent_correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = tuple(float(value) for value in xs)
    y = tuple(float(value) for value in ys)
    assert len(x) == len(y) and len(x) >= 2
    mx = math.fsum(x) / len(x)
    my = math.fsum(y) / len(y)
    xc = tuple(value - mx for value in x)
    yc = tuple(value - my for value in y)
    denominator = math.sqrt(math.fsum(value * value for value in xc)) * math.sqrt(
        math.fsum(value * value for value in yc)
    )
    assert denominator > 0.0
    return math.fsum(a * b for a, b in zip(xc, yc)) / denominator


@pytest.fixture(scope="module")
def documents():
    return strict_json(SOURCE), strict_json(RESULT)


def test_source_hash_and_strict_json_envelope(documents):
    source, result = documents
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert digest == EXPECTED_SOURCE_SHA256
    assert result["source_result_sha256"] == digest
    assert source["rung"] == result["source_rung"] == 584
    assert source["evaluated_splits"] == result["opened_splits"] == ["FIT"]
    assert source["forbidden_splits_opened"] == []
    assert result["model_forwards"] == result["model_backwards"] == 0
    assert result["model_weights_updated"] is False


def test_exact_twelve_arm_and_576_row_membership(documents):
    source, _ = documents
    raw = source["fit_raw"]
    expected_arms = {
        f"mlp{site}_{component}"
        for site in (8, 10, 12, 14)
        for component in ("background_cross", "contrast_self", "joint_response")
    }
    assert set(raw) == expected_arms
    authority_ids = None
    for arm, rows in raw.items():
        assert len(rows) == 576
        row_ids = {row["row_id"] for row in rows}
        assert len(row_ids) == 576
        authority_ids = row_ids if authority_ids is None else authority_ids
        assert row_ids == authority_ids
        site_token, component = arm.split("_", 1)
        assert all(row["arm"] == arm for row in rows)
        assert all(row["site"] == int(site_token[3:]) for row in rows)
        assert all(row["component"] == component for row in rows)
        assert all(row["split"] == "FIT" for row in rows)
        assert {
            (row["representation"], int(row["source_level"])) for row in rows
        } == {(representation, level) for representation in ("list", "digit", "word") for level in (0, 1)}
        assert {
            condition: sum(row["condition"] == condition for row in rows)
            for condition in {
                "factorial_successor", "surface_successor", "factorial_copy",
                "surface_copy", "relation_break", "step_two",
            }
        } == {
            "factorial_successor": 96,
            "surface_successor": 96,
            "factorial_copy": 96,
            "surface_copy": 96,
            "relation_break": 96,
            "step_two": 96,
        }


def test_actual_semantic_metadata_is_aligned_across_arms(documents):
    source, _ = documents
    indexed = {
        arm: {row["row_id"]: row for row in rows}
        for arm, rows in source["fit_raw"].items()
    }
    arms = sorted(indexed)
    fields = (
        "group_id", "split", "representation", "source_level", "source_value",
        "condition", "action", "token_ids", "query_position", "source_position",
        "source_id", "answer_id", "structural_answer_id", "arithmetic_answer_id",
    )
    for row_id, reference in indexed[arms[0]].items():
        expected = {field: reference[field] for field in fields}
        for arm in arms[1:]:
            assert {field: indexed[arm][row_id][field] for field in fields} == expected


def test_signed_response_definition_from_primitive_logits_for_every_row(documents):
    source, _ = documents
    r589 = load_r589()
    conditions = set()
    for rows in source["fit_raw"].values():
        for row in rows:
            conditions.add(row["condition"])
            assert r589.signed_response(row) == pytest.approx(independent_response(row), abs=2e-7)
    assert conditions == set(r589.EXPECTED_CONDITIONS)


def test_exact_54_cross_site_pairs(documents):
    _, result = documents
    reports = result["all_pair_reports"]
    assert result["cross_site_pair_count"] == len(reports) == 54
    # Six unordered site pairs crossed with all 3x3 component pairs.
    assert 6 * 3 * 3 == 54
    keys = set()
    for report in reports:
        arm_a, arm_b = report["arm_a"], report["arm_b"]
        assert int(arm_a.split("_", 1)[0][3:]) != int(arm_b.split("_", 1)[0][3:])
        keys.add(tuple(sorted((arm_a, arm_b))))
    assert len(keys) == 54


def test_every_saved_cell_and_leave_out_correlation_recomputes(documents):
    source, result = documents
    by_arm = {
        arm: {row["row_id"]: row for row in rows}
        for arm, rows in source["fit_raw"].items()
    }
    ordered_ids = sorted(next(iter(by_arm.values())))
    for report in result["all_pair_reports"]:
        arm_a, arm_b = report["arm_a"], report["arm_b"]
        meta = by_arm[arm_a]

        def correlation(ids):
            return independent_correlation(
                (independent_response(by_arm[arm_a][row_id]) for row_id in ids),
                (independent_response(by_arm[arm_b][row_id]) for row_id in ids),
            )

        assert report["overall_correlation"] == pytest.approx(correlation(ordered_ids), abs=2e-15)
        for representation in ("list", "digit", "word"):
            for level in (0, 1):
                ids = [
                    row_id for row_id in ordered_ids
                    if meta[row_id]["representation"] == representation
                    and int(meta[row_id]["source_level"]) == level
                ]
                assert len(ids) == 96
                key = f"{representation}:source{level}"
                assert report["cell_correlations"][key] == pytest.approx(correlation(ids), abs=2e-15)
        for representation in ("list", "digit", "word"):
            ids = [row_id for row_id in ordered_ids if meta[row_id]["representation"] != representation]
            assert len(ids) == 384
            assert report["leave_one_representation_out_correlations"][representation] == pytest.approx(
                correlation(ids), abs=2e-15
            )
        for level in (0, 1):
            ids = [row_id for row_id in ordered_ids if int(meta[row_id]["source_level"]) != level]
            assert len(ids) == 288
            assert report["leave_one_source_out_correlations"][f"source{level}"] == pytest.approx(
                correlation(ids), abs=2e-15
            )


def test_result_exactly_replays_current_source(documents):
    source, result = documents
    r589 = load_r589()
    assert r589.analyze(source, EXPECTED_SOURCE_SHA256) == result


def test_no_lead_decision_is_only_a_post_outcome_filter_result(documents):
    _, result = documents
    assert result["evidence_level"] == "screen_only"
    assert result["discovery_filter"]["confirmatory_status"] == "post_outcome_filter_not_a_registered_gate"
    assert result["discovery_leads"] == []
    assert result["decision"] == "no_pair_passed_recorded_post_outcome_filter"
    assert all(not report["discovery_lead"] for report in result["all_pair_reports"])
    # The best pair clears the chosen 0.60 cell floor but narrowly misses one
    # 0.75 representation leave-out floor and more clearly misses the source
    # leave-out floor.  Thus the label is a threshold result, not evidence that
    # the candidate response profiles contain no cross-site similarity.
    best = result["all_pair_reports"][0]
    assert best["minimum_cell_correlation"] >= 0.60
    assert best["minimum_leave_one_representation_out_correlation"] < 0.75
    assert best["minimum_leave_one_source_out_correlation"] < 0.75
    assert best["overall_correlation"] > 0.75


def test_result_binds_analysis_and_primary_test_source_hash(documents):
    _, result = documents
    assert result["implementation_sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    primary_test = ROOT / "ops/test_numbered_list_downstream_response_grouping_rung589.py"
    assert result["primary_test_sha256"] == hashlib.sha256(primary_test.read_bytes()).hexdigest()


def test_target_validator_fails_closed_on_cross_arm_metadata_drift(documents):
    source, _ = documents
    r589 = load_r589()
    corrupted = json.loads(json.dumps(source))
    arm = sorted(corrupted["fit_raw"])[-1]
    corrupted["fit_raw"][arm][0]["representation"] = "digit"
    with pytest.raises(ValueError, match="semantic row metadata differs"):
        r589.analyze(corrupted, "synthetic")
