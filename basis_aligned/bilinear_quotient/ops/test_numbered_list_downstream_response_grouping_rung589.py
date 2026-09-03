#!/usr/bin/env python3
"""Model-free tests for R589."""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("numbered_list_downstream_response_grouping_rung589.py")
SPEC = importlib.util.spec_from_file_location("r589", SCRIPT)
assert SPEC and SPEC.loader
R589 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R589)


def row(row_id: str, arm: str, site: int, component: str, index: int) -> dict:
    representation = R589.REPRESENTATIONS[(index // 192) % 3]
    source_level = (index // 96) % 2
    condition = sorted(R589.EXPECTED_CONDITIONS)[index % 6]
    value = float(index % 17) + 0.1 * (index // 17)
    state_key_values = (
        {
            "arithmetic_minus_structural": value,
            "arithmetic_logit": value,
            "structural_logit": 0.0,
        },
        {
            "arithmetic_minus_structural": 0.0,
            "arithmetic_logit": 0.0,
            "structural_logit": 0.0,
        },
    ) if condition == "step_two" else (
        {"margin": value, "answer_logit": value, "max_other_candidate_logit": 0.0},
        {"margin": 0.0, "answer_logit": 0.0, "max_other_candidate_logit": 0.0},
    )
    return {
        "row_id": row_id,
        "group_id": f"group-{index // 6:03d}",
        "split": "FIT",
        "representation": representation,
        "source_level": source_level,
        "source_value": 10 + source_level,
        "condition": condition,
        "action": "step_two" if condition == "step_two" else ("copy" if "copy" in condition else "successor"),
        "token_ids": [index % 101, (index + 1) % 101],
        "query_position": 1,
        "source_position": 0,
        "source_id": index % 101,
        "answer_id": None if condition == "step_two" else (index + 2) % 101,
        "structural_answer_id": (index + 2) % 101 if condition == "step_two" else None,
        "arithmetic_answer_id": (index + 3) % 101 if condition == "step_two" else None,
        "site": site,
        "component": component,
        "arm": arm,
        "native": state_key_values[0],
        "intervened": state_key_values[1],
    }


def planted_document() -> dict:
    raw = {}
    for site in R589.EXPECTED_SITES:
        for component in R589.EXPECTED_COMPONENTS:
            arm = R589.arm_name(site, component)
            raw[arm] = [row(f"row-{i:03d}", arm, site, component, i) for i in range(576)]
    return {"rung": 584, "evaluated_splits": ["FIT"], "forbidden_splits_opened": [], "fit_raw": raw}


def test_planted_identical_cross_site_pair_is_a_screen_lead() -> None:
    result = R589.analyze(planted_document(), "synthetic")
    assert result["evidence_level"] == "screen_only"
    assert result["rows_per_arm"] == 576
    assert result["cross_site_pair_count"] == 54
    assert result["discovery_leads"]
    assert result["discovery_leads"][0]["minimum_cell_correlation"] == pytest.approx(1.0)


def test_source_role_flip_is_rejected_by_cell_stability() -> None:
    document = planted_document()
    arm = R589.arm_name(10, "background_cross")
    for item in document["fit_raw"][arm]:
        if item["source_level"] == 1:
            key = "arithmetic_minus_structural" if item["condition"] == "step_two" else "margin"
            item["native"][key] *= -1.0
            if item["condition"] == "step_two":
                item["native"]["arithmetic_logit"] *= -1.0
            else:
                item["native"]["answer_logit"] *= -1.0
    result = R589.analyze(document, "synthetic")
    target = next(
        report
        for report in result["all_pair_reports"]
        if {report["arm_a"], report["arm_b"]}
        == {R589.arm_name(8, "background_cross"), arm}
    )
    assert target["minimum_cell_correlation"] == pytest.approx(-1.0)
    assert target["discovery_lead"] is False


def test_mismatched_row_membership_fails_closed() -> None:
    document = planted_document()
    document["fit_raw"][R589.arm_name(14, "joint_response")].pop()
    with pytest.raises(ValueError, match="row membership differs"):
        R589.analyze(document, "synthetic")


def test_forbidden_split_fails_closed() -> None:
    document = planted_document()
    document["evaluated_splits"] = ["FIT", "SELECT"]
    with pytest.raises(ValueError, match="FIT-only"):
        R589.analyze(document, "synthetic")


def test_step_two_uses_conflict_preference() -> None:
    item = {
        "row_id": "x",
        "condition": "step_two",
        "native": {"arithmetic_minus_structural": 2.0, "arithmetic_logit": 4.0, "structural_logit": 2.0},
        "intervened": {"arithmetic_minus_structural": 0.5, "arithmetic_logit": 1.5, "structural_logit": 1.0},
    }
    assert R589.signed_response(item) == pytest.approx(1.5)


def test_nonfinite_response_fails_closed() -> None:
    item = {
        "row_id": "x",
        "condition": "factorial_copy",
        "native": {"margin": float("inf"), "answer_logit": float("inf"), "max_other_candidate_logit": 0.0},
        "intervened": {"margin": 0.0, "answer_logit": 0.0, "max_other_candidate_logit": 0.0},
    }
    with pytest.raises(ValueError, match="non-finite"):
        R589.signed_response(item)
