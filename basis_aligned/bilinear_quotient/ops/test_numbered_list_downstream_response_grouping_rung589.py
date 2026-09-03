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
    key = "arithmetic_minus_structural" if condition == "step_two" else "margin"
    return {
        "row_id": row_id,
        "split": "FIT",
        "representation": representation,
        "source_level": source_level,
        "condition": condition,
        "site": site,
        "component": component,
        "arm": arm,
        "native": {key: value},
        "intervened": {key: 0.0},
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
        "native": {"arithmetic_minus_structural": 2.0, "margin": 999.0},
        "intervened": {"arithmetic_minus_structural": 0.5, "margin": -999.0},
    }
    assert R589.signed_response(item) == pytest.approx(1.5)


def test_nonfinite_response_fails_closed() -> None:
    item = {
        "row_id": "x",
        "condition": "factorial_copy",
        "native": {"margin": float("inf")},
        "intervened": {"margin": 0.0},
    }
    with pytest.raises(ValueError, match="non-finite"):
        R589.signed_response(item)
