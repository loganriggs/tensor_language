#!/usr/bin/env python3
"""R589: post-outcome, CPU-only functional grouping screen over R584 rows.

This analysis cannot identify a circuit because R584 FIT outcomes were visible before
the grouping rule was chosen.  It asks a narrower discovery question: do removal arms
from different MLP sites produce the same signed downstream response across matched
behavioral rows?  Any lead must be tested by a prospectively frozen joint intervention.

# BQLANE: cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "numbered_list_cached_value_downstream_use_rung584_results.json"
DEFAULT_OUTPUT = ROOT / "numbered_list_downstream_response_grouping_rung589_results.json"
FROZEN_INPUT_SHA256 = "7980753636fab422ed6c609a1afd054f99ed7f903e2bb3e61eddf0617316fdf6"
REPRESENTATIONS = ("list", "digit", "word")
SOURCE_LEVELS = (0, 1)
EXPECTED_CONDITIONS = {
    "factorial_successor",
    "surface_successor",
    "factorial_copy",
    "surface_copy",
    "relation_break",
    "step_two",
}
EXPECTED_SITES = (8, 10, 12, 14)
EXPECTED_COMPONENTS = ("background_cross", "contrast_self", "joint_response")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_name(site: int, component: str) -> str:
    return f"mlp{site}_{component}"


def signed_response(row: dict[str, Any]) -> float:
    """Native minus intervened registered margin.

    Positive means removing the term reduced the registered preference.  The +2
    conflict row uses arithmetic-minus-structural preference; other rows use the
    registered answer margin.
    """

    key = "arithmetic_minus_structural" if row["condition"] == "step_two" else "margin"
    value = float(row["native"][key]) - float(row["intervened"][key])
    if not math.isfinite(value):
        raise ValueError(f"non-finite response for {row['row_id']}")
    return value


def centered_correlation(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = [float(v) for v in xs]
    y = [float(v) for v in ys]
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("correlation requires equal vectors with at least two entries")
    mx, my = sum(x) / len(x), sum(y) / len(y)
    xc = [v - mx for v in x]
    yc = [v - my for v in y]
    nx = math.sqrt(sum(v * v for v in xc))
    ny = math.sqrt(sum(v * v for v in yc))
    if nx == 0.0 or ny == 0.0:
        raise ValueError("constant response vector")
    return sum(a * b for a, b in zip(xc, yc)) / (nx * ny)


def validate_and_index(raw: dict[str, list[dict[str, Any]]]) -> tuple[list[str], dict[str, dict[str, dict[str, Any]]]]:
    expected_arms = [arm_name(site, component) for site in EXPECTED_SITES for component in EXPECTED_COMPONENTS]
    if set(raw) != set(expected_arms):
        raise ValueError(f"arm membership mismatch: got {sorted(raw)}")

    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    authority_ids: set[str] | None = None
    for arm in expected_arms:
        rows = raw[arm]
        by_id = {str(row["row_id"]): row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate row id in {arm}")
        row_ids = set(by_id)
        if authority_ids is None:
            authority_ids = row_ids
        elif row_ids != authority_ids:
            raise ValueError(f"row membership differs in {arm}")
        for row in rows:
            if row["split"] != "FIT":
                raise ValueError("R589 may open FIT only")
            if row["condition"] not in EXPECTED_CONDITIONS:
                raise ValueError(f"unexpected condition {row['condition']}")
            if row["arm"] != arm:
                raise ValueError(f"row arm mismatch in {arm}")
            if int(row["site"]) not in EXPECTED_SITES:
                raise ValueError("unexpected site")
            signed_response(row)
        indexed[arm] = by_id

    assert authority_ids is not None
    ordered_ids = sorted(authority_ids)
    if len(ordered_ids) != 576:
        raise ValueError(f"expected 576 FIT rows per arm, got {len(ordered_ids)}")
    return ordered_ids, indexed


def pair_report(
    arm_a: str,
    arm_b: str,
    ordered_ids: list[str],
    indexed: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    meta = indexed[arm_a]

    def correlation(row_ids: list[str]) -> float:
        return centered_correlation(
            (signed_response(indexed[arm_a][row_id]) for row_id in row_ids),
            (signed_response(indexed[arm_b][row_id]) for row_id in row_ids),
        )

    cell_correlations: dict[str, float] = {}
    for representation in REPRESENTATIONS:
        for source_level in SOURCE_LEVELS:
            cell_ids = [
                row_id
                for row_id in ordered_ids
                if meta[row_id]["representation"] == representation
                and int(meta[row_id]["source_level"]) == source_level
            ]
            if len(cell_ids) != 96:
                raise ValueError(f"expected 96 rows in {representation}/source{source_level}, got {len(cell_ids)}")
            cell_correlations[f"{representation}:source{source_level}"] = correlation(cell_ids)

    leave_representation_out = {
        representation: correlation(
            [row_id for row_id in ordered_ids if meta[row_id]["representation"] != representation]
        )
        for representation in REPRESENTATIONS
    }
    leave_source_out = {
        f"source{source_level}": correlation(
            [row_id for row_id in ordered_ids if int(meta[row_id]["source_level"]) != source_level]
        )
        for source_level in SOURCE_LEVELS
    }
    cell_values = list(cell_correlations.values())
    rep_values = list(leave_representation_out.values())
    source_values = list(leave_source_out.values())
    # These are discovery filters chosen after R584 was visible.  They may rank a
    # lead for confirmation but cannot supply confirmatory evidence themselves.
    discovery_lead = (
        min(cell_values) >= 0.60
        and min(rep_values) >= 0.75
        and min(source_values) >= 0.75
    )
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "overall_correlation": correlation(ordered_ids),
        "cell_correlations": cell_correlations,
        "minimum_cell_correlation": min(cell_values),
        "leave_one_representation_out_correlations": leave_representation_out,
        "minimum_leave_one_representation_out_correlation": min(rep_values),
        "leave_one_source_out_correlations": leave_source_out,
        "minimum_leave_one_source_out_correlation": min(source_values),
        "discovery_lead": discovery_lead,
    }


def analyze(document: dict[str, Any], input_sha: str) -> dict[str, Any]:
    if int(document.get("rung", -1)) != 584:
        raise ValueError("wrong source rung")
    if document.get("evaluated_splits") != ["FIT"] or document.get("forbidden_splits_opened") != []:
        raise ValueError("source did not preserve the FIT-only boundary")
    ordered_ids, indexed = validate_and_index(document["fit_raw"])
    arms = sorted(indexed)
    reports: list[dict[str, Any]] = []
    for i, arm_a in enumerate(arms):
        site_a = int(arm_a.split("_", 1)[0][3:])
        for arm_b in arms[i + 1 :]:
            site_b = int(arm_b.split("_", 1)[0][3:])
            if site_a == site_b:
                continue
            reports.append(pair_report(arm_a, arm_b, ordered_ids, indexed))
    reports.sort(
        key=lambda report: (
            report["discovery_lead"],
            report["minimum_cell_correlation"],
            report["overall_correlation"],
        ),
        reverse=True,
    )
    leads = [report for report in reports if report["discovery_lead"]]
    return {
        "rung": 589,
        "stage": "post_outcome_downstream_response_grouping_screen",
        "evidence_level": "screen_only",
        "source_rung": 584,
        "source_result_sha256": input_sha,
        "opened_splits": ["FIT"],
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "signed_response_definition": (
            "native_minus_intervened_registered_margin; step_two uses "
            "arithmetic_minus_structural, all other conditions use answer margin"
        ),
        "rows_per_arm": len(ordered_ids),
        "cross_site_pair_count": len(reports),
        "discovery_filter": {
            "minimum_representation_by_source_cell_correlation": 0.60,
            "minimum_leave_one_representation_out_correlation": 0.75,
            "minimum_leave_one_source_out_correlation": 0.75,
            "confirmatory_status": "post_outcome_filter_not_a_registered_gate",
        },
        "discovery_leads": leads,
        "all_pair_reports": reports,
        "decision": "freeze_fresh_joint_intervention_for_best_lead" if leads else "no_stable_cross_mlp_grouping_lead",
        "licensed_interpretation": (
            "A lead means two cross-site removals have similar saved FIT response profiles. "
            "It does not establish shared computation, additivity, or an executable grouped circuit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-noncanonical-input", action="store_true")
    args = parser.parse_args()

    input_sha = sha256(args.input)
    if not args.allow_noncanonical_input and input_sha != FROZEN_INPUT_SHA256:
        raise SystemExit(f"source SHA mismatch: {input_sha}")
    with args.input.open() as handle:
        document = json.load(handle)
    result = analyze(document, input_sha)
    encoded = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(encoded)
    print(json.dumps({
        "output": str(args.output),
        "source_sha256": input_sha,
        "lead_count": len(result["discovery_leads"]),
        "decision": result["decision"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
