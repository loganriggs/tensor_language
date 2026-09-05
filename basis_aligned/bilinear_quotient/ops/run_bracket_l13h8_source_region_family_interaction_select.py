#!/usr/bin/env python3
"""Thin held-out SELECT facade over the exact L13H8 source-region executor."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_source_regions_select as authority
import run_bracket_l13h8_source_region_payload_factorial as shared


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_source_region_family_interaction_select_v1_result.json"


def score_select(raw: list[dict], replay_error: float, native_capability: list[dict]) -> dict:
    bars = authority.compile_plan()["bars"]
    grouped = defaultdict(list)
    for cell in raw:
        grouped[(cell["family_id"], cell["direction"], cell["condition"])].append(cell)

    native = defaultdict(list)
    for cell in native_capability:
        native[(cell["family_id"], cell["direction"])].append(cell["answer_margin"])
    native_fractions = {"|".join(key): sum(value > 0 for value in values) / len(values)
                        for key, values in sorted(native.items())}

    metrics = {}
    family_checks = []
    for direction in shared.DIRECTIONS:
        direct_prefix = [cell["normalized_effect"] for cell in grouped[("direct_type", direction, "payload_PREFIX")]]
        completed_prefix = [cell["normalized_effect"] for cell in grouped[("completed_then_reopened", direction, "payload_PREFIX")]]
        direct_open = [cell["normalized_effect"] for cell in grouped[("direct_type", direction, "payload_OPEN+POST")]]
        completed_open = [cell["normalized_effect"] for cell in grouped[("completed_then_reopened", direction, "payload_OPEN+POST")]]
        item = {
            "direct_prefix_mean_absolute_recovery": sum(map(abs, direct_prefix)) / len(direct_prefix),
            "completed_prefix_mean_recovery": sum(completed_prefix) / len(completed_prefix),
            "prefix_between_family_gap": (sum(direct_prefix) / len(direct_prefix)
                                           - sum(completed_prefix) / len(completed_prefix)),
            "direct_open_post_median_recovery": statistics.median(direct_open),
            "completed_open_post_median_recovery": statistics.median(completed_open),
            "open_post_between_family_gap": statistics.median(completed_open) - statistics.median(direct_open),
            "direct_open_post_positive_fraction": sum(value > 0 for value in direct_open) / len(direct_open),
            "completed_open_post_positive_fraction": sum(value > 0 for value in completed_open) / len(completed_open),
        }
        item["passed"] = bool(
            item["direct_prefix_mean_absolute_recovery"] <= bars["direct_prefix_mean_absolute_recovery_max"]
            and item["completed_prefix_mean_recovery"] <= bars["completed_prefix_mean_recovery_max"]
            and item["prefix_between_family_gap"] >= bars["prefix_between_family_gap_min"]
            and item["direct_open_post_median_recovery"] >= bars["direct_open_post_median_recovery_min"]
            and item["completed_open_post_median_recovery"] >= bars["completed_open_post_median_recovery_min"]
            and item["open_post_between_family_gap"] >= bars["open_post_between_family_gap_min"]
            and item["direct_open_post_positive_fraction"]
            >= bars["open_post_positive_fraction_each_family_direction_min"]
            and item["completed_open_post_positive_fraction"]
            >= bars["open_post_positive_fraction_each_family_direction_min"]
        )
        metrics[direction] = item
        family_checks.append(item["passed"])

    complete = [cell for cell in raw if cell["role"] == "target" and cell["condition"] == "complete_head"]
    complete_positive = sum(cell["effect"] > 0 for cell in complete) / len(complete)
    controls = [cell for cell in raw if cell["role"] == "control" and cell["condition"].startswith("payload_")]
    control_abs = sum(abs(cell["effect"]) for cell in controls) / len(controls)
    control_ratio = sum(abs(cell["normalized_effect"]) for cell in controls) / len(controls)
    instrument = {
        "native_capability": min(native_fractions.values())
        >= bars["native_positive_fraction_each_family_direction_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "complete_head_ceiling": complete_positive >= bars["complete_head_target_positive_fraction_min"],
        "same_state_controls": (control_abs <= bars["control_mean_absolute_closer_margin_change_max"]
                                and control_ratio <= bars["control_mean_absolute_fraction_of_complete_head_max"]),
    }
    live = all(instrument.values())
    held = live and all(family_checks)
    return {
        "instrument_checks": instrument,
        "instrument_live": live,
        "family_interaction_held": held,
        "predictions": {"pred_a_instrument_live": live,
                        "pred_b_family_interaction_held": held,
                        "pred_c_family_interaction_failed": live and not held},
        "family_direction_metrics": metrics,
        "native_positive_fraction_by_family_direction": native_fractions,
        "complete_head_target_positive_fraction": complete_positive,
        "control_mean_absolute_closer_margin_change": control_abs,
        "control_mean_absolute_fraction_of_complete_head": control_ratio,
    }


def main() -> None:
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    # The shared executor resolves its declarative authority dynamically; no intervention math is copied here.
    shared.candidate = authority
    torch, F, facade = shared._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        raw, replay_error, native_capability = shared.evaluate(model, torch, F, facade)
    screen = score_select(raw, replay_error, native_capability)
    terminal = "screen" if screen["family_interaction_held"] else ("null" if screen["instrument_live"] else "invalid")
    result = {
        "schema": "bracket_source_region_family_interaction_select_result_v1",
        "plan": plan,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "native_replay_max_absolute_logit_error": replay_error,
        "native_capability": native_capability,
        "raw": raw,
        "summary": shared.summarize(raw),
        "screen": screen,
        "evaluated_splits": ["SELECT"],
        "forbidden_splits_opened": [],
        "model_forwards": plan["price"]["model_forwards"],
        "terminal": terminal,
        "reason": ("family_interaction_confirmed" if terminal == "screen" else
                   "family_interaction_not_confirmed" if terminal == "null" else "instrument_failed"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal,
                      "model_forwards": result["model_forwards"]}, indent=2))


if __name__ == "__main__":
    main()
