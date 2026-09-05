#!/usr/bin/env python3
"""Thin fresh confirmation facade for the L13H8 OPEN/POST payload split."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_open_post_confirm as authority
import run_bracket_l13h8_source_region_payload_factorial as shared


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_open_post_family_confirm_v1_result.json"


def score(raw: list[dict], replay_error: float, native_capability: list[dict]) -> dict:
    bars = authority.compile_plan()["bars"]
    grouped = defaultdict(list)
    for cell in raw:
        grouped[(cell["family_id"], cell["direction"], cell["condition"])].append(cell)
    native = defaultdict(list)
    for cell in native_capability:
        native[(cell["family_id"], cell["direction"])].append(cell["answer_margin"])
    native_fraction = {"|".join(key): sum(value > 0 for value in values) / len(values)
                       for key, values in sorted(native.items())}
    metrics, family_passes = {}, []
    for direction in shared.DIRECTIONS:
        family_values = {}
        for family in authority.TARGET_FAMILIES:
            by_row = defaultdict(dict)
            for condition in ("payload_OPEN", "payload_POST", "payload_OPEN+POST"):
                for cell in grouped[(family, direction, condition)]:
                    by_row[cell["row_id"]][condition] = cell["normalized_effect"]
            opens = [values["payload_OPEN"] for values in by_row.values()]
            posts = [values["payload_POST"] for values in by_row.values()]
            combined = [values["payload_OPEN+POST"] for values in by_row.values()]
            interaction = [both - opened - post for opened, post, both in zip(opens, posts, combined)]
            family_values[family] = {
                "open_median_recovery": statistics.median(opens),
                "post_mean_absolute_recovery": sum(map(abs, posts)) / len(posts),
                "open_post_interaction_mean_absolute": sum(map(abs, interaction)) / len(interaction),
                "open_vs_open_post_mean_absolute_difference":
                    sum(abs(opened - both) for opened, both in zip(opens, combined)) / len(opens),
                "interaction_values": interaction,
            }
        gap = (family_values["completed_then_reopened"]["open_median_recovery"]
               - family_values["direct_type"]["open_median_recovery"])
        passed = bool(
            family_values["direct_type"]["open_median_recovery"] >= bars["direct_open_median_recovery_min"]
            and family_values["completed_then_reopened"]["open_median_recovery"]
            >= bars["completed_open_median_recovery_min"]
            and gap >= bars["open_between_family_gap_min"]
            and all(values["post_mean_absolute_recovery"] <= bars["post_mean_absolute_recovery_max"]
                    and values["open_post_interaction_mean_absolute"]
                    <= bars["open_post_interaction_mean_absolute_max"]
                    and values["open_vs_open_post_mean_absolute_difference"]
                    <= bars["open_vs_open_post_mean_absolute_difference_max"]
                    for values in family_values.values())
        )
        metrics[direction] = {"families": family_values, "open_between_family_gap": gap, "passed": passed}
        family_passes.append(passed)
    complete = [cell for cell in raw if cell["role"] == "target" and cell["condition"] == "complete_head"]
    complete_positive = sum(cell["effect"] > 0 for cell in complete) / len(complete)
    controls = [cell for cell in raw if cell["role"] == "control" and cell["condition"].startswith("payload_")]
    control_abs = sum(abs(cell["effect"]) for cell in controls) / len(controls)
    control_ratio = sum(abs(cell["normalized_effect"]) for cell in controls) / len(controls)
    instrument = {
        "native_capability": min(native_fraction.values())
        >= bars["native_positive_fraction_each_family_direction_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "complete_head_ceiling": complete_positive >= bars["complete_head_target_positive_fraction_min"],
        "same_state_controls": control_abs <= bars["control_mean_absolute_closer_margin_change_max"]
        and control_ratio <= bars["control_mean_absolute_fraction_of_complete_head_max"]
    }
    live = all(instrument.values())
    held = live and all(family_passes)
    return {"instrument_checks": instrument, "instrument_live": live,
            "opener_payload_hypothesis_held": held,
            "predictions": {"pred_a_instrument_live": live,
                            "pred_b_opener_payload_held": held,
                            "pred_c_post_or_synergy_material": live and not held},
            "family_direction_metrics": metrics,
            "native_positive_fraction_by_family_direction": native_fraction,
            "complete_head_target_positive_fraction": complete_positive,
            "control_mean_absolute_closer_margin_change": control_abs,
            "control_mean_absolute_fraction_of_complete_head": control_ratio}


def main() -> None:
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite {OUT}")
    shared.candidate = authority
    torch, F, facade = shared._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        raw, replay_error, native_capability = shared.evaluate(model, torch, F, facade)
    screen = score(raw, replay_error, native_capability)
    terminal = "screen" if screen["opener_payload_hypothesis_held"] else ("null" if screen["instrument_live"] else "invalid")
    result = {"schema": "bracket_l13h8_open_post_family_confirm_result_v1", "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error, "native_capability": native_capability,
              "raw": raw, "summary": shared.summarize(raw), "screen": screen,
              "evaluated_splits": ["FRESH_CONFIRM"], "forbidden_splits_opened": [],
              "model_forwards": plan["price"]["model_forwards"], "terminal": terminal,
              "reason": "opener_payload_confirmed" if terminal == "screen" else
                        "opener_payload_not_confirmed" if terminal == "null" else "instrument_failed"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": result["model_forwards"]}, indent=2))


if __name__ == "__main__":
    main()
