#!/usr/bin/env python3
"""Exact semantic-OPEN contribution zero-removal using the shared L13H8 executor."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_open_zero_removal as authority
import run_bracket_l13h8_source_region_payload_factorial as shared


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_semantic_open_zero_removal_v1_result.json"


def evaluate(model, torch, F, facade):
    rows, device = authority.ROWS, next(model.parameters()).device
    base_tokens, base_finals = shared.pad(rows, "base", torch, device)
    donor_tokens, donor_finals = shared.pad(rows, "donor", torch, device)
    masks = shared.region_masks(rows, "base", base_tokens.size(1), torch, device)
    native_base = shared.native_logits(model, base_tokens, torch, F)
    native_donor = shared.native_logits(model, donor_tokens, torch, F)
    replay_base, base = shared.factor_forward(model, base_tokens, base_finals, masks, torch, F, facade)
    replay_donor, donor = shared.factor_forward(model, donor_tokens, donor_finals, masks, torch, F, facade)
    replay_error = max(float((native_base - replay_base).abs().max()),
                       float((native_donor - replay_donor).abs().max()))
    zero_base = {"u": torch.zeros_like(base["u"]), "head": torch.zeros_like(base["head"])}
    zero_donor = {"u": torch.zeros_like(donor["u"]), "head": torch.zeros_like(donor["head"])}
    base_head_zero = shared.factor_forward(model, base_tokens, base_finals, masks, torch, F, facade,
                                           donor=zero_base, complete=True)[0]
    donor_head_zero = shared.factor_forward(model, donor_tokens, donor_finals, masks, torch, F, facade,
                                            donor=zero_donor, complete=True)[0]
    base_open_zero = shared.factor_forward(model, base_tokens, base_finals, masks, torch, F, facade,
                                           donor=zero_base, corner=("OPEN",))[0]
    donor_open_zero = shared.factor_forward(model, donor_tokens, donor_finals, masks, torch, F, facade,
                                            donor=zero_donor, corner=("OPEN",))[0]
    records, native_capability, term_norms = [], [], []
    for index, row in enumerate(rows):
        for direction, endpoint, native, replay, head_zero, open_zero, finals, factors in (
            ("base_to_donor", "base", native_base, replay_base, base_head_zero, base_open_zero, base_finals, base),
            ("donor_to_base", "donor", native_donor, replay_donor, donor_head_zero, donor_open_zero, donor_finals, donor),
        ):
            q, answer = int(finals[index]), row[f"{endpoint}_answer_id"]
            native_margin = shared.closer_margin(native[index, q], answer)
            replay_margin = shared.closer_margin(replay[index, q], answer)
            head_margin = shared.closer_margin(head_zero[index, q], answer)
            open_margin = shared.closer_margin(open_zero[index, q], answer)
            head_damage, open_damage = replay_margin - head_margin, replay_margin - open_margin
            normalized = open_damage / head_damage if head_damage > 1e-9 else None
            opener = row[f"{endpoint}_open_position"]
            term = factors["p"][index, opener].float() * factors["u"][index, opener].float()
            term_norm = float(term.norm())
            term_norms.append(term_norm)
            native_capability.append({"row_id": row["row_id"], "family_id": row["family_id"],
                                      "direction": direction, "answer_margin": native_margin})
            records.append({"row_id": row["row_id"], "group_id": row["group_id"],
                            "family_id": row["family_id"], "role": row["role"], "direction": direction,
                            "native_margin": native_margin, "replay_margin": replay_margin,
                            "complete_head_zero_margin": head_margin, "semantic_open_zero_margin": open_margin,
                            "complete_head_margin_damage": head_damage, "semantic_open_margin_damage": open_damage,
                            "normalized_damage": normalized, "answer_preserved": open_margin > 0,
                            "semantic_open_term_norm": term_norm})
    return records, replay_error, native_capability, term_norms


def score(records: list[dict], replay_error: float, native_capability: list[dict], term_norms: list[float]) -> dict:
    bars = authority.compile_plan()["bars"]
    native = defaultdict(list)
    for cell in native_capability:
        native[(cell["family_id"], cell["direction"])].append(cell["answer_margin"])
    native_fraction = {"|".join(key): sum(value > 0 for value in values) / len(values)
                       for key, values in sorted(native.items())}
    targets = [cell for cell in records if cell["role"] == "target"]
    head_positive = sum(cell["complete_head_margin_damage"] > 0 for cell in targets) / len(targets)
    target_metrics, target_passes = {}, []
    for family in authority.TARGET_FAMILIES:
        for direction in shared.DIRECTIONS:
            cells = [cell for cell in targets if cell["family_id"] == family and cell["direction"] == direction]
            values = [cell["normalized_damage"] for cell in cells]
            valid = all(value is not None for value in values)
            metric = {"n": len(cells), "median_normalized_damage": statistics.median(values) if valid else None,
                      "positive_damage_fraction": sum(cell["semantic_open_margin_damage"] > 0 for cell in cells) / len(cells)}
            metric["passed"] = bool(valid
                and metric["median_normalized_damage"]
                >= bars["target_median_normalized_damage_min_each_family_direction"]
                and metric["positive_damage_fraction"]
                >= bars["target_positive_damage_fraction_min_each_family_direction"])
            target_metrics[f"{family}|{direction}"] = metric
            target_passes.append(metric["passed"])
    controls = [cell for cell in records if cell["role"] == "control"]
    control_abs = sum(abs(cell["semantic_open_margin_damage"]) for cell in controls) / len(controls)
    ratios = [abs(cell["normalized_damage"]) for cell in controls if cell["normalized_damage"] is not None]
    control_ratio = sum(ratios) / len(ratios) if len(ratios) == len(controls) else None
    preservation = {}
    for family in authority.CONTROL_FAMILIES:
        for direction in shared.DIRECTIONS:
            cells = [cell for cell in controls if cell["family_id"] == family and cell["direction"] == direction]
            preservation[f"{family}|{direction}"] = sum(cell["answer_preserved"] for cell in cells) / len(cells)
    instrument = {
        "native_capability": min(native_fraction.values())
        >= bars["native_positive_fraction_each_family_direction_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "live_semantic_open_term": min(term_norms) >= bars["semantic_open_term_norm_min"],
        "complete_head_removal_ceiling": head_positive >= bars["full_head_target_damage_positive_fraction_min"]
    }
    collateral = bool(control_ratio is not None
        and control_abs <= bars["control_mean_absolute_margin_damage_max"]
        and control_ratio <= bars["control_mean_absolute_normalized_damage_max"]
        and min(preservation.values()) >= bars["control_answer_preservation_fraction_min_each_family_direction"])
    live = all(instrument.values())
    held = live and all(target_passes) and collateral
    return {"instrument_checks": instrument, "instrument_live": live,
            "selective_necessity_held": held,
            "predictions": {"pred_a_instrument_live": live,
                            "pred_b_selective_necessity_held": held,
                            "pred_c_not_selectively_necessary": live and not held},
            "target_metrics": target_metrics, "same_state_collateral_passed": collateral,
            "control_mean_absolute_margin_damage": control_abs,
            "control_mean_absolute_normalized_damage": control_ratio,
            "control_answer_preservation_by_family_direction": preservation,
            "native_positive_fraction_by_family_direction": native_fraction,
            "minimum_semantic_open_term_norm": min(term_norms),
            "complete_head_target_damage_positive_fraction": head_positive}


def main() -> None:
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    shared.candidate = authority
    torch, F, facade = shared._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad(): records, replay_error, native, norms = evaluate(model, torch, F, facade)
    screen = score(records, replay_error, native, norms)
    terminal = "screen" if screen["selective_necessity_held"] else ("null" if screen["instrument_live"] else "invalid")
    result = {"schema": "bracket_l13h8_semantic_open_zero_removal_result_v1", "plan": plan,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error, "native_capability": native,
              "raw": records, "screen": screen, "evaluated_splits": ["FRESH_HELDOUT_REMOVAL"],
              "forbidden_splits_opened": [], "model_forwards": plan["price"]["model_forwards"],
              "terminal": terminal, "reason": "selective_necessity_held" if terminal == "screen" else
              "selective_necessity_failed" if terminal == "null" else "instrument_failed"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "model_forwards": result["model_forwards"]}, indent=2))


if __name__ == "__main__": main()
