#!/usr/bin/env python3
"""Thin SELECT completion of the exact shared/contrast two-factor factorial."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import circuit_fast_screen_candidate_bracket_l13h8_shared_contrast_interaction_select as authority
import run_bracket_l13h8_semantic_open_shared_contrast as parent


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "circuits" / "fast_screens" / "bracket_l13h8_shared_contrast_interaction_select_v1_result.json"


def evaluate_interaction(model, torch, F, facade):
    """Record the frozen parent's five arms, then add exactly one joint-removal call."""
    captured = []
    original = parent.shared.factor_forward
    def recording_forward(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result)
        return result
    parent.shared.factor_forward = recording_forward
    try:
        records, replay_error = parent.evaluate(model, torch, F, facade)
    finally:
        parent.shared.factor_forward = original
    assert len(captured) == 4  # replay, natural swap, contrast removed, shared removed
    replay_logits, factors = captured[0]
    contrast_logits = captured[2][0]
    shared_logits = captured[3][0]
    tokens, finals, sources = parent.pad_rows(authority.ROWS, torch, next(model.parameters()).device)
    arange = torch.arange(len(authority.ROWS), device=tokens.device)
    terms = factors["p"][arange, sources].unsqueeze(-1) * factors["u"][arange, sources]
    both_logits = original(model, tokens, finals, {}, torch, F, facade,
                           replacement_terms=torch.zeros_like(terms), source_positions=sources)[0]
    for index, (row, record) in enumerate(zip(authority.ROWS, records)):
        q, answer = int(finals[index]), row["answer_id"]
        contrast_type, contrast_common = parent.coding.closer_type_and_common_axes(
            contrast_logits[index, q], parent.CLOSERS, answer)
        shared_type, shared_common = parent.coding.closer_type_and_common_axes(
            shared_logits[index, q], parent.CLOSERS, answer)
        both_type, both_common = parent.coding.closer_type_and_common_axes(
            both_logits[index, q], parent.CLOSERS, answer)
        record.update(contrast_removed_type_axis=float(contrast_type),
                      contrast_removed_common_axis=float(contrast_common),
                      shared_removed_type_axis=float(shared_type),
                      shared_removed_common_axis=float(shared_common),
                      both_removed_type_axis=float(both_type),
                      both_removed_common_axis=float(both_common))
    return records, replay_error


def score(records, replay_error):
    bars = authority.compile_plan()["bars"]
    by_family = defaultdict(list)
    for row in records:
        i_type = (row["both_removed_type_axis"] - row["shared_removed_type_axis"]
                  - row["contrast_removed_type_axis"] + row["native_type_axis"])
        i_common = (row["both_removed_common_axis"] - row["shared_removed_common_axis"]
                    - row["contrast_removed_common_axis"] + row["native_common_axis"])
        scale = math.hypot(row["both_removed_type_axis"] - row["native_type_axis"],
                           row["both_removed_common_axis"] - row["native_common_axis"]) + 1e-6
        row["mobius_interaction_type_axis"] = i_type
        row["mobius_interaction_common_axis"] = i_common
        row["normalized_interaction_magnitude"] = math.hypot(i_type, i_common) / scale
        by_family[row["family_id"]].append(row)
    reports = {}
    for family, rows in by_family.items():
        reports[family] = {
            "median_normalized_interaction": statistics.median(
                row["normalized_interaction_magnitude"] for row in rows),
            "mean_type_axis_interaction": sum(row["mobius_interaction_type_axis"] for row in rows) / len(rows),
            "mean_common_axis_interaction": sum(row["mobius_interaction_common_axis"] for row in rows) / len(rows),
            "natural_swap_positive_fraction": sum(row["natural_swap_type_transfer"] > 0 for row in rows) / len(rows),
        }
    instrument = {
        "native_capability": min(sum(row["native_type_axis"] > 0 for row in rows) / len(rows)
                                 for rows in by_family.values())
        >= bars["native_positive_fraction_each_family_min"],
        "native_replay": replay_error <= bars["native_replay_max_absolute_logit_error_max"],
        "live_terms": min(row["semantic_open_term_norm"] for row in records)
        >= bars["semantic_open_term_norm_min"],
        "natural_swap": min(reports[family]["natural_swap_positive_fraction"]
                            for family in authority.TARGET_FAMILIES)
        >= bars["natural_swap_positive_type_transfer_fraction_each_target_family_min"]
    }
    additive = all(reports[family]["median_normalized_interaction"]
                   <= bars["median_normalized_interaction_each_family_max"]
                   for family in authority.TARGET_FAMILIES)
    live = all(instrument.values())
    return {"instrument_checks": instrument, "instrument_live": live,
            "family_reports": reports,
            "additive_oblique_held": live and additive,
            "large_interaction_held": live and not additive,
            "predictions": {"pred_a_instrument_live": live,
                            "pred_b_additive_oblique": live and additive,
                            "pred_c_nonlinear_interaction": live and not additive}}


def main():
    plan = authority.compile_plan()
    if os.environ.get("BQLIB_DRYRUN") == "1" or "--dry-run" in sys.argv:
        print(json.dumps(plan, indent=2, sort_keys=True)); return
    if OUT.exists(): raise RuntimeError(f"refusing to overwrite {OUT}")
    parent.authority = authority
    parent.shared.candidate = authority
    torch, F, facade = parent.shared._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad(): records, replay_error = evaluate_interaction(model, torch, F, facade)
    screen = score(records, replay_error)
    terminal = "invalid" if not screen["instrument_live"] else "screen"
    result = {"schema": "bracket_l13h8_shared_contrast_interaction_select_result_v1",
              "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "native_replay_max_absolute_logit_error": replay_error, "raw": records, "screen": screen,
              "evaluated_splits": ["SELECT"], "forbidden_splits_opened": [],
              "model_forwards": 6, "terminal": terminal,
              "reason": "invalid_instrument" if terminal == "invalid" else
                        "additive_oblique" if screen["additive_oblique_held"] else "large_interaction"}
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"result": str(OUT), "terminal": terminal, "reason": result["reason"]}, indent=2))


if __name__ == "__main__": main()
