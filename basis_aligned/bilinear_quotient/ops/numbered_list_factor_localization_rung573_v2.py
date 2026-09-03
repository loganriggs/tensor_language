#!/usr/bin/env python3
"""R573 v2: price-only execution repair of the exact list-label factor test."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import json
import time

import torch

import numbered_list_factor_localization_rung573 as parent


AMENDMENT = parent.POLY / "NUMBERED_LIST_FACTOR_LOCALIZATION_RUNG573_V2_IMPLEMENTATION_AMENDMENT.md"
OUT = parent.ROOT / "numbered_list_factor_localization_rung573_v2_results.json"
AMENDMENT_SHA256 = "27729a0e1405221f989ad0f6b9fef5d2f797c137fbe71a044148e9a5e3e0b4d7"
MAXIMUM_FORWARDS = 280


def load_authority():
    rows, positions = parent.load_authority()
    if not AMENDMENT.is_file() or parent.sha256(AMENDMENT) != AMENDMENT_SHA256:
        raise RuntimeError("R573 v2 implementation amendment changed")
    return rows, positions


def main() -> None:
    started = time.time()
    rows, positions = load_authority()
    fit_chunks = parent.count_chunks(rows, "FIT")
    select_chunks = parent.count_chunks(rows, "SELECT")
    assert fit_chunks == 12 and select_chunks == 10
    if parent.os.environ.get("BQLIB_DRYRUN") == "1":
        parent.synthetic_choice_test()
        print(json.dumps({"status": "dryrun_passed", "rung": "573_v2", "rows": len(rows),
                          "fit_chunks": fit_chunks, "select_chunks": select_chunks,
                          "maximum_forwards": MAXIMUM_FORWARDS, "model_backwards": 0,
                          "model_loaded": False, "scientific_protocol_changed": False,
                          "outcome_blind_replication": False,
                          "FINAL_TEST_or_OOD_opened": False}, indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R573 v2 result namespace already exists")
    model, checkpoint = parent.facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    observed_lambda = float(model.transformer.h[parent.LAYER].attn.lamb.detach().cpu())
    if abs(observed_lambda - 4.0) > 1e-7:
        raise RuntimeError(f"layer-8 value mixing coefficient changed: {observed_lambda}")
    fit_raw, fit_execution = parent.evaluate(model, rows, positions, "FIT", parent.ALL_ARMS)
    scales = parent.fit_scales(fit_raw)
    fit_ceiling, fit_reports, fit_ceiling_pass = parent.score(
        fit_raw, parent.ALL_ARMS, scales, parent.SEED)
    choice = parent.choose(fit_reports) if fit_ceiling_pass else parent.choose({})
    select_raw = select_ceiling = select_reports = select_execution = None
    select_ceiling_pass = selected_held = False
    opened = ["FIT"]
    if choice["selected_arm"] is not None:
        select_arms = ("complete_heads", choice["selected_arm"])
        select_raw, select_execution = parent.evaluate(model, rows, positions, "SELECT", select_arms)
        select_ceiling, select_reports, select_ceiling_pass = parent.score(
            select_raw, select_arms, scales, parent.SEED + 1000)
        selected_held = bool(select_ceiling_pass and select_reports[choice["selected_arm"]]["passed"])
        opened.append("SELECT")
    exact = bool(checkpoint.weights_sha256 == parent.facade.WEIGHTS_SHA256
                 and fit_execution["native_replay_relative_squared_error"] <= 1e-12
                 and fit_execution["head_source_sum_relative_squared_error"] <= 1e-10
                 and fit_execution["value_split_relative_squared_error"] <= 1e-10
                 and (select_execution is None or (
                     select_execution["native_replay_relative_squared_error"] <= 1e-12
                     and select_execution["head_source_sum_relative_squared_error"] <= 1e-10
                     and select_execution["value_split_relative_squared_error"] <= 1e-10)))
    total_forwards = fit_execution["model_forwards"] + (
        select_execution["model_forwards"] if select_execution else 0)
    pred_a = bool(exact and fit_ceiling_pass)
    pred_b = bool(pred_a and choice["selected_arm"] is not None)
    pred_c = bool(pred_b and selected_held)
    result = {"rung": "573_v2", "stage": "numbered_list_exact_label_factor_localization_price_repair",
              "implementation_repair_only": True, "outcome_blind_replication": False,
              "v1_leaked_fact": "FIT selected some arm because conditional SELECT opened",
              "pred_a_exact_replay_and_fit_complete_head_ceiling": pred_a,
              "pred_b_fit_exact_factor_selected": pred_b,
              "pred_c_selected_factor_holds_on_select": pred_c,
              "all_gates_pass": bool(pred_a and pred_b and pred_c),
              "heads": [f"L{parent.LAYER}H{head}" for head in parent.HEADS],
              "layer8_value_lambda": observed_lambda, "fit_control_scales": scales,
              "fit_ceiling": fit_ceiling, "fit_factor_reports": fit_reports, "fit_choice": choice,
              "select_ceiling": select_ceiling, "select_factor_reports": select_reports,
              "selected_factor_held": selected_held, "fit_raw": fit_raw, "select_raw": select_raw,
              "execution": {"fit": fit_execution, "select": select_execution,
                            "maximum_forwards": MAXIMUM_FORWARDS},
              "model_forwards": total_forwards, "model_backwards": 0, "model_weights_updated": False,
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "input_sha256": {str(path): parent.sha256(path) for path in parent.HASHES} |
                              {str(AMENDMENT): parent.sha256(AMENDMENT)},
              "evaluated_splits": opened, "forbidden_splits_opened": [],
              "elapsed_seconds": time.time() - started,
              "decision": "held_exact_label_factor" if pred_c else (
                  "complete_head_site_null" if not pred_a else "exact_label_factor_null"),
              "next_step": "compile_selected_factor_to_weights_and_downstream_consumers" if pred_c else
                           "retain_behavior_circuit_and_test_complete_state_cross_format_sites"}
    if total_forwards > MAXIMUM_FORWARDS:
        raise RuntimeError(f"forward price exceeded: {total_forwards} > {MAXIMUM_FORWARDS}")
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key.startswith("pred_")
                      or key in {"fit_choice", "selected_factor_held", "model_forwards",
                                 "evaluated_splits", "decision", "next_step"}}, indent=2))


if __name__ == "__main__":
    main()
