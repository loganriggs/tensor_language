#!/usr/bin/env python3
"""Sign-gauge validation + the causally available reverse direction.

# BQGATE: EXPERIMENT
# pred_a_exact_live_validation_instrument
# pred_b_forward_gauge_validates_500_1000
# pred_c_reverse_L8H4_to_L8H3_edge_on_discovery
# pred_d_reverse_validates_500_1000
# pred_e_mutual_sign_gauge_licensed

Parallel-lane successor to the sign-gauge quotient FULL PASS (section 2630).
Runs the two passed negated directions plus the one causally available reverse
(same-layer L8H4->L8H3, negated) over all four document quarters, under rung
501's verbatim edge criteria. Imports rung 501's module and pins the probe
receipt hash-exactly; modifies no registered file. Preregistration:
polynomial_causal/EQUALITY_SCORE_SIGN_GAUGE_VALIDATION_AND_REVERSE_PREREGISTRATION.md
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump

ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_score_directed_action_graph_rung501 as r501

PREREG = POLY / "EQUALITY_SCORE_SIGN_GAUGE_VALIDATION_AND_REVERSE_PREREGISTRATION.md"
R501_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
R501_RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
PROBE_SOURCE = ROOT / "ops/equality_score_sign_gauge_quotient.py"
PROBE_RESULT = ROOT / "equality_score_sign_gauge_quotient_results.json"
OUT = ROOT / "equality_score_sign_gauge_validation_results.json"
BUNDLE = ROOT / "equality_score_sign_gauge_validation_bundle.pt"
HASHES = {
    PREREG: "06dae216c588942fb2dff01c7ccb77c12ecb7888f12b2143fd608c67b8a729aa",
    R501_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    R501_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
    PROBE_SOURCE: "a6007d7faff747b8467d50ef6ae934a9bc7b617735b8f01c808b417f142a00c7",
    PROBE_RESULT: "eff94038395d4da9571f5ace8c9e69f5a18aae2382c6385b5724c8937d7ef8b9",
}

MY_PAIRS = ((1, 3), (2, 3), (3, 2))
MY_NAMES = tuple(f"{r501.TERMS[l]}->{r501.TERMS[r]}" for l, r in MY_PAIRS)
FORWARD_NAMES = ("L7H3->L8H4", "L8H3->L8H4")
REVERSE_NAME = "L8H4->L8H3"
HYBRIDS = ("neg_score", "pos_score", "neg_payload")
HYBRID_STATES = {"neg_score": "score_donor", "pos_score": "score_donor",
                 "neg_payload": "payload_donor"}
ANALYSIS_SLOT = {"neg_score": "score_donor", "neg_payload": "payload_donor",
                 "pos_score": "positive_control"}
BACKGROUNDS = r501.BACKGROUNDS
CELLS = r501.CELLS
BATCH = r501.BATCH
QUARTERS = ((0, 250), (250, 500), (500, 750), (750, 1000))
DOCS = 1000
BATCHES = DOCS // BATCH
FORWARDS_PER_BATCH = 2 + len(MY_PAIRS) * 9
EXPECTED_PHASE_FORWARDS = BATCHES * FORWARDS_PER_BATCH


def _modified_scales(scales, name, hybrid):
    row = dict(scales[name])
    if hybrid == "neg_score":
        row["score_ratio"] = -row["score_ratio"]
    elif hybrid == "neg_payload":
        row["payload_ratio"] = -row["payload_ratio"]
    return row


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or r501.sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    probe = json.loads(PROBE_RESULT.read_text())
    required = {
        "pred_a_exact_live_sign_gauge_instrument": True,
        "pred_b_L7H3_score_is_L8H4_computation_up_to_sign": True,
        "pred_c_gauge_extends_to_L8H3": True,
        "strong_null": False,
    }
    if any(probe.get(key) != value for key, value in required.items()):
        raise RuntimeError("probe receipt does not license validation+reverse")
    rows, metadata = r501.validate_inputs()
    return rows, {**metadata, "gauge_pairs": list(MY_NAMES),
                  "hybrids": list(HYBRIDS), "quarters": [list(q) for q in QUARTERS]}


def _empty_reader_stats():
    shape = (len(MY_PAIRS), len(BACKGROUNDS), len(HYBRIDS), len(QUARTERS), len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("ref2", "hyb2", "cross", "write2", "tokens")}


def _empty_background_stats():
    shape = (len(MY_PAIRS), len(QUARTERS), len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("present2", "absent2", "cross", "tokens")}


@torch.no_grad()
def collect(model, rows, scales):
    states_order = ("late_native", "late_absent") + HYBRIDS
    task_sums = torch.zeros(len(MY_PAIRS), len(BACKGROUNDS), len(states_order),
                            DOCS, len(CELLS), dtype=torch.float64)
    task_counts = torch.zeros(DOCS, len(CELLS), dtype=torch.float64)
    reader_stats = _empty_reader_stats()
    background_stats = _empty_background_stats()
    task_masks = r501._task_masks(rows)
    diagnostics = {
        "native_replay_logit_max_abs": 0.0, "native_replay_mlp9_max_abs": 0.0,
        "factor_reconstruction_max": 0.0, "minimum_nonzero_edit_rms": math.inf,
        "zero_intended_edit_actions": 0, "capture_calls": 0,
    }
    calls = {"native": 0, "analytical": 0}
    device = next(model.parameters()).device
    for start in range(0, DOCS, BATCH):
        stop = start + BATCH
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks_cpu = {cell: task_masks[cell][start:stop] for cell in CELLS}
        masks_gpu = {cell: value.to(device) for cell, value in masks_cpu.items()}
        global_rows = torch.arange(start, stop, device=device)
        quarter_conditions = [
            (global_rows >= lo) & (global_rows < hi) for lo, hi in QUARTERS]
        native_logits, native_write, _diag, _audit = r501._captured_forward(
            model, tokens, direct=True)
        replay_logits, replay_write, diag, _audit = r501._captured_forward(
            model, tokens, pair=None)
        calls["native"] += 1
        calls["analytical"] += 1
        diagnostics["capture_calls"] += 2
        diagnostics["native_replay_logit_max_abs"] = max(
            diagnostics["native_replay_logit_max_abs"],
            float((native_logits.float() - replay_logits.float()).abs().max()))
        diagnostics["native_replay_mlp9_max_abs"] = max(
            diagnostics["native_replay_mlp9_max_abs"],
            float((native_write.float() - replay_write.float()).abs().max()))
        diagnostics["factor_reconstruction_max"] = max(
            diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
        replay_sums, observed_counts = r501._cell_sums(replay_logits, batch_rows, masks_cpu)
        task_counts[start:stop] = observed_counts
        for pair_index, pair in enumerate(MY_PAIRS):
            name = MY_NAMES[pair_index]
            neg_responses = {}
            for background_index, background in enumerate(BACKGROUNDS):
                writes, sums = {}, {}
                if background == "early_present":
                    writes["late_native"] = replay_write
                    sums["late_native"] = replay_sums
                plan = ([] if background == "early_present" else ["late_native"])
                plan += ["late_absent"] + list(HYBRIDS)
                for arm in plan:
                    state = HYBRID_STATES.get(arm, arm)
                    arm_scales = (_modified_scales(scales, name, arm)
                                  if arm in HYBRIDS else
                                  (scales[name] if state.endswith("donor") else None))
                    logits, write, diag, _audit = r501._captured_forward(
                        model, tokens, pair=pair, background=background,
                        state=state, scales=arm_scales)
                    calls["analytical"] += 1
                    diagnostics["capture_calls"] += 1
                    diagnostics["factor_reconstruction_max"] = max(
                        diagnostics["factor_reconstruction_max"],
                        diag["factor_reconstruction_max"])
                    if background == "early_absent" and diag["early_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    if state != "late_native" and diag["late_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    for key in ("early_edit_rms", "late_edit_rms"):
                        if diag[key] > 0:
                            diagnostics["minimum_nonzero_edit_rms"] = min(
                                diagnostics["minimum_nonzero_edit_rms"], diag[key])
                    cell_sums, counts = r501._cell_sums(logits, batch_rows, masks_cpu)
                    if not torch.equal(counts, observed_counts):
                        raise RuntimeError("task supports changed across actions")
                    writes[arm] = write
                    sums[arm] = cell_sums
                    del logits
                for state_index, arm in enumerate(states_order):
                    task_sums[pair_index, background_index, state_index,
                              start:stop] = sums[arm]
                absent = writes["late_absent"].float()
                native = writes["late_native"].float()
                reference = absent - native
                for hybrid_index, hybrid in enumerate(HYBRIDS):
                    response = absent - writes[hybrid].float()
                    for quarter, condition in enumerate(quarter_conditions):
                        for cell_index, cell in enumerate(CELLS):
                            selected = masks_gpu[cell] & condition[:, None]
                            if not bool(selected.any()):
                                continue
                            ref = reference[selected].double()
                            hyb = response[selected].double()
                            writer = native[selected].double()
                            index = (pair_index, background_index, hybrid_index,
                                     quarter, cell_index)
                            reader_stats["ref2"][index] += ref.square().sum().cpu()
                            reader_stats["hyb2"][index] += hyb.square().sum().cpu()
                            reader_stats["cross"][index] += (ref * hyb).sum().cpu()
                            reader_stats["write2"][index] += writer.square().sum().cpu()
                            reader_stats["tokens"][index] += int(selected.sum())
                neg_responses[background] = absent - writes["neg_score"].float()
            for quarter, condition in enumerate(quarter_conditions):
                for cell_index, cell in enumerate(CELLS):
                    selected = masks_gpu[cell] & condition[:, None]
                    if not bool(selected.any()):
                        continue
                    left = neg_responses["early_present"][selected].double()
                    right = neg_responses["early_absent"][selected].double()
                    index = (pair_index, quarter, cell_index)
                    background_stats["present2"][index] += left.square().sum().cpu()
                    background_stats["absent2"][index] += right.square().sum().cpu()
                    background_stats["cross"][index] += (left * right).sum().cpu()
                    background_stats["tokens"][index] += int(selected.sum())
        del native_logits, replay_logits, native_write, replay_write
    diagnostics["calls"] = calls
    diagnostics["expected_calls"] = {
        "native": BATCHES, "analytical": BATCHES * (FORWARDS_PER_BATCH - 1)}
    diagnostics["calls_exact"] = calls == diagnostics["expected_calls"]
    diagnostics["capture_calls_exact"] = (
        diagnostics["capture_calls"] == BATCHES * FORWARDS_PER_BATCH)
    diagnostics["support"] = {
        cell: [int(task_masks[cell][lo:hi].sum()) for lo, hi in QUARTERS]
        for cell in CELLS}
    return {"task_sums": task_sums, "task_counts": task_counts,
            "reader_stats": reader_stats, "background_stats": background_stats,
            "states_order": states_order, "diagnostics": diagnostics}


def analyze(collection):
    task_sums = collection["task_sums"]
    counts = collection["task_counts"]
    states_order = collection["states_order"]
    analysis, background_cmp = {}, {}
    for pair_index, name in enumerate(MY_NAMES):
        analysis[name] = {}
        for background_index, background in enumerate(BACKGROUNDS):
            analysis[name][background] = {}
            absent = task_sums[pair_index, background_index,
                               states_order.index("late_absent")]
            native = task_sums[pair_index, background_index,
                               states_order.index("late_native")]
            for hybrid_index, hybrid in enumerate(HYBRIDS):
                hybrid_sums = task_sums[pair_index, background_index,
                                        states_order.index(hybrid)]
                rows_out = []
                for quarter, (lo, hi) in enumerate(QUARTERS):
                    ci = CELLS.index("copy_positive")
                    denom = counts[lo:hi, ci].clamp_min(1)
                    native_rows = (absent[lo:hi, ci] - native[lo:hi, ci]) / denom
                    hybrid_rows = (absent[lo:hi, ci] - hybrid_sums[lo:hi, ci]) / denom
                    task = r501.action_parent._fit_report(native_rows, hybrid_rows)
                    native_total = float((absent[lo:hi, ci] - native[lo:hi, ci]).sum())
                    hybrid_total = float((absent[lo:hi, ci] - hybrid_sums[lo:hi, ci]).sum())
                    recovery = (hybrid_total / native_total
                                if abs(native_total) > 1e-30 else None)
                    off = CELLS.index("all_noncopy")
                    off_change = float(
                        (hybrid_sums[lo:hi, off] - native[lo:hi, off]).sum()
                        / counts[lo:hi, off].sum().clamp_min(1))
                    reader = {
                        cell: r501._reader_report(
                            collection["reader_stats"],
                            (pair_index, background_index, hybrid_index,
                             quarter, cell_index))
                        for cell_index, cell in enumerate(CELLS)}
                    rows_out.append({
                        "equality_recovery": recovery, "task_effect": task,
                        "off_target_signed_mean_hybrid_minus_native_nat": off_change,
                        "reader": reader,
                    })
                analysis[name][background][ANALYSIS_SLOT[hybrid]] = rows_out
        background_cmp[name] = [
            {cell: r501._background_report(collection["background_stats"],
                                           (pair_index, quarter, cell_index))
             for cell_index, cell in enumerate(CELLS)}
            for quarter in range(len(QUARTERS))]
    return analysis, background_cmp


def _positive_control_check(analysis, name, quarter):
    rows = []
    for background in BACKGROUNDS:
        row = analysis[name][background]["positive_control"][quarter]
        rows.append({
            "background": background,
            "task_cosine": row["task_effect"]["cosine"],
            "reader_cosine": row["reader"]["copy_positive"]["cosine"],
            "holds": bool(row["task_effect"]["cosine"] < 0
                          and row["reader"]["copy_positive"]["cosine"] < 0),
        })
    return {"holds": all(row["holds"] for row in rows), "backgrounds": rows}


def _direction_report(analysis, background_cmp, name, quarters):
    edges = [r501._partition_edge(analysis, background_cmp, name, quarter)
             for quarter in quarters]
    controls = [_positive_control_check(analysis, name, quarter)
                for quarter in quarters]
    return {
        "quarters": list(quarters),
        "edge_all": bool(all(row["edge"] for row in edges)),
        "control_all": bool(all(row["holds"] for row in controls)),
        "holds": bool(all(row["edge"] for row in edges)
                      and all(row["holds"] for row in controls)),
        "partitions": edges, "positive_control": controls,
    }


def score(analysis, background_cmp, collection, scale_diagnostics, checkpoint):
    diagnostics = collection["diagnostics"]
    pred_a = bool(
        checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        and diagnostics["native_replay_logit_max_abs"] == 0.0
        and diagnostics["native_replay_mlp9_max_abs"] == 0.0
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["minimum_nonzero_edit_rms"] > 0
        and diagnostics["zero_intended_edit_actions"] == 0
        and diagnostics["calls_exact"] and diagnostics["capture_calls_exact"]
        and scale_diagnostics["calls_exact"] and scale_diagnostics["all_scales_live"]
        and scale_diagnostics["factor_reconstruction_max"] <= 1e-10
        and all(min(values) > 0 for values in diagnostics["support"].values())
        and bool((collection["reader_stats"]["tokens"] > 0).all()))
    reports = {}
    for name in FORWARD_NAMES:
        reports[name] = {
            "validation": _direction_report(analysis, background_cmp, name, (2, 3)),
            "discovery_re_measured": _direction_report(
                analysis, background_cmp, name, (0, 1)),
        }
    reports[REVERSE_NAME] = {
        "discovery": _direction_report(analysis, background_cmp, REVERSE_NAME, (0, 1)),
        "validation": _direction_report(analysis, background_cmp, REVERSE_NAME, (2, 3)),
    }
    pred_b = bool(all(reports[name]["validation"]["holds"] for name in FORWARD_NAMES))
    pred_c = reports[REVERSE_NAME]["discovery"]["holds"]
    pred_d = reports[REVERSE_NAME]["validation"]["holds"]
    pred_e = bool(pred_a and pred_b and pred_c and pred_d)
    return pred_a, pred_b, pred_c, pred_d, pred_e, reports


def _synthetic_row(cos, rec):
    reader_cell = {"cosine": cos, "positive_fit_scale": abs(cos),
                   "scaled_residual": max(0.0, 1 - abs(cos)),
                   "native_response_rms_over_native_write_rms": .1,
                   "hybrid_response_rms_over_native_write_rms": .1, "tokens": 100}
    weak = dict(reader_cell); weak["cosine"] = cos - .5
    return {"equality_recovery": rec,
            "task_effect": {"cosine": cos, "positive_fit_scale": 1.0,
                            "scaled_residual": max(0.0, 1 - abs(cos))},
            "off_target_signed_mean_hybrid_minus_native_nat": 0.001,
            "reader": {"copy_positive": reader_cell,
                       "noncopy_equality": weak, "all_noncopy": weak}}


def _synthetic_analysis():
    analysis, background_cmp = {}, {}
    for name in MY_NAMES:
        analysis[name] = {}
        for background in BACKGROUNDS:
            analysis[name][background] = {
                "score_donor": [_synthetic_row(.9, 1.0) for _ in range(4)],
                "payload_donor": [_synthetic_row(.1, .2) for _ in range(4)],
                "positive_control": [_synthetic_row(-.8, -2.0) for _ in range(4)],
            }
        background_cmp[name] = [
            {cell: {"cosine": .95, "present_over_absent_rms": 1.0, "tokens": 100}
             for cell in CELLS} for _ in range(4)]
    return analysis, background_cmp


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert FORWARDS_PER_BATCH == 29
        assert EXPECTED_PHASE_FORWARDS == 7250
        assert MY_NAMES == ("L7H3->L8H4", "L8H3->L8H4", "L8H4->L8H3")
        analysis, background_cmp = _synthetic_analysis()
        for name in MY_NAMES:
            for quarters in ((0, 1), (2, 3)):
                report = _direction_report(analysis, background_cmp, name, quarters)
                assert report["holds"] is True
        for path, expected in HASHES.items():
            if not path.is_file() or r501.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed",
            "rung": "equality_score_sign_gauge_validation_and_reverse",
            "model_loaded": False, "outcomes_opened": False,
            "expected_forwards": EXPECTED_PHASE_FORWARDS + r501.SCALE_FORWARDS,
            "synthetic_edge_path_exercised": True,
        }, indent=2, sort_keys=True))
        return
    rows, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("sign-gauge validation output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scales, scale_diagnostics, scale_metadata = r501.collect_scales(model)
    collection = collect(model, rows, scales)
    analysis, background_cmp = analyze(collection)
    pred_a, pred_b, pred_c, pred_d, pred_e, reports = score(
        analysis, background_cmp, collection, scale_diagnostics, checkpoint)
    strong_null = bool(not pred_a or not pred_b)
    torch.save({
        "schema": "equality_score_sign_gauge_validation_stats_v1",
        "task_ce_sums": collection["task_sums"],
        "task_counts": collection["task_counts"],
        "reader_dot_products": collection["reader_stats"],
        "background_dot_products": collection["background_stats"],
        "raw_tokens_logits_or_mlp9_vectors_included": False,
    }, BUNDLE)
    result = {
        "status": "complete",
        "rung": "equality_score_sign_gauge_validation_and_reverse",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "z2_sign_gauge_validation_and_mutual_direction_test",
        "source_hashes": {str(path): r501.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "scale_input_identity": scale_metadata,
        "pairs": list(MY_NAMES), "hybrids": list(HYBRIDS),
        "quarters": [list(q) for q in QUARTERS],
        "analysis": analysis, "background_comparisons": background_cmp,
        "direction_reports": {
            name: {phase: {k: v for k, v in block.items()
                           if k in ("quarters", "edge_all", "control_all", "holds")}
                   for phase, block in phases.items()}
            for name, phases in reports.items()},
        "detail_checks": reports,
        "instrument": {**collection["diagnostics"],
                       "scale_diagnostics": scale_diagnostics},
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": r501.sha256(BUNDLE)},
        'pred_a_exact_live_validation_instrument': pred_a,
        'pred_b_forward_gauge_validates_500_1000': pred_b,
        'pred_c_reverse_L8H4_to_L8H3_edge_on_discovery': pred_c,
        'pred_d_reverse_validates_500_1000': pred_d,
        'pred_e_mutual_sign_gauge_licensed': pred_e,
        "validation_documents_opened_for_negated_outcomes": True,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": EXPECTED_PHASE_FORWARDS + r501.SCALE_FORWARDS,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "instrument_breach_repair_only" if not pred_a else
            ("sixth_standing_claim_language_licensed" if pred_e else
             ("forward_gauge_validated_reverse_reported_and_closed" if pred_b else
              "gauge_failed_validation_retain_2630_as_discovery_only"))),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": result["rung"],
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "direction_reports": result["direction_reports"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
