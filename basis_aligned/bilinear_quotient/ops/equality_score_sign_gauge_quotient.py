#!/usr/bin/env python3
"""Z2 sign-gauge quotient of the equality-score families.

# BQGATE: EXPERIMENT
# pred_a_exact_live_sign_gauge_instrument
# pred_b_L7H3_score_is_L8H4_computation_up_to_sign
# pred_c_gauge_extends_to_L8H3

Parallel-lane probe (Claude), math-review 1907 move #3. Rung 501 split the four
equality scores into two sign-coherent families; this asks whether the
anti-alignment is a pure Z2 gauge by transplanting cross-family scores with the
NEGATED in-run frozen scale under rung 501's verbatim edge criteria, with the
positive-scale arm re-measured as the in-run anti-alignment control and a
negated-payload specificity control. Imports the frozen rung 501 module
hash-pinned; modifies no registered file. Preregistration:
polynomial_causal/EQUALITY_SCORE_SIGN_GAUGE_QUOTIENT_PREREGISTRATION.md
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

PREREG = POLY / "EQUALITY_SCORE_SIGN_GAUGE_QUOTIENT_PREREGISTRATION.md"
R501_SOURCE = ROOT / "ops/equality_score_directed_action_graph_rung501.py"
R501_RESULT = ROOT / "equality_score_directed_action_graph_rung501_results.json"
OUT = ROOT / "equality_score_sign_gauge_quotient_results.json"
BUNDLE = ROOT / "equality_score_sign_gauge_quotient_bundle.pt"
HASHES = {
    PREREG: "5ada45c2a025c9c7ba5b511400ec104cb04d694015b41a96ee39ae95ad7971f4",
    R501_SOURCE: "97f3946f558f3d61fc952a9b6ddc7c334b51ccc0ccfe5f02c6ecced417f1e077",
    R501_RESULT: "b17a9b274e4c61e0b4a3fc68d8ce84ec6f8e76f257c3d898e6a6990492301c4f",
}

MY_PAIRS = ((1, 3), (2, 3))
MY_NAMES = tuple(f"{r501.TERMS[l]}->{r501.TERMS[r]}" for l, r in MY_PAIRS)
HYBRIDS = ("neg_score", "pos_score", "neg_payload")
HYBRID_STATES = {"neg_score": "score_donor", "pos_score": "score_donor",
                 "neg_payload": "payload_donor"}
ANALYSIS_SLOT = {"neg_score": "score_donor", "neg_payload": "payload_donor",
                 "pos_score": "positive_control"}
BACKGROUNDS = r501.BACKGROUNDS
CELLS = r501.CELLS
BATCH = r501.BATCH
DOC_RANGE = (0, 500, 250)
BATCHES = 125
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
    receipt = json.loads(R501_RESULT.read_text())
    required = {
        "pred_a_exact_live_isolated_instrument": True,
        "pred_b_calibration_tripwires_reproduce": True,
        "pred_c_new_confirmed_directed_edge": False,
        "pred_d_graph_semantics_typed": False,
        "strong_null": True,
        "validation_licensed_and_opened": False,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise RuntimeError("rung 501 receipt does not license the sign-gauge probe")
    rows, metadata = r501.validate_inputs()
    return rows, {**metadata, "gauge_pairs": list(MY_NAMES), "hybrids": list(HYBRIDS)}


def _empty_reader_stats():
    shape = (len(MY_PAIRS), len(BACKGROUNDS), len(HYBRIDS), 2, len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("ref2", "hyb2", "cross", "write2", "tokens")}


def _empty_background_stats():
    shape = (len(MY_PAIRS), 2, len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("present2", "absent2", "cross", "tokens")}


@torch.no_grad()
def collect(model, rows, scales):
    docs = DOC_RANGE[1] - DOC_RANGE[0]
    states_order = ("late_native", "late_absent") + HYBRIDS
    task_sums = torch.zeros(len(MY_PAIRS), len(BACKGROUNDS), len(states_order),
                            docs, len(CELLS), dtype=torch.float64)
    task_counts = torch.zeros(docs, len(CELLS), dtype=torch.float64)
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
    for start in range(DOC_RANGE[0], DOC_RANGE[1], BATCH):
        stop = min(start + BATCH, DOC_RANGE[1])
        local = start - DOC_RANGE[0]
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        masks_cpu = {cell: task_masks[cell][start:stop] for cell in CELLS}
        masks_gpu = {cell: value.to(device) for cell, value in masks_cpu.items()}
        global_rows = torch.arange(start, stop, device=device)
        halves = (global_rows < DOC_RANGE[2], global_rows >= DOC_RANGE[2])
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
        task_counts[local:local + len(batch_rows)] = observed_counts
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
                              local:local + len(batch_rows)] = sums[arm]
                absent = writes["late_absent"].float()
                native = writes["late_native"].float()
                reference = absent - native
                for hybrid_index, hybrid in enumerate(HYBRIDS):
                    response = absent - writes[hybrid].float()
                    for half, condition in enumerate(halves):
                        for cell_index, cell in enumerate(CELLS):
                            selected = masks_gpu[cell] & condition[:, None]
                            if not bool(selected.any()):
                                continue
                            ref = reference[selected].double()
                            hyb = response[selected].double()
                            writer = native[selected].double()
                            index = (pair_index, background_index, hybrid_index,
                                     half, cell_index)
                            reader_stats["ref2"][index] += ref.square().sum().cpu()
                            reader_stats["hyb2"][index] += hyb.square().sum().cpu()
                            reader_stats["cross"][index] += (ref * hyb).sum().cpu()
                            reader_stats["write2"][index] += writer.square().sum().cpu()
                            reader_stats["tokens"][index] += int(selected.sum())
                neg_responses[background] = absent - writes["neg_score"].float()
            for half, condition in enumerate(halves):
                for cell_index, cell in enumerate(CELLS):
                    selected = masks_gpu[cell] & condition[:, None]
                    if not bool(selected.any()):
                        continue
                    left = neg_responses["early_present"][selected].double()
                    right = neg_responses["early_absent"][selected].double()
                    index = (pair_index, half, cell_index)
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
        cell: [int(task_masks[cell][DOC_RANGE[0]:DOC_RANGE[2]].sum()),
               int(task_masks[cell][DOC_RANGE[2]:DOC_RANGE[1]].sum())]
        for cell in CELLS}
    return {"task_sums": task_sums, "task_counts": task_counts,
            "reader_stats": reader_stats, "background_stats": background_stats,
            "states_order": states_order, "diagnostics": diagnostics}


def analyze(collection):
    task_sums = collection["task_sums"]
    counts = collection["task_counts"]
    states_order = collection["states_order"]
    docs = counts.shape[0]
    boundary = docs // 2
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
                halves = []
                for half, (lo, hi) in enumerate(((0, boundary), (boundary, docs))):
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
                            (pair_index, background_index, hybrid_index, half, cell_index))
                        for cell_index, cell in enumerate(CELLS)}
                    halves.append({
                        "equality_recovery": recovery, "task_effect": task,
                        "off_target_signed_mean_hybrid_minus_native_nat": off_change,
                        "reader": reader,
                    })
                analysis[name][background][ANALYSIS_SLOT[hybrid]] = halves
        background_cmp[name] = [
            {cell: r501._background_report(collection["background_stats"],
                                           (pair_index, half, cell_index))
             for cell_index, cell in enumerate(CELLS)}
            for half in range(2)]
    return analysis, background_cmp


def _positive_control_check(analysis, name, half):
    rows = []
    for background in BACKGROUNDS:
        row = analysis[name][background]["positive_control"][half]
        rows.append({
            "background": background,
            "task_cosine": row["task_effect"]["cosine"],
            "reader_cosine": row["reader"]["copy_positive"]["cosine"],
            "holds": bool(row["task_effect"]["cosine"] < 0
                          and row["reader"]["copy_positive"]["cosine"] < 0),
        })
    return {"holds": all(row["holds"] for row in rows), "backgrounds": rows}


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
    verdicts = {}
    for name in MY_NAMES:
        edges = [r501._partition_edge(analysis, background_cmp, name, half)
                 for half in range(2)]
        controls = [_positive_control_check(analysis, name, half) for half in range(2)]
        verdicts[name] = {
            "negated_edge_both_halves": bool(all(row["edge"] for row in edges)),
            "positive_control_anti_aligned": bool(all(row["holds"] for row in controls)),
            "partitions": edges, "positive_control": controls,
        }
        verdicts[name]["holds"] = bool(
            verdicts[name]["negated_edge_both_halves"]
            and verdicts[name]["positive_control_anti_aligned"])
    pred_b = verdicts["L7H3->L8H4"]["holds"]
    pred_c = verdicts["L8H3->L8H4"]["holds"]
    return pred_a, pred_b, pred_c, verdicts


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
                "score_donor": [_synthetic_row(.9, 1.0), _synthetic_row(.9, 1.0)],
                "payload_donor": [_synthetic_row(.1, .2), _synthetic_row(.1, .2)],
                "positive_control": [_synthetic_row(-.8, -2.0), _synthetic_row(-.8, -2.0)],
            }
        background_cmp[name] = [
            {cell: {"cosine": .95, "present_over_absent_rms": 1.0, "tokens": 100}
             for cell in CELLS} for _ in range(2)]
    return analysis, background_cmp


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        assert FORWARDS_PER_BATCH == 20
        assert EXPECTED_PHASE_FORWARDS == 2500
        assert MY_NAMES == ("L7H3->L8H4", "L8H3->L8H4")
        analysis, background_cmp = _synthetic_analysis()
        for name in MY_NAMES:
            edges = [r501._partition_edge(analysis, background_cmp, name, half)
                     for half in range(2)]
            assert all(isinstance(row["edge"], bool) for row in edges)
            assert all(row["edge"] for row in edges), "synthetic pass case must be an edge"
            controls = [_positive_control_check(analysis, name, half)
                        for half in range(2)]
            assert all(row["holds"] for row in controls)
        for path, expected in HASHES.items():
            if not path.is_file() or r501.sha256(path) != expected:
                raise RuntimeError(f"frozen hash mismatch: {path}")
        print(json.dumps({
            "status": "dry_run_passed", "rung": "equality_score_sign_gauge_quotient",
            "model_loaded": False, "outcomes_opened": False,
            "validation_or_sealed_opened": False,
            "expected_forwards": EXPECTED_PHASE_FORWARDS + r501.SCALE_FORWARDS,
            "synthetic_edge_path_exercised": True,
        }, indent=2, sort_keys=True))
        return
    rows, metadata = validate_inputs()
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("sign-gauge output namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    scales, scale_diagnostics, scale_metadata = r501.collect_scales(model)
    collection = collect(model, rows, scales)
    analysis, background_cmp = analyze(collection)
    pred_a, pred_b, pred_c, verdicts = score(
        analysis, background_cmp, collection, scale_diagnostics, checkpoint)
    strong_null = bool(not pred_a or (not pred_b and not pred_c))
    torch.save({
        "schema": "equality_score_sign_gauge_quotient_stats_v1",
        "task_ce_sums": collection["task_sums"],
        "task_counts": collection["task_counts"],
        "reader_dot_products": collection["reader_stats"],
        "background_dot_products": collection["background_stats"],
        "raw_tokens_logits_or_mlp9_vectors_included": False,
    }, BUNDLE)
    result = {
        "status": "complete", "rung": "equality_score_sign_gauge_quotient",
        "owner_lane": "claude_parallel_probe",
        "claim_level": "z2_sign_gauge_test_of_score_families_not_equivalence",
        "source_hashes": {str(path): r501.sha256(path) for path in HASHES},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "scale_input_identity": scale_metadata,
        "pairs": list(MY_NAMES), "hybrids": list(HYBRIDS),
        "backgrounds": list(BACKGROUNDS),
        "analysis": analysis, "background_comparisons": background_cmp,
        "verdicts": {name: {k: v for k, v in row.items()
                            if k in ("negated_edge_both_halves",
                                     "positive_control_anti_aligned", "holds")}
                     for name, row in verdicts.items()},
        "detail_checks": verdicts,
        "instrument": {**collection["diagnostics"],
                       "scale_diagnostics": scale_diagnostics},
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": r501.sha256(BUNDLE)},
        'pred_a_exact_live_sign_gauge_instrument': pred_a,
        'pred_b_L7H3_score_is_L8H4_computation_up_to_sign': pred_b,
        'pred_c_gauge_extends_to_L8H3': pred_c,
        "validation_or_sealed_opened": False,
        "strong_null": strong_null,
        "execution_price": {
            "full_model_forwards": EXPECTED_PHASE_FORWARDS + r501.SCALE_FORWARDS,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
            "deployed_parameters_saved": 0, "deployed_parameters_added": 0,
        },
        "next_step": (
            "instrument_breach_repair_only" if not pred_a else
            ("register_validation_and_reverse_directions" if (pred_b or pred_c) else
             "family_split_is_not_a_sign_gauge_close_review_move_3")),
        "runtime_s": time.time() - started,
    }
    dump(result, OUT)
    print(json.dumps({
        "status": result["status"], "rung": result["rung"],
        "predictions": {key: value for key, value in result.items()
                        if key.startswith("pred_")},
        "strong_null": strong_null,
        "verdicts": result["verdicts"],
        "runtime_s": result["runtime_s"], "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
