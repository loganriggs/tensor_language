#!/usr/bin/env python3
"""RUNG500 -- prospective MLP9 reader calibration for the shared copy score."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
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
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import rung498_copy_task_portability_diagnosis as mask_parent


PREREG = POLY / "EQUALITY_MATCHER_MLP9_READER_CALIBRATION_RUNG500_PREREGISTRATION.md"
R499_SOURCE = ROOT / "ops/equality_matcher_copy_task_calibration_rung499.py"
R499_RESULT = ROOT / "equality_matcher_copy_task_calibration_rung499_results.json"
R499_BUNDLE = ROOT / "equality_matcher_copy_task_calibration_rung499_bundle.pt"
ACTION_SOURCE = ROOT / "ops/equality_matcher_causal_action_quotient_rung498.py"
OUT = ROOT / "equality_matcher_mlp9_reader_calibration_rung500_results.json"
BUNDLE = ROOT / "equality_matcher_mlp9_reader_calibration_rung500_bundle.pt"
HASHES = {
    PREREG: "c2fd99ff688856e4c92e9b9e9091d1bc264d7470d3b3750f6fd1b10a0ef30236",
    R499_SOURCE: "2c85a758f3083df467761d199f8dd108602ed7528789910f655030497d088904",
    R499_RESULT: "2c5b7faabe613a6774e4f7b8f9648f70d4e2a83b31ff08db7f11af04e8eaae81",
    R499_BUNDLE: "b2cddade765190cea0bc4e0969bb9721785ce5468c0a3ad4c1ac4447f7f1f59b",
    ACTION_SOURCE: "3186d610b77e1684849a54af79e83ce3d7a6a4338e36b3ec27ce2d7cc8696e59",
}
BOUNDS = (500, 1000, 750)
CELLS = ("copy_positive", "noncopy_equality", "all_noncopy")
HYBRIDS = ("score_donor", "payload_donor", "whole_donor")
D = 1152


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _empty_stats():
    shape = (len(action_parent.DONORS), len(action_parent.BACKGROUNDS), len(HYBRIDS),
             2, len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("ref2", "hyb2", "cross", "write2", "tokens")}


def _empty_background_stats():
    shape = (2, len(CELLS))
    return {key: torch.zeros(shape, dtype=torch.float64)
            for key in ("present2", "absent2", "cross", "tokens")}


def _report(stats, index):
    ref2 = float(stats["ref2"][index])
    hyb2 = float(stats["hyb2"][index])
    cross = float(stats["cross"][index])
    write2 = float(stats["write2"][index])
    cosine = cross / math.sqrt(max(ref2 * hyb2, 1e-30))
    scale = cross / max(hyb2, 1e-30)
    fitted_scale = max(scale, 0.0)
    residual = math.sqrt(max(ref2 - 2 * fitted_scale * cross
                             + fitted_scale * fitted_scale * hyb2, 0.0)
                         / max(ref2, 1e-30))
    return {"cosine": cosine, "positive_fit_scale": scale,
            "scaled_residual": residual,
            "native_response_rms_over_native_write_rms": math.sqrt(ref2 / max(write2, 1e-30)),
            "hybrid_response_rms_over_native_write_rms": math.sqrt(hyb2 / max(write2, 1e-30)),
            "tokens": int(stats["tokens"][index])}


def _background_report(stats, index):
    left2 = float(stats["present2"][index])
    right2 = float(stats["absent2"][index])
    cross = float(stats["cross"][index])
    return {"cosine": cross / math.sqrt(max(left2 * right2, 1e-30)),
            "present_over_absent_rms": math.sqrt(left2 / max(right2, 1e-30)),
            "tokens": int(stats["tokens"][index])}


@torch.no_grad()
def _captured_forward(model, tokens, **kwargs):
    captures = []

    def hook(_module, _inputs, output):
        captures.append(output.detach().clone())

    handle = model.transformer.h[9].mlp.register_forward_hook(hook)
    try:
        logits, diagnostics, audit = action_parent.run_forward(model, tokens, **kwargs)
    finally:
        handle.remove()
    if len(captures) != 1 or list(captures[0].shape) != [len(tokens), 256, D]:
        raise RuntimeError(f"MLP9 capture count/shape changed: {[x.shape for x in captures]}")
    return logits, captures[0], diagnostics, audit


def _task_masks(rows):
    masks = mask_parent.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"])
    valid[:, 64:] = True
    return {"copy_positive": masks["copy_positive"],
            "noncopy_equality": masks["noncopy_equality"],
            "all_noncopy": valid & ~masks["copy_positive"]}


@torch.no_grad()
def collect(model, rows, scales):
    masks = _task_masks(rows)
    stats = _empty_stats()
    background_stats = _empty_background_stats()
    diagnostics = {"native_replay_logit_max_abs": 0.0, "native_replay_mlp9_max_abs": 0.0,
                   "factor_reconstruction_max": 0.0, "minimum_nonzero_edit_rms": math.inf,
                   "zero_intended_edit_actions": 0, "capture_calls": 0}
    calls = {"native": 0, "analytical": 0}
    device = next(model.parameters()).device
    for start in range(BOUNDS[0], BOUNDS[1], action_parent.BATCH):
        stop = min(start + action_parent.BATCH, BOUNDS[1])
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].to(device)
        native_logits, native_write, _diag, _audit = _captured_forward(
            model, tokens, direct=True)
        replay_logits, replay_write, diag, _audit = _captured_forward(model, tokens, pair=None)
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
        writes = {}
        for donor_index, (donor, pair) in enumerate(zip(action_parent.DONORS, action_parent.PAIRS)):
            writes[donor_index] = {}
            for background_index, background in enumerate(action_parent.BACKGROUNDS):
                writes[donor_index][background_index] = {"late_native": replay_write}
                for state in action_parent.STATES:
                    if background == "early_present" and state == "late_native":
                        continue
                    logits, write, diag, _audit = _captured_forward(
                        model, tokens, pair=pair, background=background, state=state,
                        scales=scales[donor])
                    calls["analytical"] += 1
                    diagnostics["capture_calls"] += 1
                    diagnostics["factor_reconstruction_max"] = max(
                        diagnostics["factor_reconstruction_max"], diag["factor_reconstruction_max"])
                    if background == "early_absent" and diag["early_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    if state != "late_native" and diag["late_edit_rms"] <= 0:
                        diagnostics["zero_intended_edit_actions"] += 1
                    for key in ("early_edit_rms", "late_edit_rms"):
                        if diag[key] > 0:
                            diagnostics["minimum_nonzero_edit_rms"] = min(
                                diagnostics["minimum_nonzero_edit_rms"], diag[key])
                    writes[donor_index][background_index][state] = write
                    del logits
        local_masks = {name: mask[start:stop].to(device) for name, mask in masks.items()}
        global_rows = torch.arange(start, stop, device=device)
        for donor_index in range(len(action_parent.DONORS)):
            for background_index in range(len(action_parent.BACKGROUNDS)):
                absent = writes[donor_index][background_index]["late_absent"].float()
                native = writes[donor_index][background_index]["late_native"].float()
                reference = absent - native
                for hybrid_index, state in enumerate(HYBRIDS):
                    hybrid = absent - writes[donor_index][background_index][state].float()
                    for half, condition in enumerate((global_rows < BOUNDS[2], global_rows >= BOUNDS[2])):
                        for cell_index, cell in enumerate(CELLS):
                            selected = local_masks[cell] & condition[:, None]
                            if not bool(selected.any()):
                                continue
                            ref = reference[selected]
                            hyb = hybrid[selected]
                            writer = native[selected]
                            index = (donor_index, background_index, hybrid_index, half, cell_index)
                            stats["ref2"][index] += ref.double().square().sum().cpu()
                            stats["hyb2"][index] += hyb.double().square().sum().cpu()
                            stats["cross"][index] += (ref.double() * hyb.double()).sum().cpu()
                            stats["write2"][index] += writer.double().square().sum().cpu()
                            stats["tokens"][index] += int(selected.sum())
        present = (writes[0][0]["late_absent"].float()
                   - writes[0][0]["score_donor"].float())
        absent = (writes[0][1]["late_absent"].float()
                  - writes[0][1]["score_donor"].float())
        for half, condition in enumerate((global_rows < BOUNDS[2], global_rows >= BOUNDS[2])):
            for cell_index, cell in enumerate(CELLS):
                selected = local_masks[cell] & condition[:, None]
                if not bool(selected.any()):
                    continue
                left, right = present[selected].double(), absent[selected].double()
                index = (half, cell_index)
                background_stats["present2"][index] += left.square().sum().cpu()
                background_stats["absent2"][index] += right.square().sum().cpu()
                background_stats["cross"][index] += (left * right).sum().cpu()
                background_stats["tokens"][index] += int(selected.sum())
        del native_logits, replay_logits, writes
    diagnostics["calls"] = calls
    diagnostics["expected_calls"] = {"native": 125, "analytical": 2375}
    diagnostics["calls_exact"] = calls == diagnostics["expected_calls"]
    diagnostics["capture_calls_exact"] = diagnostics["capture_calls"] == 2500
    return stats, background_stats, diagnostics


def analyze(stats, background_stats):
    reports = {}
    for donor_index, donor in enumerate(action_parent.DONORS):
        reports[donor] = {}
        for background_index, background in enumerate(action_parent.BACKGROUNDS):
            reports[donor][background] = {}
            for hybrid_index, hybrid in enumerate(HYBRIDS):
                reports[donor][background][hybrid] = [
                    {cell: _report(stats, (donor_index, background_index, hybrid_index,
                                           half, cell_index))
                     for cell_index, cell in enumerate(CELLS)}
                    for half in range(2)]
    background = [{cell: _background_report(background_stats, (half, cell_index))
                   for cell_index, cell in enumerate(CELLS)} for half in range(2)]
    return reports, background


def score(reports, background, diagnostics):
    positive_cells = [reports["L5H5"][g]["score_donor"][h]["copy_positive"]
                      for g in action_parent.BACKGROUNDS for h in range(2)]
    pred_a = bool(
        diagnostics["native_replay_logit_max_abs"] == 0.0
        and diagnostics["native_replay_mlp9_max_abs"] == 0.0
        and diagnostics["factor_reconstruction_max"] <= 1e-10
        and diagnostics["minimum_nonzero_edit_rms"] > 0
        and diagnostics["zero_intended_edit_actions"] == 0
        and diagnostics["calls_exact"] and diagnostics["capture_calls_exact"]
        and all(row["tokens"] > 0
                for donor in reports.values() for g in donor.values()
                for hybrid in g.values() for half in hybrid for row in half.values())
        and all(min(row["native_response_rms_over_native_write_rms"],
                    row["hybrid_response_rms_over_native_write_rms"]) >= 1e-4
                for row in positive_cells))
    pred_b = all(row["cosine"] >= .75 and row["scaled_residual"] <= .70
                 and row["positive_fit_scale"] > 0 for row in positive_cells)
    control_checks = []
    for background_name in action_parent.BACKGROUNDS:
        for half in range(2):
            positive = reports["L5H5"][background_name]["score_donor"][half]["copy_positive"]
            controls = (
                ("L7H3_score", reports["L7H3"][background_name]["score_donor"][half]["copy_positive"]),
                ("L5H5_payload", reports["L5H5"][background_name]["payload_donor"][half]["copy_positive"]),
            )
            for name, control in controls:
                holds = bool(positive["cosine"] >= control["cosine"] + .30
                             or positive["scaled_residual"]
                             <= control["scaled_residual"] - .30)
                control_checks.append({"background": background_name, "quarter": half,
                                       "control": name, "holds": holds})
    pred_c = all(row["holds"] for row in control_checks)
    closure = []
    for half in range(2):
        present = reports["L5H5"]["early_present"]["score_donor"][half]["copy_positive"]
        absent = reports["L5H5"]["early_absent"]["score_donor"][half]["copy_positive"]
        scales = [present["positive_fit_scale"], absent["positive_fit_scale"]]
        drift = abs(scales[0] - scales[1]) / max(abs(scales[0]), abs(scales[1]), 1e-30)
        holds = bool(background[half]["copy_positive"]["cosine"] >= .75
                     and min(scales) > 0 and drift <= .50)
        closure.append({"quarter": half, "response_cosine": background[half]["copy_positive"]["cosine"],
                        "scale_drift": drift, "holds": holds})
    pred_d = bool(pred_b and all(row["holds"] for row in closure))
    specificity = []
    for background_name in action_parent.BACKGROUNDS:
        for half in range(2):
            cells = reports["L5H5"][background_name]["score_donor"][half]
            margin = cells["copy_positive"]["cosine"] - max(
                cells["noncopy_equality"]["cosine"], cells["all_noncopy"]["cosine"])
            specificity.append({"background": background_name, "quarter": half,
                                "cosine_margin": margin, "holds": margin >= .10})
    pred_e = all(row["holds"] for row in specificity)
    return pred_a, pred_b, pred_c, pred_d, pred_e, {
        "control_checks": control_checks, "background_closure": closure,
        "copy_specificity": specificity,
    }


def main():
    started = time.time()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        stats = _empty_stats(); index = (0, 0, 0, 0, 0)
        stats["ref2"][index] = stats["hyb2"][index] = stats["cross"][index] = 4
        stats["write2"][index] = 100; stats["tokens"][index] = 2
        assert _report(stats, index)["cosine"] == 1.0
        print(json.dumps({"status": "dry_run_passed", "rung": 500, "model_loaded": False,
                          "rung499_outcomes_loaded": False, "forwards": 2500,
                          "predictions": ["pred_a", "pred_b", "pred_c", "pred_d", "pred_e", "pred_f"]},
                         indent=2, sort_keys=True))
        return
    if OUT.exists() or BUNDLE.exists():
        raise RuntimeError("rung500 output namespace already exists")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    rows, _circuit_masks, _discovery_tags, _validation_tags, scales, metadata = \
        action_parent.validate_inputs()
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    stats, background_stats, diagnostics = collect(model, rows, scales)
    reports, background = analyze(stats, background_stats)
    pred_a, pred_b, pred_c, pred_d, pred_e, checks = score(
        reports, background, diagnostics)
    pred_a = bool(pred_a and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256)
    pred_f = bool(pred_a and pred_b and pred_c and pred_d and pred_e)
    bundle = {"schema": "equality_matcher_mlp9_reader_calibration_rung500_stats_v1",
              "stats": stats, "background_stats": background_stats,
              "raw_tokens_logits_or_mlp9_vectors_included": False,
              "rung499_outcomes_included": False}
    torch.save(bundle, BUNDLE)
    result = {
        "status": "complete", "rung": 500,
        "claim_level": "prospective_named_reader_calibration_not_circuit_discovery_or_compression",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_identity": metadata, "documents": [500, 1000],
        "quarters": [[500, 750], [750, 1000]], "reader": "MLP9 write",
        "cells": list(CELLS), "donors": list(action_parent.DONORS),
        "backgrounds": list(action_parent.BACKGROUNDS), "hybrids": list(HYBRIDS),
        "diagnostics": diagnostics, "reports": reports,
        "background_comparisons": background, "checks": checks,
        'pred_a_exact_live_reader_instrument': pred_a,
        'pred_b_mlp9_reads_known_score_relation': pred_b,
        'pred_c_mlp9_rejects_typed_controls': pred_c,
        'pred_d_reader_stable_under_early_removal': pred_d,
        'pred_e_reader_copy_task_selective': pred_e,
        'pred_f_named_reader_calibrated': pred_f,
        "strong_null": bool(not pred_a or not pred_b or not pred_c or not pred_d or not pred_e),
        "rung499_outcomes_deserialized": False,
        "sufficient_statistics": {"path": str(BUNDLE), "sha256": sha256(BUNDLE),
                                  "bytes": BUNDLE.stat().st_size},
        "execution_price": {"forwards": sum(diagnostics["calls"].values()),
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_added": 0, "deployed_parameters_saved": 0},
        "runtime_s": time.time() - started,
        "next_step": (
            "register_four_score_directed_search_with_mlp9_and_task_observations" if pred_f else
            "repair_instrument_only" if not pred_a else
            "retire_mlp9_reader_on_current_actions" if not pred_b else
            "mlp9_does_not_discriminate_controls" if not pred_c else
            "record_background_conditional_reader" if not pred_d else
            "reader_is_generic_not_copy_selective"),
    }
    dump(result, OUT)
    print(json.dumps({"status": "complete", "rung": 500,
                      "predictions": {k: v for k, v in result.items() if k.startswith("pred_")},
                      "strong_null": result["strong_null"], "runtime_s": result["runtime_s"],
                      "execution_price": result["execution_price"], "next_step": result["next_step"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
