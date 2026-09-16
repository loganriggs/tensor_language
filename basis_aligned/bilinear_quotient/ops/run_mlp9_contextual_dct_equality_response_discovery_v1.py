#!/usr/bin/env python3
# BQGATE:64opened equality documents;16batches*4forwards;180seconds;natural-response bridge.
"""Test frozen DCT coverage of the established equality/copy MLP9 response."""
import json
import math
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
BQ = ROOT / "basis_aligned/bilinear_quotient"
HERE = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
WEIGHTS = P / "extracted_circuits/mlp9_contextual_dct_node_v1/weights_rank16_v2.pt"
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import equality_matcher_mlp9_reader_calibration_rung500 as r500
from sparse_path_stability_atlas_v1 import digest

STEM = "MLP9_CONTEXTUAL_DCT_EQUALITY_RESPONSE_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
R500_RESULT = BQ / "equality_matcher_mlp9_reader_calibration_rung500_results.json"
START, STOP = 500, 564
CELLS = ("copy_positive", "noncopy_equality", "all_noncopy")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    authority = json.loads(R500_RESULT.read_text())
    authority_keys = tuple("pred_" + suffix for suffix in (
        "a_exact_live_reader_instrument", "b_mlp9_reads_known_score_relation",
        "c_mlp9_rejects_typed_controls", "d_reader_stable_under_early_removal",
        "e_reader_copy_task_selective", "f_named_reader_calibrated",
    ))
    if not all(authority.get(key) is True for key in authority_keys) or authority.get("strong_null"):
        raise ValueError("rung-500 authority changed")
    rows, _, _, _, scales, _ = action_parent.validate_inputs()
    if len(rows) != 1000:
        raise ValueError("equality rows changed")
    return rows, scales, torch.load(WEIGHTS, map_location="cpu", weights_only=True)


def plan():
    rows, _, program = load_bound()
    return {"schema": "mlp9_contextual_dct_equality_response_discovery_v1_plan", "opened_documents": STOP - START, "tokens_per_document": int(rows.shape[1] - 1), "rank": int(program["selected_rank"]), "cells": list(CELLS), "random_rank16_controls": 16, "forward_calls": math.ceil((STOP - START) / action_parent.BATCH) * 4, "new_text": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def response_report(response, selected, basis, controls):
    values = response[selected].float()
    if not len(values):
        raise ValueError("empty response cell")
    projected = values @ basis @ basis.T
    coverage = float(projected.norm() / values.norm().clamp_min(1e-30))
    random = [float((values @ control @ control.T).norm() / values.norm().clamp_min(1e-30)) for control in controls]
    return {"tokens": int(selected.sum()), "coverage": coverage, "random_rank16_coverage": random, "random_median": float(torch.tensor(random).median()), "random_maximum": max(random), "response_norm": float(values.norm()), "projected_norm": float(projected.norm())}, projected


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(OUT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, scales, program = load_bound()
    basis = program["basis"].cuda().float()
    generator = torch.Generator().manual_seed(2026091646)
    controls = [torch.linalg.qr(torch.randn(1152, 16, generator=generator, dtype=torch.float64), mode="reduced").Q.cuda().float() for _ in range(16)]
    masks = r500._task_masks(rows)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    response_parts = {kind: {cell: [] for cell in CELLS} for kind in ("native", "restored")}
    projected_pairs = {cell: {"native": [], "restored": []} for cell in CELLS}
    nll = {name: [] for name in ("native", "absent", "restored")}
    replay_logit_max = 0.0
    replay_write_max = 0.0
    support = {cell: 0 for cell in CELLS}
    started = time.perf_counter()

    for start in range(START, STOP, action_parent.BATCH):
        stop = min(start + action_parent.BATCH, STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].cuda()
        direct_logits, direct_write, _, _ = r500._captured_forward(model, tokens, direct=True)
        native_logits, native_write, _, _ = r500._captured_forward(model, tokens, pair=None)
        absent_logits, absent_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="late_absent", scales=scales["L5H5"])
        restored_logits, restored_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="score_donor", scales=scales["L5H5"])
        replay_logit_max = max(replay_logit_max, float((direct_logits.float() - native_logits.float()).abs().max()))
        replay_write_max = max(replay_write_max, float((direct_write.float() - native_write.float()).abs().max()))
        targets = batch_rows[:, 1:].cuda()
        for name, logits in (("native", native_logits), ("absent", absent_logits), ("restored", restored_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
        native_response = absent_write.float() - native_write.float()
        restored_response = absent_write.float() - restored_write.float()
        for cell in CELLS:
            selected = masks[cell][start:stop].cuda()
            support[cell] += int(selected.sum())
            response_parts["native"][cell].append(native_response[selected].cpu())
            response_parts["restored"][cell].append(restored_response[selected].cpu())

    reports = {kind: {} for kind in ("native", "restored")}
    projected = {kind: {} for kind in ("native", "restored")}
    for kind in ("native", "restored"):
        for cell in CELLS:
            values = torch.cat(response_parts[kind][cell]).cuda()
            selected = torch.ones(len(values), dtype=torch.bool, device="cuda")
            reports[kind][cell], projected[kind][cell] = response_report(values, selected, basis, controls)
    native_projected = projected["native"]["copy_positive"].reshape(-1).double()
    restored_projected = projected["restored"]["copy_positive"].reshape(-1).double()
    projected_cosine = float((native_projected * restored_projected).sum() / (native_projected.norm() * restored_projected.norm()).clamp_min(1e-30))
    projected_scale = float((native_projected * restored_projected).sum() / restored_projected.square().sum().clamp_min(1e-30))
    nll = {name: torch.cat(values) for name, values in nll.items()}
    selected = masks["copy_positive"][START:STOP]
    native_task = nll["absent"] - nll["native"]
    restored_task = nll["absent"] - nll["restored"]
    native_sum = float(native_task[selected].sum())
    restored_sum = float(restored_task[selected].sum())
    task_recovery = restored_sum / native_sum if abs(native_sum) > 1e-30 else None
    pred_a = bool(replay_logit_max == 0.0 and replay_write_max == 0.0 and min(support.values()) > 0 and native_sum > 0)
    native_copy = reports["native"]["copy_positive"]
    restored_copy = reports["restored"]["copy_positive"]
    pred_b = bool(pred_a and native_copy["coverage"] >= .50 and native_copy["coverage"] >= native_copy["random_maximum"] + .20)
    pred_c = bool(pred_a and restored_copy["coverage"] >= .50 and projected_cosine >= .75)
    pred_d = bool(pred_a and all(reports[kind]["copy_positive"]["coverage"] >= reports[kind][control]["coverage"] + .15 for kind in ("native", "restored") for control in ("noncopy_equality", "all_noncopy")))
    pred_e = bool(pred_a and .70 <= task_recovery <= 1.30 and projected_scale > 0)
    predictions = {"pred_a_authority_and_instrument": pred_a, "pred_b_native_response_coverage": pred_b, "pred_c_restored_response_coverage": pred_c, "pred_d_semantic_selectivity": pred_d, "pred_e_task_bridge": pred_e}
    terminal = "mlp9_contextual_dct_equality_response_candidate" if all(predictions.values()) else "valid_mlp9_contextual_dct_equality_response_null" if pred_a else "invalid"
    result = {"schema": "mlp9_contextual_dct_equality_response_discovery_v1_result", "terminal": terminal, "predictions": predictions, "reports": reports, "projected_native_restored_cosine": projected_cosine, "projected_restored_to_native_scale": projected_scale, "task": {"copy_tokens": int(selected.sum()), "native_effect_sum_nat": native_sum, "restored_effect_sum_nat": restored_sum, "score_restoration_recovery": task_recovery}, "support": support, "native_replay_logit_max_absolute": replay_logit_max, "native_replay_mlp9_write_max_absolute": replay_write_max, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Opened-panel bridge from the frozen generic contextual-DCT output basis to the previously established semantic equality/copy MLP9 response; no new rows, causal projection, extraction of the equality source, or fresh circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "native": reports["native"], "restored": reports["restored"], "projected_cosine": projected_cosine, "task_recovery": task_recovery, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
