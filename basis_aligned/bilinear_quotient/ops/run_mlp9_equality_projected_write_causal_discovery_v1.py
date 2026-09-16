#!/usr/bin/env python3
# BQGATE:64opened equality documents;176forwards;180seconds;projected MLP9 write causal test.
"""Test causal sufficiency of projected equality-response writes at MLP9."""
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
DCT_WEIGHTS = P / "extracted_circuits/mlp9_contextual_dct_node_v1/weights_rank16_v2.pt"
EQ_ARTIFACT = P / "MLP9_EQUALITY_CONSUMER_RESPONSE_BASIS_DISCOVERY_V1_ARTIFACT.pt"
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import equality_matcher_mlp9_reader_calibration_rung500 as r500
import rung498_copy_task_portability_diagnosis as diagnosis
from sparse_path_stability_atlas_v1 import digest

STEM = "MLP9_EQUALITY_PROJECTED_WRITE_CAUSAL_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
BASIS_RESULT = P / "MLP9_EQUALITY_CONSUMER_RESPONSE_BASIS_DISCOVERY_V1_RESULT.json"
START, STOP = 564, 628
ARMS = ("exact", "dct16", "eq16", "eq64", "random0", "random1", "random2", "random3")
CELLS = ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors", "all_noncopy")


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
    result = json.loads(BASIS_RESULT.read_text())
    if result["terminal"] != "valid_mlp9_equality_consumer_response_basis_null" or result["selected_rank"] != 64:
        raise ValueError("basis receipt changed")
    rows, _, _, _, scales, _ = action_parent.validate_inputs()
    return rows, scales, torch.load(DCT_WEIGHTS, map_location="cpu", weights_only=True), torch.load(EQ_ARTIFACT, map_location="cpu", weights_only=True)


def plan():
    rows, _, _, _ = load_bound()
    batches = math.ceil((STOP - START) / action_parent.BATCH)
    return {"schema": "mlp9_equality_projected_write_causal_discovery_v1_plan", "opened_documents": STOP - START, "tokens_per_document": int(rows.shape[1] - 1), "arms": list(ARMS), "forward_calls": batches * (3 + len(ARMS)), "new_text": 0, "fits": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


@torch.no_grad()
def installed_absent_forward(model, tokens, scales, projected):
    captured = []
    def hook(_module, _inputs, output):
        installed = output - projected.to(output.dtype)
        captured.append(installed.detach().float())
        return installed
    handle = model.transformer.h[9].mlp.register_forward_hook(hook)
    try:
        logits, _, _ = action_parent.run_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="late_absent", scales=scales["L5H5"])
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("MLP9 installation hook count changed")
    return logits, captured[0]


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
    rows, scales, dct_program, eq_artifact = load_bound()
    dct = dct_program["basis"].cuda().float()
    eq64 = eq_artifact["basis_rank64"].cuda().float()
    eq16 = eq64[:, :16]
    generator = torch.Generator().manual_seed(2026091647)
    randoms = [torch.linalg.qr(torch.randn(1152, 16, generator=generator, dtype=torch.float64), mode="reduced").Q.cuda().float() for _ in range(4)]
    projections = {"dct16": dct, "eq16": eq16, "eq64": eq64, **{f"random{i}": basis for i, basis in enumerate(randoms)}}
    masks = diagnosis.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"])
    valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    if any(name not in masks or not bool(masks[name][START:STOP].any()) for name in CELLS):
        raise ValueError("required copy-task mask support changed")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    nll = {name: [] for name in ("native", "absent", *ARMS)}
    replay_logits, replay_writes, install_errors = [], [], {arm: [] for arm in ARMS}
    started = time.perf_counter()

    for start in range(START, STOP, action_parent.BATCH):
        stop = min(start + action_parent.BATCH, STOP)
        batch_rows = rows[start:stop]
        tokens = batch_rows[:, :-1].cuda()
        direct_logits, direct_write, _, _ = r500._captured_forward(model, tokens, direct=True)
        native_logits, native_write, _, _ = r500._captured_forward(model, tokens, pair=None)
        absent_logits, absent_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="late_absent", scales=scales["L5H5"])
        replay_logits.append(float((direct_logits.float() - native_logits.float()).abs().max()))
        replay_writes.append(float((direct_write.float() - native_write.float()).abs().max()))
        response = absent_write.float() - native_write.float()
        arm_projected = {"exact": response, **{name: response @ basis @ basis.T for name, basis in projections.items()}}
        targets = batch_rows[:, 1:].cuda()
        for name, logits in (("native", native_logits), ("absent", absent_logits)):
            nll[name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
        for arm in ARMS:
            logits, installed = installed_absent_forward(model, tokens, scales, arm_projected[arm])
            expected = (absent_write.to(torch.bfloat16) - arm_projected[arm].to(torch.bfloat16)).float()
            install_errors[arm].append(float((installed - expected).norm() / expected.norm().clamp_min(1e-30)))
            nll[arm].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())

    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    native_effect = nll["absent"] - nll["native"]
    reports = {arm: {} for arm in ARMS}
    for arm in ARMS:
        arm_effect = nll["absent"] - nll[arm]
        for cell in CELLS:
            selected = masks[cell][START:STOP]
            native_sum = float(native_effect[selected].sum())
            arm_sum = float(arm_effect[selected].sum())
            reports[arm][cell] = {"tokens": int(selected.sum()), "native_effect_sum_nat": native_sum, "arm_effect_sum_nat": arm_sum, "recovery": arm_sum / native_sum if abs(native_sum) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["native"])[selected].mean()), "arm_minus_native_rms_nat": float((nll[arm] - nll["native"])[selected].square().mean().sqrt())}
        half_reports = []
        for lo, hi in ((0, 32), (32, 64)):
            selected = masks["copy_positive"][START + lo:START + hi]
            ns = float(native_effect[lo:hi][selected].sum())
            ars = float(arm_effect[lo:hi][selected].sum())
            half_reports.append({"documents": 32, "tokens": int(selected.sum()), "recovery": ars / ns if abs(ns) > 1e-30 else None})
        reports[arm]["halves"] = half_reports
    exact_recovery = reports["exact"]["copy_positive"]["recovery"]
    pred_a = bool(max(replay_logits) == 0.0 and max(replay_writes) == 0.0 and max(max(values) for values in install_errors.values()) <= 2e-6)
    pred_b = bool(pred_a and exact_recovery >= .30 and abs(reports["exact"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01)
    pred_c = bool(pred_a and reports["eq16"]["copy_positive"]["recovery"] >= .80 * exact_recovery and abs(reports["eq16"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01)
    pred_d = bool(pred_a and reports["eq64"]["copy_positive"]["recovery"] >= .90 * exact_recovery and reports["eq64"]["copy_positive"]["recovery"] >= .30)
    pred_e = bool(pred_a and reports["eq16"]["copy_positive"]["recovery"] >= reports["dct16"]["copy_positive"]["recovery"] + .15 and reports["eq16"]["copy_positive"]["recovery"] >= max(reports[f"random{i}"]["copy_positive"]["recovery"] for i in range(4)) + .10)
    stability_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_f = bool(pred_a and all(reports[arm][cell]["recovery"] > 0 for arm in ("exact", "eq16", "eq64") for cell in stability_cells) and all(value["recovery"] > 0 for arm in ("exact", "eq16", "eq64") for value in reports[arm]["halves"]))
    predictions = {"pred_a_instrument": pred_a, "pred_b_exact_write_sufficiency": pred_b, "pred_c_rank16_consumer_sufficiency": pred_c, "pred_d_rank64_consumer_sufficiency": pred_d, "pred_e_behavior_conditioned_advantage": pred_e, "pred_f_sign_and_cell_stability": pred_f}
    terminal = "mlp9_equality_projected_write_causal_candidate" if all(predictions.values()) else "valid_mlp9_equality_projected_write_causal_null" if pred_a else "invalid"
    result = {"schema": "mlp9_equality_projected_write_causal_discovery_v1_result", "terminal": terminal, "predictions": predictions, "reports": reports, "maximum_native_replay_logit_absolute": max(replay_logits), "maximum_native_replay_write_absolute": max(replay_writes), "maximum_install_relative_l2_by_arm": {arm: max(values) for arm, values in install_errors.items()}, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Opened-panel causal sufficiency test of exact and projected MLP9 write restoration under the established equality-score removal; no new rows, fresh OOD, upstream-source extraction, or complete-circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "copy_recovery": {arm: reports[arm]["copy_positive"]["recovery"] for arm in ARMS}, "noncopy_mean_damage": {arm: reports[arm]["all_noncopy"]["arm_minus_native_mean_nat"] for arm in ARMS}, "halves": {arm: reports[arm]["halves"] for arm in ("exact", "eq16", "eq64")}, "install_errors": result["maximum_install_relative_l2_by_arm"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
