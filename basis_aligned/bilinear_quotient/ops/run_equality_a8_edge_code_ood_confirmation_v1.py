#!/usr/bin/env python3
# BQGATE:192frozen code-OOD documents;336forwards;180seconds;frozen A8 edge confirmation.
"""Confirm the frozen attention8-only equality support on code OOD."""
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
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import equality_matcher_mlp9_reader_calibration_rung500 as r500
import rung498_copy_task_portability_diagnosis as diagnosis
import run_equality_pre_mlp9_three_edge_factorial_discovery_v1 as discovery
from sparse_path_stability_atlas_v1 import digest

STEM = "EQUALITY_A8_EDGE_CODE_OOD_CONFIRMATION_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
PARENT = P / "EQUALITY_PRE_MLP9_THREE_EDGE_FACTORIAL_DISCOVERY_V1_RESULT.json"
ROWS = BQ / ".rowcache_induction_equality_tensor_final_ood_v2/ood_code.pt"
ROW_RECEIPT = BQ / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
ARMS = ("100", "010", "001")
CELLS = ("copy_positive", "copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors", "all_noncopy")
DOCUMENTS = 192


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
    parent = json.loads(PARENT.read_text())
    if parent["terminal"] != "equality_pre_mlp9_sparse_oracle_edge_candidate" or parent["selected_support"] != "100" or not all(parent["predictions"].values()):
        raise ValueError("frozen discovery authority changed")
    receipt = json.loads(ROW_RECEIPT.read_text())
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if (receipt.get("status") != "frozen_before_any_v2_model_forward"
            or receipt["entries"]["ood_code"]["file_sha256"] != digest(ROWS)
            or payload.get("schema") != "induction_equality_tensor_final_ood_v2_role"
            or payload.get("role") != "ood_code"
            or list(payload["rows"].shape) != [DOCUMENTS, 257]):
        raise ValueError("code-OOD row authority changed")
    _, _, _, _, scales, _ = action_parent.validate_inputs()
    return payload["rows"], scales, parent


def plan():
    rows, _, _ = load_bound()
    batches = math.ceil(len(rows) / action_parent.BATCH)
    return {"schema": "equality_a8_edge_code_ood_confirmation_v1_plan", "ood_documents": len(rows), "tokens_per_document": int(rows.shape[1] - 1), "arms": list(ARMS), "forward_calls": batches * 7, "fits": 0, "new_text": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def reports_for(nll, masks):
    native_effect = nll["absent"] - nll["native"]
    reports = {}
    for arm in ARMS:
        effect = nll["absent"] - nll[arm]
        reports[arm] = {}
        for cell in CELLS:
            selected = masks[cell]
            ns = float(native_effect[selected].sum()); ars = float(effect[selected].sum())
            reports[arm][cell] = {"tokens": int(selected.sum()), "native_effect_sum_nat": ns, "arm_effect_sum_nat": ars, "recovery": ars / ns if abs(ns) > 1e-30 else None, "arm_minus_native_mean_nat": float((nll[arm] - nll["native"])[selected].mean())}
        reports[arm]["halves"] = []
        for lo, hi in ((0, 96), (96, 192)):
            selected = masks["copy_positive"][lo:hi]
            ns = float(native_effect[lo:hi][selected].sum()); ars = float(effect[lo:hi][selected].sum())
            reports[arm]["halves"].append({"documents": 96, "tokens": int(selected.sum()), "native_effect_sum_nat": ns, "recovery": ars / ns if abs(ns) > 1e-30 else None})
    return reports


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
    rows, scales, parent = load_bound()
    masks = diagnosis.build_masks(rows)
    valid = torch.zeros_like(masks["copy_positive"]); valid[:, 64:] = True
    masks["all_noncopy"] = valid & ~masks["copy_positive"]
    if any(not bool(masks[cell].any()) for cell in CELLS):
        raise ValueError("code-OOD mask support changed")
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    nll = {name: [] for name in ("native", "absent", *ARMS)}
    replay = {name: [] for name in ("native_logits", "absent_logits", "native_mlp9", "absent_mlp9", "edge_install")}
    started = time.perf_counter()

    for start in range(0, DOCUMENTS, action_parent.BATCH):
        batch_rows = rows[start:start + action_parent.BATCH]
        tokens = batch_rows[:, :-1].cuda()
        native_authority, native_write, _, _ = r500._captured_forward(model, tokens, direct=True)
        absent_authority, absent_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="late_absent", scales=scales["L5H5"])
        native_logits, native, _ = discovery.trajectory(model, tokens, absent=False)
        absent_logits, absent, _ = discovery.trajectory(model, tokens, absent=True)
        sources = {"native": native, "absent": absent}
        replay["native_logits"].append(float((native_logits - native_authority.float()).norm() / native_authority.float().norm().clamp_min(1e-30)))
        replay["absent_logits"].append(float((absent_logits - absent_authority.float()).norm() / absent_authority.float().norm().clamp_min(1e-30)))
        replay["native_mlp9"].append(float((native["mlp9"] - native_write.float()).norm() / native_write.float().norm().clamp_min(1e-30)))
        replay["absent_mlp9"].append(float((absent["mlp9"] - absent_write.float()).norm() / absent_write.float().norm().clamp_min(1e-30)))
        logits = {"native": native_logits, "absent": absent_logits}
        for arm in ARMS:
            logits[arm], _capture, install_error = discovery.trajectory(model, tokens, absent=True, sources=sources, arm=arm)
            replay["edge_install"].append(install_error)
        targets = batch_rows[:, 1:].cuda()
        for name, value in logits.items():
            nll[name].append(F.cross_entropy(value.reshape(-1, value.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())

    nll = {name: torch.cat(parts) for name, parts in nll.items()}
    reports = reports_for(nll, masks)
    maxima = {name: max(values) for name, values in replay.items()}
    target = reports["100"]["copy_positive"]["recovery"]
    frozen = parent["reports"]["100"]["copy_positive"]["recovery"]
    controls = [reports[arm]["copy_positive"]["recovery"] for arm in ("010", "001")]
    stable_cells = ("copy_near", "copy_far", "copy_one_predecessor", "copy_multiple_predecessors")
    pred_a = bool(max(maxima.values()) <= 2e-6)
    pred_b = bool(pred_a and target >= .85 and abs(target - frozen) <= .15)
    pred_c = bool(pred_a and abs(reports["100"]["all_noncopy"]["arm_minus_native_mean_nat"]) <= .01)
    pred_d = bool(pred_a and max(controls) <= .35 and target >= max(controls) + .40)
    pred_e = bool(pred_a and all(reports["100"][cell]["recovery"] > .70 for cell in stable_cells) and all(row["recovery"] > .70 for row in reports["100"]["halves"]))
    pred_f = bool(reports["100"]["copy_positive"]["native_effect_sum_nat"] > 0 and all(row["native_effect_sum_nat"] > 0 for row in reports["100"]["halves"]))
    predictions = {"pred_a_instrument": pred_a, "pred_b_frozen_a8_transfer": pred_b, "pred_c_preservation": pred_c, "pred_d_singleton_ordering": pred_d, "pred_e_cell_stability": pred_e, "pred_f_removal_live": pred_f}
    terminal = "equality_a8_oracle_edge_code_ood_confirmed" if all(predictions.values()) else "valid_equality_a8_edge_code_ood_null" if pred_a else "invalid"
    result = {"schema": "equality_a8_edge_code_ood_confirmation_v1_result", "terminal": terminal, "predictions": predictions, "frozen_natural_recovery": frozen, "reports": reports, "instrument_maxima": maxima, "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "row_file_sha256": digest(ROWS), "row_receipt_sha256": digest(ROW_RECEIPT), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 0, "gradients": 0, "parameter_updates": 0, "model_loaded": True, "gpu_accessed": True, "queue_touched": True}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Frozen singleton oracle-write confirmation on the separately frozen repository-disjoint code-OOD role; establishes OOD behavioral sufficiency/removal for A8 but does not yet replace its row-specific native write with an extracted executor."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "recovery": {arm: reports[arm]["copy_positive"]["recovery"] for arm in ARMS}, "natural_recovery": frozen, "a8_noncopy_mean": reports["100"]["all_noncopy"]["arm_minus_native_mean_nat"], "a8_halves": reports["100"]["halves"], "instrument": maxima, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
