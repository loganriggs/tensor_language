#!/usr/bin/env python3
# BQGATE:128opened equality documents;128forwards;180seconds;consumer-response basis discovery.
"""Discover a semantic MLP9 response basis for the equality/copy reader."""
import io
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
sys.path[:0] = [str(HERE), str(P), str(BQ), str(ROOT)]

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import equality_matcher_causal_action_quotient_rung498 as action_parent
import equality_matcher_mlp9_reader_calibration_rung500 as r500
from sparse_path_stability_atlas_v1 import digest

STEM = "MLP9_EQUALITY_CONSUMER_RESPONSE_BASIS_DISCOVERY_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
BRIDGE_RESULT = P / "MLP9_CONTEXTUAL_DCT_EQUALITY_RESPONSE_DISCOVERY_V1_RESULT.json"
RANKS = (4, 8, 16, 32, 64)
PANELS = {"discovery": (500, 564), "confirmation": (564, 628)}
CELLS = ("copy_positive", "noncopy_equality", "all_noncopy")


def atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    if digest(RUNNER) != binding["runner_sha256"]:
        raise ValueError("runner changed")
    bridge = json.loads(BRIDGE_RESULT.read_text())
    if bridge["terminal"] != "valid_mlp9_contextual_dct_equality_response_null" or not bridge["predictions"]["pred_a_authority_and_instrument"]:
        raise ValueError("bridge authority changed")
    rows, _, _, _, scales, _ = action_parent.validate_inputs()
    return rows, scales, torch.load(DCT_WEIGHTS, map_location="cpu", weights_only=True)


def plan():
    rows, _, _ = load_bound()
    return {"schema": "mlp9_equality_consumer_response_basis_discovery_v1_plan", "panels": {name: stop - start for name, (start, stop) in PANELS.items()}, "tokens_per_document": int(rows.shape[1] - 1), "ranks": list(RANKS), "forward_calls": sum(math.ceil((stop - start) / action_parent.BATCH) * 4 for start, stop in PANELS.values()), "eigendecompositions": 1, "new_text": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def coverage(values, basis):
    values = values.float()
    return float((values @ basis @ basis.T).norm() / values.norm().clamp_min(1e-30))


def cosine(left, right):
    left, right = left.reshape(-1).double(), right.reshape(-1).double()
    return float((left * right).sum() / (left.norm() * right.norm()).clamp_min(1e-30))


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists() or ARTIFACT.exists():
        raise FileExistsError(OUT if OUT.exists() else ARTIFACT)
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, scales, dct_program = load_bound()
    dct_basis = dct_program["basis"].cuda().float()
    masks = r500._task_masks(rows)
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True)
    model.eval()
    collected = {panel: {kind: {cell: [] for cell in CELLS} for kind in ("native", "restored")} for panel in PANELS}
    task = {panel: {name: [] for name in ("native", "absent", "restored")} for panel in PANELS}
    replay_logits, replay_writes = [], []
    started = time.perf_counter()

    for panel, (panel_start, panel_stop) in PANELS.items():
        for start in range(panel_start, panel_stop, action_parent.BATCH):
            stop = min(start + action_parent.BATCH, panel_stop)
            batch_rows = rows[start:stop]
            tokens = batch_rows[:, :-1].cuda()
            direct_logits, direct_write, _, _ = r500._captured_forward(model, tokens, direct=True)
            native_logits, native_write, _, _ = r500._captured_forward(model, tokens, pair=None)
            absent_logits, absent_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="late_absent", scales=scales["L5H5"])
            restored_logits, restored_write, _, _ = r500._captured_forward(model, tokens, pair=action_parent.PAIRS[0], background="early_present", state="score_donor", scales=scales["L5H5"])
            replay_logits.append(float((direct_logits.float() - native_logits.float()).abs().max()))
            replay_writes.append(float((direct_write.float() - native_write.float()).abs().max()))
            targets = batch_rows[:, 1:].cuda()
            for name, logits in (("native", native_logits), ("absent", absent_logits), ("restored", restored_logits)):
                task[panel][name].append(F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none").reshape(len(batch_rows), -1).cpu())
            responses = {"native": absent_write.float() - native_write.float(), "restored": absent_write.float() - restored_write.float()}
            for kind, response in responses.items():
                for cell in CELLS:
                    collected[panel][kind][cell].append(response[masks[cell][start:stop].cuda()].cpu())

    vectors = {panel: {kind: {cell: torch.cat(parts).cuda().float() for cell, parts in cells.items()} for kind, cells in kinds.items()} for panel, kinds in collected.items()}
    discovery = vectors["discovery"]["native"]["copy_positive"]
    covariance = discovery.double().T @ discovery.double()
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    spectrum = eigenvalues[order].clamp_min(0)
    full_basis = eigenvectors[:, order[:max(RANKS)]].float()
    reports = {}
    for rank in RANKS:
        basis = full_basis[:, :rank]
        reports[str(rank)] = {panel: {kind: {cell: coverage(vectors[panel][kind][cell], basis) for cell in CELLS} for kind in ("native", "restored")} for panel in PANELS}
    selected_rank = next((rank for rank in RANKS if reports[str(rank)]["discovery"]["native"]["copy_positive"] >= .80), None)
    selected_basis = None if selected_rank is None else full_basis[:, :selected_rank]
    selected_report = None if selected_rank is None else reports[str(selected_rank)]
    projected_cosine = None
    if selected_basis is not None:
        projected_cosine = cosine(vectors["confirmation"]["native"]["copy_positive"] @ selected_basis, vectors["confirmation"]["restored"]["copy_positive"] @ selected_basis)
    task_reports = {}
    for panel, values in task.items():
        values = {name: torch.cat(parts) for name, parts in values.items()}
        start, stop = PANELS[panel]
        selected = masks["copy_positive"][start:stop]
        native_effect = values["absent"] - values["native"]
        restored_effect = values["absent"] - values["restored"]
        native_sum = float(native_effect[selected].sum())
        restored_sum = float(restored_effect[selected].sum())
        task_reports[panel] = {"copy_tokens": int(selected.sum()), "native_effect_sum_nat": native_sum, "restored_effect_sum_nat": restored_sum, "recovery": restored_sum / native_sum if abs(native_sum) > 1e-30 else None}
    dct_confirmation = {kind: {cell: coverage(vectors["confirmation"][kind][cell], dct_basis) for cell in CELLS} for kind in ("native", "restored")}
    pred_a = bool(max(replay_logits) == 0.0 and max(replay_writes) == 0.0 and all(task_reports[p]["copy_tokens"] > 0 and task_reports[p]["native_effect_sum_nat"] > 0 for p in PANELS))
    pred_b = bool(pred_a and selected_rank is not None and selected_rank <= 32 and selected_report["confirmation"]["native"]["copy_positive"] >= .70)
    pred_c = bool(pred_a and selected_rank is not None and selected_report["confirmation"]["restored"]["copy_positive"] >= .65 and projected_cosine >= .75)
    pred_d = bool(pred_a and selected_rank is not None and all(selected_report["confirmation"][kind]["copy_positive"] >= selected_report["confirmation"][kind][control] + .10 for kind in ("native", "restored") for control in ("noncopy_equality", "all_noncopy")))
    rank16 = reports["16"]["confirmation"]
    pred_e = bool(pred_a and all(rank16[kind]["copy_positive"] >= dct_confirmation[kind]["copy_positive"] + .30 for kind in ("native", "restored")))
    pred_f = bool(pred_a and all(.70 <= task_reports[panel]["recovery"] <= 1.30 for panel in PANELS))
    predictions = {"pred_a_instrument": pred_a, "pred_b_low_rank_transfer": pred_b, "pred_c_causal_background_reuse": pred_c, "pred_d_semantic_selectivity": pred_d, "pred_e_generic_basis_improvement": pred_e, "pred_f_task_stability": pred_f}
    terminal = "mlp9_equality_consumer_response_basis_candidate" if all(predictions.values()) else "valid_mlp9_equality_consumer_response_basis_null" if pred_a else "invalid"
    artifact = {"schema": "mlp9_equality_consumer_response_basis_discovery_v1_artifact", "ranks": RANKS, "spectrum": spectrum.cpu(), "basis_rank64": full_basis.cpu(), "selected_rank": selected_rank, "selected_basis": None if selected_basis is None else selected_basis.cpu(), "reports": reports}
    buffer = io.BytesIO(); torch.save(artifact, buffer); atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {"schema": "mlp9_equality_consumer_response_basis_discovery_v1_result", "terminal": terminal, "predictions": predictions, "selected_rank": selected_rank, "selected_report": selected_report, "rank_reports": reports, "confirmation_dct_rank16": dct_confirmation, "confirmation_projected_native_restored_cosine": projected_cosine, "task_reports": task_reports, "spectrum_top64": spectrum[:64].tolist(), "artifact_relative": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": digest(ARTIFACT), "maximum_native_replay_logit_absolute": max(replay_logits), "maximum_native_replay_write_absolute": max(replay_writes), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": checkpoint.weights_sha256, "seconds": time.perf_counter() - started, "price": planned | {"checkpoint_loads": 1, "fits": 1, "gradients": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Behavior-conditioned MLP9 output-basis discovery and opened-split confirmation for the established equality/copy response; no new rows, fresh OOD, causal projection/removal, upstream-source extraction, or complete-circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_rank": selected_rank, "selected": selected_report, "rank16_confirmation": rank16, "dct_confirmation": dct_confirmation, "projected_cosine": projected_cosine, "task": task_reports, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
