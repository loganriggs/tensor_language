#!/usr/bin/env python3
"""Split the v20 M11 task-functional tensor into exact writer and reader factors."""

# BQGATE: EXPERIMENT pred_a_authority_gradients_hooks_masks_finiteness_and_exact_price pred_b_split_reconstructs_immutable_task_functional_tensor pred_c_sign_reversal_is_reader_dominated pred_d_result_is_diagnostic_and_does_not_select_a_repair
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np

from circuit_fast_screen_managed_runner import atomic_create_json
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20 as fresh
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
import run_temporal_iswas_m11_crossconstruction_exact_factor_greedy_v1 as factor_executor
import run_temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1 as tangent
import run_temporal_iswas_v17_a11_m11_reader_loss_output_rescue_v1 as parent
import run_temporal_iswas_v18_frozen_shared_tensor_transfer_v1 as transfer


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_m11_writer_reader_factor_split_v1.json"
RESPONSE_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1_result.json"
RESPONSE_TENSOR = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
V20_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_frozen_top16_gain125_confirmation_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v20_capability_v1_result.json"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20.py"
TANGENT_RUNNER = ROOT / "ops/run_temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1_result.json"
TENSOR_OUT = ROOT / "circuits/followups/temporal_iswas_v20_m11_writer_reader_factor_split_v1.npz"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_m11_writer_reader_factor_split_v1"
EXPECTED = {
    "prior": "b5d0300d68252a4cb7f10d46480da3c778daacd54bef54276c49127116fb57bc",
    "response_result": "da97abea6f3dae8721c2cda383e0b7af844217051e199046a23c6b9c264da396",
    "response_tensor": "a79ff1b0e8eac3e3b135a8cb02eaa58e83c317ade23f0d65950b90b2ef2dff12",
    "v20_result": "7d3981cf12ef8ff00f83ca0420e9a3d0787b57434f9c2dee86c8d96ee6c13a8e",
    "capability": "d03534d18f9d3662e5765f7d79294f9ea5f45d09f59d891a093c1739bd20c2ef",
    "builder": "cafe9120c4ba8fdbeff27c5a2d05a247c2000641fd401529a4ca0a089137d1d9",
    "tangent_runner": "78ac32a90ef3a0182fe2eb024a3b687ddefe2b5fb1b816df5dd0febaf8f857ec",
}
PANELS = ("A1", "A2", "P", "C")
PRICE = {"checkpoint_loads": 1, "model_forwards": 16, "sequence_evaluations": 256,
         "transformer_backwards": 4, "model_updates": 0, "fit_parameters": 0}
BARS = {"split_max_abs": 1e-6, "split_relative_squared": 1e-10,
        "reader_cosine_max": -.50, "writer_cosine_min_exclusive": -.50}
PREDICTION_KEYS = (
    "pred_a_authority_gradients_hooks_masks_finiteness_and_exact_price",
    "pred_b_split_reconstructs_immutable_task_functional_tensor",
    "pred_c_sign_reversal_is_reader_dominated",
    "pred_d_result_is_diagnostic_and_does_not_select_a_repair",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def cosine(a, b):
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30))


def split_panel(backend, rows):
    torch, model = backend.torch, backend.model
    batch = das._batch(backend, rows, side="base")
    donor_batch = das._batch(backend, rows, side="donor")
    positions = tuple(tuple(range(int(query) + 1)) for query in batch.semantic_positions)
    base_source, donor_source = transfer.suffix_position_rows(batch, donor_batch)
    calls = {"leaf": 0}
    with torch.no_grad():
        native_bundle, base_heads, calls["native_source"] = parent.capture_source_heads(
            lambda: factor_executor.capture_m11(lambda: parent.full_forward(backend, batch), model),
            model, batch)
        _native, absent_readers, _absent_outputs, calls["native_sites"] = native_bundle
        _donor, donor_heads, calls["donor_source"] = parent.capture_source_heads(
            lambda: parent.full_forward(backend, donor_batch), model, donor_batch)
    saved = {}
    def m11_pre(_module, arguments): saved["reader"] = arguments[0].detach().clone()
    def m11_post(_module, _arguments, output):
        calls["leaf"] += 1
        leaf = output.detach().requires_grad_(True)
        saved["leaf"] = leaf
        return leaf
    pre = model.transformer.h[11].mlp.register_forward_pre_hook(m11_pre)
    post = model.transformer.h[11].mlp.register_forward_hook(m11_post)
    try:
        (logits, lengths), calls["writer_source"] = transfer.with_source_heads_aligned(
            lambda: tangent.full_logits(backend, batch), model, batch, base_heads, donor_heads,
            base_source, donor_source)
    finally:
        pre.remove(); post.remove()
    margin = tangent.margin_from_logits(backend, batch, logits, lengths)
    margin.sum().backward()
    if calls["leaf"] != 1 or saved["leaf"].grad is None:
        raise RuntimeError("M11 writer-reader gradient was not live")
    gradient = saved["leaf"].grad.detach().float()
    x1, x0 = saved["reader"].float(), absent_readers["M11"].float()
    mlp = model.transformer.h[11].mlp
    H = ((x1 @ mlp.Left.weight.detach().float().T)
         * (x1 @ mlp.Right.weight.detach().float().T)
         - (x0 @ mlp.Left.weight.detach().float().T)
         * (x0 @ mlp.Right.weight.detach().float().T))
    R = gradient @ mlp.Down.weight.detach().float()
    mask = torch.zeros(H.shape[:2], dtype=torch.bool, device=H.device)
    for row, row_positions in enumerate(positions): mask[row, list(row_positions)] = True
    Q = torch.stack([(H[row, list(row_positions)] * R[row, list(row_positions)]).sum(dim=0)
                     for row, row_positions in enumerate(positions)])
    with torch.no_grad():
        absent, calls["M11_absent"] = transfer.run_tensor_arm(
            backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
            {"M11": absent_readers["M11"]})
    actual = margin.detach().float().cpu().numpy() - parent.toward_donor_margin(absent)
    report = parent.metrics(Q.sum(dim=1).detach().cpu().numpy(), actual)
    return (H.detach().cpu().numpy().astype(np.float32),
            R.detach().cpu().numpy().astype(np.float32),
            mask.detach().cpu().numpy(), Q.detach().cpu().numpy().astype(np.float32),
            report, calls, float(gradient.abs().max()))


def valid_mean(array, mask):
    return array[mask].mean(axis=0, dtype=np.float64)


def cross_vector(H, Hmask, R, Rmask):
    values = []
    for row in range(H.shape[0]):
        hp, rp = np.flatnonzero(Hmask[row]), np.flatnonzero(Rmask[row])
        count = min(len(hp), len(rp))
        values.append((H[row, hp[-count:]] * R[row, rp[-count:]]).sum(axis=0, dtype=np.float64))
    return np.stack(values).mean(axis=0)


def main():
    paths = {"prior": PRIOR, "response_result": RESPONSE_RESULT,
             "response_tensor": RESPONSE_TENSOR, "v20_result": V20_RESULT,
             "capability": CAPABILITY, "builder": BUILDER, "tangent_runner": TANGENT_RUNNER}
    observed = {name: sha(path) for name, path in paths.items()}
    response, capability = json.loads(RESPONSE_RESULT.read_text()), json.loads(CAPABILITY.read_text())
    old = np.load(RESPONSE_TENSOR, allow_pickle=False)
    old_Q = old["Q"]
    all_rows = fresh.build_rows()
    jointly = {family: set(capability["jointly_capable_row_ids"][family]) for family in ("A1", "A2")}
    rows = {family: [row for row in all_rows if row["transform_id"] == family and (
        family in ("P", "C") or row["row_id"] in jointly[family])] for family in PANELS}
    authority_ok = bool(observed == EXPECTED and response.get("terminal") == "shared_context_signed_factor_core"
        and old_Q.shape == (4, 16, 4608) and all(len(panel) == 16 for panel in rows.values()))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "panels": {key: len(value) for key, value in rows.items()}, "price": PRICE,
           "evidence_status": "post_failure_read_write_diagnostic"}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v20 writer-reader split authority changed")
    if OUT.exists() or TENSOR_OUT.exists(): raise FileExistsError("refusing writer-reader overwrite")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    Hs, Rs, masks, Qs, reports, calls, gradient_max = {}, {}, {}, {}, {}, {}, {}
    for family in PANELS:
        (Hs[family], Rs[family], masks[family], Qs[family], reports[family], calls[family],
         gradient_max[family]) = split_panel(backend, rows[family])
    Q = np.stack([Qs[family] for family in PANELS])
    diff = Q.astype(np.float64) - old_Q.astype(np.float64)
    split_max = float(np.max(np.abs(diff)))
    split_rse = float(np.square(diff).sum() / max(float(np.square(old_Q.astype(np.float64)).sum()), 1e-30))
    max_len = max(value.shape[1] for value in Hs.values())
    Hpad = np.zeros((4, 16, max_len, 4608), dtype=np.float32)
    Rpad = np.zeros_like(Hpad)
    Mpad = np.zeros((4, 16, max_len), dtype=bool)
    for index, family in enumerate(PANELS):
        length = Hs[family].shape[1]
        Hpad[index, :, :length] = Hs[family]
        Rpad[index, :, :length] = Rs[family]
        Mpad[index, :, :length] = masks[family]
    Hmean = {family: valid_mean(Hs[family], masks[family]) for family in PANELS}
    Rmean = {family: valid_mean(Rs[family], masks[family]) for family in PANELS}
    target_H = .5 * (Hmean["A1"] / max(np.linalg.norm(Hmean["A1"]), 1e-30)
                     + Hmean["A2"] / max(np.linalg.norm(Hmean["A2"]), 1e-30))
    target_R = .5 * (Rmean["A1"] / max(np.linalg.norm(Rmean["A1"]), 1e-30)
                     + Rmean["A2"] / max(np.linalg.norm(Rmean["A2"]), 1e-30))
    writer_cos = cosine(target_H, Hmean["P"])
    reader_cos = cosine(target_R, Rmean["P"])
    native_A = .5 * (cross_vector(Hs["A1"], masks["A1"], Rs["A1"], masks["A1"])
                     + cross_vector(Hs["A2"], masks["A2"], Rs["A2"], masks["A2"]))
    A_writer_P_reader = .5 * (cross_vector(Hs["A1"], masks["A1"], Rs["P"], masks["P"])
                               + cross_vector(Hs["A2"], masks["A2"], Rs["P"], masks["P"]))
    P_writer_A_reader = .5 * (cross_vector(Hs["P"], masks["P"], Rs["A1"], masks["A1"])
                               + cross_vector(Hs["P"], masks["P"], Rs["A2"], masks["A2"]))
    cross = {"A_writer_A_reader": float(native_A.sum()),
             "A_writer_P_reader": float(A_writer_P_reader.sum()),
             "P_writer_A_reader": float(P_writer_A_reader.sum()),
             "P_writer_P_reader": float(cross_vector(Hs["P"], masks["P"], Rs["P"], masks["P"]).sum())}
    fd, temp_name = tempfile.mkstemp(prefix=TENSOR_OUT.stem + ".", suffix=".npz", dir=TENSOR_OUT.parent)
    os.close(fd); temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, H=Hpad, R=Rpad, mask=Mpad, Q=Q,
            panels=np.asarray(PANELS), row_ids=np.asarray([
                [str(row["row_id"]) for row in rows[family]] for family in PANELS]),
            factor_indices=np.arange(4608, dtype=np.int32))
        os.replace(temp_path, TENSOR_OUT)
    finally:
        if temp_path.exists(): temp_path.unlink()
    tensor_sha = sha(TENSOR_OUT)
    A = bool(authority_ok and np.isfinite(Hpad).all() and np.isfinite(Rpad).all()
        and all(tangent.selection_calls_ok(value) for value in calls.values())
        and all(value > 0 and math.isfinite(value) for value in gradient_max.values())
        and PRICE["model_forwards"] == len(PANELS) * 4
        and PRICE["sequence_evaluations"] == PRICE["model_forwards"] * 16
        and PRICE["transformer_backwards"] == len(PANELS))
    B = split_max <= BARS["split_max_abs"] and split_rse <= BARS["split_relative_squared"]
    C = bool(reader_cos <= BARS["reader_cosine_max"]
             and writer_cos > BARS["writer_cosine_min_exclusive"]
             and np.sign(cross["A_writer_P_reader"]) == -np.sign(cross["A_writer_A_reader"]))
    D = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A or not B else
                "reader_dominated_context_sign" if C else "writer_or_interaction_dominated_context_sign")
    result = {"schema": "temporal_iswas_v20_m11_writer_reader_factor_split_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "post_failure_read_write_diagnostic_only",
        "tensor_artifact": {"path": str(TENSOR_OUT.relative_to(ROOT)), "sha256": tensor_sha,
                            "H_R_shape": list(Hpad.shape), "mask_shape": list(Mpad.shape)},
        "split_reconstruction": {"max_abs_error": split_max, "relative_squared_error": split_rse},
        "pooled_target_p_writer_mean_cosine": writer_cos,
        "pooled_target_p_reader_mean_cosine": reader_cos,
        "suffix_aligned_cross_contraction_total": cross,
        "tangent_reports": reports, "gradient_max_abs": gradient_max, "calls": calls,
        "selected_factor_gate_or_dose": None, "predictions": predictions,
        "terminal": terminal, "bars": BARS, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("split_reconstruction",
        "pooled_target_p_writer_mean_cosine", "pooled_target_p_reader_mean_cosine",
        "suffix_aligned_cross_contraction_total", "tangent_reports", "predictions",
        "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
