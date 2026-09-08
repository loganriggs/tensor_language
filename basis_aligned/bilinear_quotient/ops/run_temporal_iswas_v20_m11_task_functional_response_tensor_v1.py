#!/usr/bin/env python3
"""Build the exact v20 task-functional M11 native-factor response tensor."""

# BQGATE: EXPERIMENT pred_a_authority_gradient_hooks_finiteness_tensor_shape_and_exact_price pred_b_factor_tangent_closes_full_m11_effect_on_all_panels pred_c_paraphrase_damage_uses_shared_context_signed_core pred_d_decomposition_is_weight_folded_and_diagnostic_only
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
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_v20_m11_task_functional_response_tensor_v1.json"
V20_RESULT = ROOT / "circuits/followups/temporal_iswas_v20_frozen_top16_gain125_confirmation_v1_result.json"
CAPABILITY = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v20_capability_v1_result.json"
TASK_RESULT = ROOT / "circuits/followups/temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1_result.json"
TANGENT_RUNNER = ROOT / "ops/run_temporal_iswas_m11_crossconstruction_task_tangent_factor_order_v1.py"
BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v20.py"
OUT = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1_result.json"
TENSOR_OUT = ROOT / "circuits/followups/temporal_iswas_v20_m11_task_functional_response_tensor_v1.npz"
CANDIDATE_ID = "cross_task.temporal_iswas.v20_m11_task_functional_response_tensor_v1"
EXPECTED = {
    "prior": "52ade34bdb7adf8ba126b7016b1ca27acdc040b343bc3c6cea1d1f99eaaafb8d",
    "v20_result": "7d3981cf12ef8ff00f83ca0420e9a3d0787b57434f9c2dee86c8d96ee6c13a8e",
    "capability": "d03534d18f9d3662e5765f7d79294f9ea5f45d09f59d891a093c1739bd20c2ef",
    "task_result": "e0af6cc965d2d6a560dbde548e191268c77fcac7a5817ad54ea2ccaefe7050b7",
    "tangent_runner": "78ac32a90ef3a0182fe2eb024a3b687ddefe2b5fb1b816df5dd0febaf8f857ec",
    "builder": "cafe9120c4ba8fdbeff27c5a2d05a247c2000641fd401529a4ca0a089137d1d9",
}
PANELS = ("A1", "A2", "P", "C")
PRICE = {"checkpoint_loads": 1, "model_forwards": 16, "sequence_evaluations": 256,
         "scored_token_positions": 512, "transformer_backwards": 4,
         "model_updates": 0, "fit_parameters": 0,
         "stored_float32_response_values": 4 * 16 * 4608}
BARS = {"tangent_cosine": .90, "tangent_direction": .875,
        "shared_absolute_cosine": .50, "top16_signed_p_share": .50}
PREDICTION_KEYS = (
    "pred_a_authority_gradient_hooks_finiteness_tensor_shape_and_exact_price",
    "pred_b_factor_tangent_closes_full_m11_effect_on_all_panels",
    "pred_c_paraphrase_damage_uses_shared_context_signed_core",
    "pred_d_decomposition_is_weight_folded_and_diagnostic_only",
)


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def response_tensor(backend, rows):
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
        raise RuntimeError("M11 response-tensor gradient was not live")
    gradient = saved["leaf"].grad.detach().float()
    x1, x0 = saved["reader"].float(), absent_readers["M11"].float()
    mlp = model.transformer.h[11].mlp
    hidden_delta = ((x1 @ mlp.Left.weight.detach().float().T)
                    * (x1 @ mlp.Right.weight.detach().float().T)
                    - (x0 @ mlp.Left.weight.detach().float().T)
                    * (x0 @ mlp.Right.weight.detach().float().T))
    factor_tangent = hidden_delta * (gradient @ mlp.Down.weight.detach().float())
    row_factor = torch.stack([factor_tangent[row, list(row_positions)].sum(dim=0)
                              for row, row_positions in enumerate(positions)])
    predicted = row_factor.sum(dim=1)
    with torch.no_grad():
        absent, calls["M11_absent"] = transfer.run_tensor_arm(
            backend, batch, base_heads, donor_heads, base_source, donor_source, positions,
            {"M11": absent_readers["M11"]})
    actual = margin.detach().float().cpu().numpy() - parent.toward_donor_margin(absent)
    report = parent.metrics(predicted.detach().cpu().numpy(), actual)
    return row_factor.detach().cpu().numpy().astype(np.float32), report, calls, float(gradient.abs().max())


def cosine(a, b):
    return float(np.dot(a, b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-30))


def main():
    paths = {"prior": PRIOR, "v20_result": V20_RESULT, "capability": CAPABILITY,
             "task_result": TASK_RESULT, "tangent_runner": TANGENT_RUNNER, "builder": BUILDER}
    observed = {name: sha(path) for name, path in paths.items()}
    v20, capability, task_result = (json.loads(path.read_text())
                                    for path in (V20_RESULT, CAPABILITY, TASK_RESULT))
    all_rows = fresh.build_rows()
    jointly = {family: set(capability["jointly_capable_row_ids"][family]) for family in ("A1", "A2")}
    rows = {family: [row for row in all_rows if row["transform_id"] == family and (
        family in ("P", "C") or row["row_id"] in jointly[family])] for family in PANELS}
    top16 = tuple(int(index) for index in task_result["stable_factor_order_first256"][:16])
    authority_ok = bool(observed == EXPECTED
        and v20.get("terminal") == "fixed_program_nonselective_or_unstable"
        and capability.get("causal_outcomes_opened") is False
        and len(top16) == len(set(top16)) == 16
        and all(len(panel) == 16 for panel in rows.values()))
    dry = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
           "model_loaded": False, "queue_touched": False, "authority_ok": authority_ok,
           "tensor_shape": [4, 16, 4608], "top16": top16, "price": PRICE,
           "evidence_status": "post_failure_diagnostic"}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if not authority_ok: raise RuntimeError("v20 task-functional tensor authority changed")
    if OUT.exists() or TENSOR_OUT.exists(): raise FileExistsError("refusing response-tensor overwrite")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    for parameter in backend.model.parameters(): parameter.requires_grad_(False)
    arrays, reports, calls, gradient_max = [], {}, {}, {}
    for family in PANELS:
        array, reports[family], calls[family], gradient_max[family] = response_tensor(
            backend, rows[family])
        arrays.append(array)
    Q = np.stack(arrays)
    means = Q.mean(axis=1, dtype=np.float64)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    normalized = means / np.maximum(norms, 1e-30)
    _u, singular, modes = np.linalg.svd(normalized, full_matrices=False)
    target = .5 * (normalized[0] + normalized[1])
    target_p_cosine = cosine(target, normalized[2])
    p_total = float(means[2].sum())
    p_top16 = float(means[2, list(top16)].sum())
    p_share = p_top16 / p_total if abs(p_total) > 1e-30 else float("nan")
    pairwise = {f"{PANELS[i]}:{PANELS[j]}": cosine(normalized[i], normalized[j])
                for i in range(len(PANELS)) for j in range(i + 1, len(PANELS))}
    mode_top_indices = [np.argsort(np.abs(mode), kind="stable")[-32:][::-1].tolist()
                        for mode in modes]
    fd, temp_name = tempfile.mkstemp(prefix=TENSOR_OUT.stem + ".", suffix=".npz", dir=TENSOR_OUT.parent)
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, Q=Q, panels=np.asarray(PANELS),
            row_ids=np.asarray([[str(row["row_id"]) for row in rows[family]] for family in PANELS]),
            factor_indices=np.arange(Q.shape[2], dtype=np.int32), panel_means=means.astype(np.float32),
            normalized_panel_means=normalized.astype(np.float32), singular_values=singular.astype(np.float32),
            right_modes=modes.astype(np.float32), frozen_top16=np.asarray(top16, dtype=np.int32))
        os.replace(temp_path, TENSOR_OUT)
    finally:
        if temp_path.exists(): temp_path.unlink()
    tensor_sha = sha(TENSOR_OUT)
    A = bool(authority_ok and Q.shape == (4, 16, 4608) and np.isfinite(Q).all()
        and all(tangent.selection_calls_ok(value) for value in calls.values())
        and all(value > 0 and math.isfinite(value) for value in gradient_max.values())
        and PRICE["model_forwards"] == len(PANELS) * 4
        and PRICE["sequence_evaluations"] == PRICE["model_forwards"] * 16
        and PRICE["transformer_backwards"] == len(PANELS))
    B = all(reports[family]["cosine"] >= BARS["tangent_cosine"]
            and reports[family]["direction_agreement"] >= BARS["tangent_direction"]
            for family in PANELS)
    C = abs(target_p_cosine) >= BARS["shared_absolute_cosine"] and p_share >= BARS["top16_signed_p_share"]
    D = True
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = ("invalid_instrument" if not A or not B else
                "shared_context_signed_factor_core" if C else "splittable_answer_branch_hypothesis")
    result = {"schema": "temporal_iswas_v20_m11_task_functional_response_tensor_result_v1",
        "candidate_id": CANDIDATE_ID, "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "evidence_status": "post_failure_diagnostic_only",
        "tensor_artifact": {"path": str(TENSOR_OUT.relative_to(ROOT)), "sha256": tensor_sha,
                            "shape": list(Q.shape), "dtype": str(Q.dtype)},
        "tangent_reports": reports, "gradient_max_abs": gradient_max, "calls": calls,
        "panel_pairwise_cosine": pairwise, "pooled_target_p_cosine": target_p_cosine,
        "frozen_top16_signed_p_tangent_share": p_share,
        "frozen_top16_panel_mean_contributions": {
            family: means[i, list(top16)].tolist() for i, family in enumerate(PANELS)},
        "normalized_panel_singular_values": singular.tolist(),
        "mode_top_abs_factor_indices": mode_top_indices,
        "selected_or_promoted_factor_subset": None,
        "predictions": predictions, "terminal": terminal, "bars": BARS, "price": PRICE}
    if not finite(result): raise RuntimeError("non-finite response-tensor result")
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("tangent_reports", "gradient_max_abs",
        "panel_pairwise_cosine", "pooled_target_p_cosine", "frozen_top16_signed_p_tangent_share",
        "normalized_panel_singular_values", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
