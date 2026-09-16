#!/usr/bin/env python3
# BQGATE:48prefixes;16physical suffix arms;180seconds;finite-scale Hessian curve.
"""Test whether the new-endpoint Hessian miss is genuine higher-order curvature.

pred_a: scale-one physical arms replay the bound artifact.
pred_b: quarter-scale Hessian error <=.15 for every family and concept.
pred_c: licence error grows monotonically and by >=.15 from quarter to full.
pred_d: licence native-scale curvature is >=.20 of its finite interaction.
pred_e: a small-scale cubic diagnostic improves licence error by >=30%.
pred_f: all promptwise scales, arms, readers, and predictions are serialized.
"""
import io
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
HERE = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
sys.path[:0] = [str(HERE), str(P), str(ROOT)]

import torch

import live_crossfirst_prefix_v1 as live
import run_crossfirst_per_layer_causal_hessian_v1 as allocation
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest

STEM = "CROSSFIRST_HESSIAN_SCALE_CURVE_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
ROWS = P / "CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_ROWS.json"
PRIOR_ARTIFACT = P / "CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_ARTIFACT.pt"
PROGRAM = P / "MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt"
SCALES = (0.25, 0.50, 0.75, 1.00)


def atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def ratio(numerator, denominator):
    return float(numerator.norm() / denominator.norm().clamp_min(1e-30))


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    rows = json.loads(ROWS.read_text())["rows"]
    validate(rows)
    if len(rows) != 48:
        raise ValueError("expected 48 rows")
    return rows, torch.load(PRIOR_ARTIFACT, weights_only=True)


def plan():
    rows, _ = load_bound()
    return {"schema": "crossfirst_hessian_scale_curve_v1_plan", "rows": len(rows), "scales": list(SCALES), "physical_suffix_arms_per_prefix": 16, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def grouped_reports(rows, finite, hessian_prediction):
    result = {"families": [], "concepts": []}
    for key, count, width in (("family", 4, 12), ("concept", 6, 8)):
        destination = result["families" if key == "family" else "concepts"]
        for value in range(count):
            indices = torch.tensor([index for index, row in enumerate(rows) if row[key] == value], dtype=torch.long)
            errors = [ratio(hessian_prediction[scale_index, indices, 0] - finite[scale_index, indices, 0], finite[scale_index, indices, 0]) for scale_index in range(len(SCALES))]
            destination.append({key: value, "rows": int(indices.numel()), "target_relative_l2_by_scale": errors})
    return result


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
    from fastload import load_model_fast

    rows, prior = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = torch.load(PROGRAM, weights_only=True)["direction"].cuda()
    physical = torch.empty(4, 48, 4, 2, dtype=torch.float64)
    started = time.perf_counter()

    with torch.no_grad():
        for row_index, row in enumerate(rows):
            ids = torch.tensor([row["ids"]], device="cuda")
            native = live.prepare(model, ids, weights)
            _, stages = allocation.make_stages(model, native)
            readout = allocation.make_readout(model, row)
            left = -(native["child"] * direction).to(native["z9"].dtype)
            right = -(native["remainder"] * direction).to(native["z9"].dtype)
            for scale_index, scale in enumerate(SCALES):
                for arm, edit in enumerate((torch.zeros_like(left), left, right, left + right)):
                    value = native["z9"] + scale * edit
                    for stage in stages:
                        value = stage(value)
                    physical[scale_index, row_index, arm] = readout(value).double().cpu()

    finite = physical[:, :, 3] - physical[:, :, 1] - physical[:, :, 2] + physical[:, :, 0]
    complete_hessian = prior["complete_hessians"].double()
    hessian_prediction = torch.stack([scale * scale * complete_hessian for scale in SCALES])
    replay = ratio(physical[-1] - prior["physical_readouts"].double(), prior["physical_readouts"].double())
    reports = grouped_reports(rows, finite, hessian_prediction)
    quarter_max = max(report["target_relative_l2_by_scale"][0] for report in reports["families"] + reports["concepts"])
    licence = next(report for report in reports["concepts"] if report["concept"] == 3)
    licence_errors = licence["target_relative_l2_by_scale"]
    monotone = all(later + .02 >= earlier for earlier, later in zip(licence_errors, licence_errors[1:]))
    licence_indices = torch.tensor([index for index, row in enumerate(rows) if row["concept"] == 3], dtype=torch.long)
    quadratic_from_quarter = 16.0 * finite[0, licence_indices, 0]
    native_curvature = ratio(finite[-1, licence_indices, 0] - quadratic_from_quarter, finite[-1, licence_indices, 0])

    g_quarter = finite[0] / (SCALES[0] ** 2)
    g_half = finite[1] / (SCALES[1] ** 2)
    cubic_coefficient = (g_half - g_quarter) / (SCALES[1] - SCALES[0])
    cubic_prediction = complete_hessian + cubic_coefficient
    hessian_licence_error = ratio(complete_hessian[licence_indices, 0] - finite[-1, licence_indices, 0], finite[-1, licence_indices, 0])
    cubic_licence_error = ratio(cubic_prediction[licence_indices, 0] - finite[-1, licence_indices, 0], finite[-1, licence_indices, 0])
    cubic_improvement = 1.0 - cubic_licence_error / max(hessian_licence_error, 1e-30)

    pred_a = bool(replay <= 2e-6)
    pred_b = bool(pred_a and quarter_max <= .15)
    pred_c = bool(pred_a and monotone and licence_errors[-1] >= licence_errors[0] + .15)
    pred_d = bool(pred_a and native_curvature >= .20)
    pred_e = bool(pred_a and cubic_improvement >= .30)
    pred_f = bool(physical.shape == (4, 48, 4, 2) and hessian_prediction.shape == (4, 48, 2))
    predictions = {"pred_a_replay": pred_a, "pred_b_local_hessian_validity": pred_b, "pred_c_higher_order_growth": pred_c, "pred_d_material_native_scale_curvature": pred_d, "pred_e_cubic_diagnostic": pred_e, "pred_f_complete_audit": pred_f}
    terminal = "crossfirst_native_scale_higher_order_curvature" if all(predictions.values()) else "valid_crossfirst_hessian_scale_curve_null" if pred_a else "invalid"

    artifact = {"schema": "crossfirst_hessian_scale_curve_v1_artifact", "scales": SCALES, "physical_readouts": physical, "finite_interactions": finite, "complete_hessians": complete_hessian, "hessian_predictions": hessian_prediction, "cubic_coefficients": cubic_coefficient, "cubic_scale_one_predictions": cubic_prediction}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {
        "schema": "crossfirst_hessian_scale_curve_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "scale_one_replay_relative_l2": replay,
        "group_reports": reports,
        "quarter_scale_maximum_target_relative_l2": quarter_max,
        "licence_target_relative_l2_by_scale": licence_errors,
        "licence_monotone_with_slack": monotone,
        "licence_native_curvature_relative_to_finite_interaction": native_curvature,
        "licence_complete_hessian_relative_l2": hessian_licence_error,
        "licence_cubic_relative_l2": cubic_licence_error,
        "licence_cubic_fractional_improvement": cubic_improvement,
        "artifact_relative": str(ARTIFACT.relative_to(ROOT)),
        "artifact_sha256": digest(ARTIFACT),
        "runner_sha256": digest(RUNNER),
        "binding_sha256": digest(BINDING),
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "seconds": time.perf_counter() - started,
        "price": {"checkpoint_loads": 1, "prefixes": 48, "scales": len(SCALES), "physical_suffix_arms_per_prefix": 16, "jvps": 0, "backwards": 0, "fits": 0, "parameter_updates": 0},
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Opened-panel finite-scale diagnostic of the exact CrossFirst mixed Hessian; per-row cubic coefficients are oracle diagnostics, not deployable circuit parameters or fresh evidence.",
    }
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "replay": replay, "quarter_max": quarter_max, "licence_errors": licence_errors, "native_curvature": native_curvature, "cubic_error": cubic_licence_error, "cubic_improvement": cubic_improvement, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
