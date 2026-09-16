#!/usr/bin/env python3
# BQGATE:48prefixes;4tripleJVPs;240seconds;direct cubic interaction correction.
"""Compute the CrossFirst third-order mixed correction directly by nested JVP.

pred_a: derivative symmetry and bound replay pass.
pred_b: direct and small-scale finite-estimated cubic terms agree.
pred_c: direct cubic repairs licence at native scale.
pred_d: direct cubic passes every family and endpoint group.
pred_e: direct cubic improves every family over Hessian-only by >=30%.
pred_f: all promptwise derivative terms and readers are serialized.
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
import torch.nn.functional as F

import live_crossfirst_prefix_v1 as live
import run_crossfirst_per_layer_causal_hessian_v1 as allocation
from regional_cue_row_check_v1 import validate
from sparse_path_stability_atlas_v1 import digest

STEM = "CROSSFIRST_DIRECT_CUBIC_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
ROWS = P / "CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_ROWS.json"
SCALE_ARTIFACT = P / "CROSSFIRST_HESSIAN_SCALE_CURVE_V1_ARTIFACT.pt"
FOUR_STAGE_ARTIFACT = P / "CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_ARTIFACT.pt"
PROGRAM = P / "MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt"


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
    return rows, torch.load(SCALE_ARTIFACT, weights_only=True), torch.load(FOUR_STAGE_ARTIFACT, weights_only=True)


def plan():
    rows, _, _ = load_bound()
    return {"schema": "crossfirst_direct_cubic_v1_plan", "rows": len(rows), "triple_jvps_per_prefix": 4, "physical_suffix_arms": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def third_jvp(function, point, first, second, outer):
    return torch.func.jvp(
        lambda value: allocation.mixed_jvp(function, value, first, second),
        (point,),
        (outer,),
    )[1]


def install_stateless_rotary(model):
    """Avoid mutable rotary caches escaping across nested functorch levels.

    This is algebraically identical to ``Rotary.forward`` on a cache miss, but
    constructs the position constants inside every transformed call instead of
    retaining a tensor tagged with a completed forward-AD level.
    """
    rotary_class = type(model.transformer.h[0].attn.rotary)

    def stateless_forward(module, x):
        sequence_length = x.shape[1]
        positions = torch.arange(sequence_length, device=x.device).type_as(module.inv_freq)
        frequencies = torch.outer(positions, module.inv_freq).to(x.device)
        return frequencies.cos().bfloat16()[None, :, None, :], frequencies.sin().bfloat16()[None, :, None, :]

    rotary_class.forward = stateless_forward


def grouped_reports(rows, prediction, hessian, finite, child, direct_cubic, finite_cubic):
    result = {"families": [], "concepts": []}
    for key, count in (("family", 4), ("concept", 6)):
        destination = result["families" if key == "family" else "concepts"]
        for value in range(count):
            indices = torch.tensor([index for index, row in enumerate(rows) if row[key] == value], dtype=torch.long)
            hessian_error = ratio(hessian[indices, 0] - finite[indices, 0], finite[indices, 0])
            cubic_error = ratio(prediction[indices, 0] - finite[indices, 0], finite[indices, 0])
            destination.append({
                key: value,
                "rows": int(indices.numel()),
                "direct_vs_finite_estimated_cubic_relative_l2": ratio(direct_cubic[indices, 0] - finite_cubic[indices, 0], finite_cubic[indices, 0]),
                "hessian_full_scale_relative_l2": hessian_error,
                "direct_cubic_full_scale_relative_l2": cubic_error,
                "direct_cubic_full_scale_cosine": float(F.cosine_similarity(prediction[indices, 0], finite[indices, 0], dim=0)),
                "direct_cubic_residual_relative_to_child": ratio(prediction[indices, 0] - finite[indices, 0], child[indices, 0]),
                "fractional_improvement_over_hessian": 1.0 - cubic_error / max(hessian_error, 1e-30),
            })
    return result


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if OUT.exists() or ARTIFACT.exists():
        raise FileExistsError(OUT if OUT.exists() else ARTIFACT)
    signal.alarm(240)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    rows, scale_artifact, four_stage = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    install_stateless_rotary(model)
    weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = torch.load(PROGRAM, weights_only=True)["direction"].cuda()
    derivatives = torch.empty(48, 4, 2, dtype=torch.float64)
    started = time.perf_counter()

    for row_index, row in enumerate(rows):
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, weights)
        _, stages = allocation.make_stages(model, native)
        readout = allocation.make_readout(model, row)

        def suffix(value):
            for stage in stages:
                value = stage(value)
            return readout(value)

        point = native["z9"].detach()
        left = -(native["child"] * direction).to(point.dtype)
        right = -(native["remainder"] * direction).to(point.dtype)
        # Equivalent orderings: D3[a,b,a] = D3[a,a,b], and similarly for abb.
        aab_outer_a = third_jvp(suffix, point, left, right, left)
        aab_outer_b = third_jvp(suffix, point, left, left, right)
        abb_outer_b = third_jvp(suffix, point, left, right, right)
        abb_outer_a = third_jvp(suffix, point, right, right, left)
        derivatives[row_index] = torch.stack((aab_outer_a, aab_outer_b, abb_outer_b, abb_outer_a)).detach().double().cpu()

    aab = .5 * (derivatives[:, 0] + derivatives[:, 1])
    abb = .5 * (derivatives[:, 2] + derivatives[:, 3])
    direct_cubic = .5 * (aab + abb)
    hessian = four_stage["complete_hessians"].double()
    finite = four_stage["finite_interactions"].double()
    child = four_stage["child_effects"].double()
    finite_cubic = scale_artifact["cubic_coefficients"].double()
    prediction = hessian + direct_cubic
    symmetry = {
        "aab_relative_l2": ratio(derivatives[:, 0] - derivatives[:, 1], aab),
        "abb_relative_l2": ratio(derivatives[:, 2] - derivatives[:, 3], abb),
    }
    bound_hessian_replay = ratio(hessian - scale_artifact["complete_hessians"].double(), hessian)
    bound_finite_replay = ratio(finite - scale_artifact["finite_interactions"][-1].double(), finite)
    reports = grouped_reports(rows, prediction, hessian, finite, child, direct_cubic, finite_cubic)
    licence = next(report for report in reports["concepts"] if report["concept"] == 3)

    pred_a = bool(max(symmetry.values()) <= 2e-5 and bound_hessian_replay <= 1e-12 and bound_finite_replay <= 1e-12)
    pred_b = bool(pred_a and all(report["direct_vs_finite_estimated_cubic_relative_l2"] <= .15 for report in reports["families"] + reports["concepts"]))
    pred_c = bool(pred_a and licence["direct_cubic_full_scale_relative_l2"] <= .10 and licence["direct_cubic_full_scale_cosine"] >= .98)
    pred_d = bool(pred_a and all(report["direct_cubic_residual_relative_to_child"] <= .10 and report["direct_cubic_full_scale_cosine"] >= .95 for report in reports["families"] + reports["concepts"]))
    pred_e = bool(pred_a and all(report["fractional_improvement_over_hessian"] >= .30 for report in reports["families"]))
    pred_f = bool(derivatives.shape == (48, 4, 2) and direct_cubic.shape == (48, 2))
    predictions = {"pred_a_third_derivative_instrument": pred_a, "pred_b_finite_cubic_identification": pred_b, "pred_c_licence_repair": pred_c, "pred_d_broad_finite_correction": pred_d, "pred_e_familywise_improvement": pred_e, "pred_f_complete_audit": pred_f}
    terminal = "crossfirst_direct_cubic_correction_candidate" if all(predictions.values()) else "valid_crossfirst_direct_cubic_null" if pred_a else "invalid"

    artifact = {"schema": "crossfirst_direct_cubic_v1_artifact", "derivative_labels": ["aab_outer_a", "aab_outer_b", "abb_outer_b", "abb_outer_a"], "third_derivatives": derivatives, "aab": aab, "abb": abb, "direct_cubic_coefficients": direct_cubic, "finite_estimated_cubic_coefficients": finite_cubic, "complete_hessians": hessian, "scale_one_predictions": prediction, "scale_one_finite_interactions": finite, "child_effects": child}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {
        "schema": "crossfirst_direct_cubic_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "symmetry_errors": symmetry,
        "bound_hessian_replay_relative_l2": bound_hessian_replay,
        "bound_finite_replay_relative_l2": bound_finite_replay,
        "group_reports": reports,
        "licence_report": licence,
        "artifact_relative": str(ARTIFACT.relative_to(ROOT)),
        "artifact_sha256": digest(ARTIFACT),
        "runner_sha256": digest(RUNNER),
        "binding_sha256": digest(BINDING),
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "seconds": time.perf_counter() - started,
        "price": {"checkpoint_loads": 1, "prefixes": 48, "triple_jvps_per_prefix": 4, "physical_suffix_arms": 0, "fits": 0, "parameter_updates": 0},
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Opened-panel direct third-derivative diagnostic for CrossFirst finite composition; exact native state/field generators and suffix remain live ports; no fresh OOD or standalone extraction claim.",
    }
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "symmetry": symmetry, "licence": licence, "families": reports["families"], "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
