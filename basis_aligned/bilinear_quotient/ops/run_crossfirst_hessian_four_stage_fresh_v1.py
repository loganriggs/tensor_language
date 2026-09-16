#!/usr/bin/env python3
# BQGATE:48prefixes;4physical suffix arms;18stage nested JVP;240seconds;fresh four-stage composition.
"""Fresh new-endpoint test of the frozen four-stage Hessian correction.

pred_a: capability, baseline replay, and complete allocation closure pass.
pred_b: familywise child-relative composition error <=.10.
pred_c: familywise finite-vector relative error/cosine/sign gates pass.
pred_d: candidate minimax beats every one-stage-ablation minimum by 10%.
pred_e: each unseen endpoint passes child-relative and cosine gates.
pred_f: unrelated-reader residuals pass and all promptwise terms are saved.
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

STEM = "CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1"
BINDING = P / f"{STEM}_BINDING.json"
ROWS = P / f"{STEM}_ROWS.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
PROGRAM = P / "MLP9_CROSSFIRST_RESPONSE_V1_PROGRAM.pt"
CANDIDATE = (0, 2, 15, 17)  # mlp9, mlp10, attention17, readout
ABLATIONS = tuple(tuple(index for index in CANDIDATE if index != omitted) for omitted in CANDIDATE)


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
    rows = json.loads(ROWS.read_text())["rows"]
    validate(rows)
    if len(rows) != 48:
        raise ValueError("expected 48 rows")
    return rows


def ratio(numerator, denominator):
    return float(numerator.norm() / denominator.norm().clamp_min(1e-30))


def plan():
    rows = load_bound()
    return {"schema": "crossfirst_hessian_four_stage_fresh_v1_plan", "rows": len(rows), "physical_suffix_arms": 4, "stages": 18, "candidate_indices": list(CANDIDATE), "ablation_nulls": len(ABLATIONS), "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


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

    rows = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    weights = live.assembled.load_weights(model.state_dict(), "cuda")
    direction = torch.load(PROGRAM, weights_only=True)["direction"].cuda()
    stage_names = ["mlp9"] + [name for layer in range(10, 18) for name in (f"attention{layer}", f"mlp{layer}")] + ["readout"]
    expected_names = ["mlp9", "mlp10", "attention17", "readout"]
    if [stage_names[index] for index in CANDIDATE] != expected_names:
        raise RuntimeError("candidate index drift")

    allocations = torch.empty(48, 18, 2, dtype=torch.float64)
    complete = torch.empty(48, 2, dtype=torch.float64)
    graph_baseline = torch.empty(48, 2, dtype=torch.float64)
    physical = torch.empty(48, 4, 2, dtype=torch.float64)
    state_replay = []
    started = time.perf_counter()

    for row_index, row in enumerate(rows):
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, weights)
        names, stages = allocation.make_stages(model, native)
        readout = allocation.make_readout(model, row)
        if names + ["readout"] != stage_names:
            raise RuntimeError("stage order changed")

        graph_states = [native["z9"].detach().requires_grad_(True)]
        for stage in stages:
            graph_states.append(stage(graph_states[-1]))
        graph_output = readout(graph_states[-1])
        graph_baseline[row_index] = graph_output.detach().double().cpu()
        endpoint_gradients = [torch.autograd.grad(graph_output[endpoint], graph_states[1:], retain_graph=True, allow_unused=False) for endpoint in range(2)]
        gradients = [torch.stack((endpoint_gradients[0][index], endpoint_gradients[1][index])) for index in range(len(stages))]
        detached_states = [state.detach() for state in graph_states]
        state_replay.append(float((detached_states[1] - native["h9"]).float().norm() / native["h9"].float().norm().clamp_min(1e-30)))
        left = -(native["child"] * direction).to(native["z9"].dtype)
        right = -(native["remainder"] * direction).to(native["z9"].dtype)
        terms, full = allocation.allocation_for_writer(stages, detached_states, graph_states, gradients, readout, left, right)
        allocations[row_index] = terms.detach().double().cpu()
        complete[row_index] = full.detach().double().cpu()

        with torch.no_grad():
            for arm, edit in enumerate((torch.zeros_like(left), left, right, left + right)):
                value = native["z9"] + edit
                for stage in stages:
                    value = stage(value)
                physical[row_index, arm] = readout(value).double().cpu()
        del graph_output, graph_states, endpoint_gradients, gradients, detached_states

    finite = physical[:, 3] - physical[:, 1] - physical[:, 2] + physical[:, 0]
    child_effect = physical[:, 1] - physical[:, 0]
    candidate_prediction = allocations[:, CANDIDATE, :].sum(1)
    closure = [ratio(allocations[:, :, endpoint].sum(1) - complete[:, endpoint], complete[:, endpoint]) for endpoint in range(2)]
    baseline_replay = ratio(graph_baseline - physical[:, 0], physical[:, 0])
    capability = [float(physical[index, 0, 0] - physical[index + 1, 0, 0]) for index in range(0, 48, 2)]

    family_reports = []
    for family in range(4):
        sl = slice(12 * family, 12 * (family + 1))
        residual = candidate_prediction[sl] - finite[sl]
        family_reports.append({
            "family": family,
            "target_residual_relative_to_child": ratio(residual[:, 0], child_effect[sl, 0]),
            "target_finite_interaction_relative_l2": ratio(residual[:, 0], finite[sl, 0]),
            "target_finite_interaction_cosine": float(F.cosine_similarity(candidate_prediction[sl, 0], finite[sl, 0], dim=0)),
            "target_sign_agreement": float((torch.sign(candidate_prediction[sl, 0]) == torch.sign(finite[sl, 0])).double().mean()),
            "control_residual_relative_to_child_target": ratio(residual[:, 1], child_effect[sl, 0]),
        })
    candidate_minimax = max(report["target_residual_relative_to_child"] for report in family_reports)

    ablation_reports = []
    for indices in ABLATIONS:
        prediction = allocations[:, indices, 0].sum(1)
        errors = []
        for family in range(4):
            sl = slice(12 * family, 12 * (family + 1))
            errors.append(ratio(prediction[sl] - finite[sl, 0], child_effect[sl, 0]))
        ablation_reports.append({"indices": list(indices), "stages": [stage_names[index] for index in indices], "family_child_relative_errors": errors, "minimax_child_relative_error": max(errors)})
    best_ablation = min(report["minimax_child_relative_error"] for report in ablation_reports)

    concept_reports = []
    for concept in range(6):
        indices = torch.tensor([index for index, row in enumerate(rows) if row["concept"] == concept], dtype=torch.long)
        predicted = candidate_prediction[indices, 0]
        observed = finite[indices, 0]
        concept_reports.append({
            "concept": concept,
            "uk_token": rows[2 * concept]["uk_token"],
            "us_token": rows[2 * concept]["us_token"],
            "target_residual_relative_to_child": ratio(predicted - observed, child_effect[indices, 0]),
            "target_finite_interaction_cosine": float(F.cosine_similarity(predicted, observed, dim=0)),
        })

    pred_a = bool(all(value > 0 for value in capability) and baseline_replay <= 2e-6 and max(state_replay) <= 2e-6 and max(closure) <= 2e-5)
    pred_b = bool(pred_a and all(report["target_residual_relative_to_child"] <= .10 for report in family_reports))
    pred_c = bool(pred_a and all(report["target_finite_interaction_relative_l2"] <= .50 and report["target_finite_interaction_cosine"] >= .95 and report["target_sign_agreement"] >= .80 for report in family_reports))
    pred_d = bool(pred_a and candidate_minimax <= .90 * best_ablation)
    pred_e = bool(pred_a and all(report["target_residual_relative_to_child"] <= .15 and report["target_finite_interaction_cosine"] >= .80 for report in concept_reports))
    pred_f = bool(pred_a and all(report["control_residual_relative_to_child_target"] <= .10 for report in family_reports) and allocations.shape == (48, 18, 2))
    predictions = {"pred_a_instrument_capability": pred_a, "pred_b_familywise_composition": pred_b, "pred_c_finite_interaction_fidelity": pred_c, "pred_d_stage_necessity": pred_d, "pred_e_endpoint_transfer": pred_e, "pred_f_control_preservation": pred_f}
    terminal = "fresh_crossfirst_four_stage_hessian_composition_rule" if all(predictions.values()) else "valid_crossfirst_hessian_four_stage_fresh_null" if pred_a else "invalid"

    artifact = {"schema": "crossfirst_hessian_four_stage_fresh_v1_artifact", "stage_names": stage_names, "candidate_indices": CANDIDATE, "ablation_indices": ABLATIONS, "allocations": allocations, "complete_hessians": complete, "graph_baseline_readouts": graph_baseline, "physical_readouts": physical, "finite_interactions": finite, "child_effects": child_effect, "capability_contrasts": capability}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {
        "schema": "crossfirst_hessian_four_stage_fresh_v1_result",
        "terminal": terminal,
        "predictions": predictions,
        "candidate_stages": expected_names,
        "candidate_minimax_child_relative_error": candidate_minimax,
        "best_ablation_minimax_child_relative_error": best_ablation,
        "family_reports": family_reports,
        "concept_reports": concept_reports,
        "ablation_reports": ablation_reports,
        "capability_minimum_paired_contrast": min(capability),
        "allocation_closure_relative_l2": closure,
        "baseline_graph_replay_relative_l2": baseline_replay,
        "maximum_stage_state_replay_relative_l2": max(state_replay),
        "artifact_relative": str(ARTIFACT.relative_to(ROOT)),
        "artifact_sha256": digest(ARTIFACT),
        "runner_sha256": digest(RUNNER),
        "binding_sha256": digest(BINDING),
        "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "seconds": time.perf_counter() - started,
        "price": {"checkpoint_loads": 1, "prefixes": 48, "physical_suffix_arms_per_prefix": 4, "nested_jvp_stages_per_prefix": 17, "readout_hessian_stages_per_prefix": 1, "fits": 0, "parameter_updates": 0},
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "scope": "Fresh construction-and-endpoint test of a frozen four-stage CrossFirst Hessian correction; native child/remainder generators and suffix remain live ports; no corpus-OOD, standalone extraction, or selective-removal claim.",
    }
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "families": family_reports, "concepts": concept_reports, "candidate_minimax": candidate_minimax, "best_ablation": best_ablation, "closure": closure, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
