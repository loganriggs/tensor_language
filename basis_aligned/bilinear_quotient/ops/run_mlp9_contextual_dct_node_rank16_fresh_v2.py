#!/usr/bin/env python3
# BQGATE:48discovery+64fresh prefixes;rank16 pairwise DCT node;180seconds;second-panel correction.
"""Prospective second-panel test of the frozen rank-16 contextual DCT node."""
import importlib.util
import io
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
P = ROOT / "basis_aligned/polynomial_causal"
HERE = Path(__file__).resolve().parent
RUNNER = Path(__file__).resolve()
PACKAGE = P / "extracted_circuits/mlp9_contextual_dct_node_v1"
EXECUTOR = PACKAGE / "execute.py"
README = PACKAGE / "README_RANK16_V2.md"
WEIGHTS = PACKAGE / "weights_rank16_v2.pt"
FIXTURE = PACKAGE / "fixture_rank16_v2.pt"
sys.path[:0] = [str(HERE), str(P), str(ROOT)]

import torch

import live_crossfirst_prefix_v1 as live
import run_crossfirst_per_layer_causal_hessian_v1 as allocation
import run_mlp9_contextual_dct_node_fresh_v1 as v1
from sparse_path_stability_atlas_v1 import digest

STEM = "MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
DISCOVERY_ROWS = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ROWS.json"
FRESH_ROWS = P / f"{STEM}_ROWS.json"
CONTEXTUAL_ARTIFACT = P / "MLP9_CONTEXTUAL_DCT_RANGE_V1_ARTIFACT.pt"
V1_RESULT = P / "MLP9_CONTEXTUAL_DCT_NODE_FRESH_V1_RESULT.json"
V1_ARTIFACT = P / "MLP9_CONTEXTUAL_DCT_NODE_FRESH_V1_ARTIFACT.pt"
RANK = 16
STABLE_SLICE = slice(16, 20)


def atomic_bytes(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def atomic_json(path, value):
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def atomic_torch(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic_bytes(path, buffer.getvalue())


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    discovery = json.loads(DISCOVERY_ROWS.read_text())["rows"]
    fresh = json.loads(FRESH_ROWS.read_text())["rows"]
    if len(discovery) != 48 or len(fresh) != 64:
        raise ValueError("row count changed")
    return discovery, fresh, torch.load(CONTEXTUAL_ARTIFACT, weights_only=True), json.loads(V1_RESULT.read_text()), torch.load(V1_ARTIFACT, weights_only=True)


def plan():
    discovery, fresh, _, _, _ = load_bound()
    return {"schema": "mlp9_contextual_dct_node_rank16_fresh_v2_plan", "discovery_prefixes": len(discovery), "fresh_prefixes": len(fresh), "rank": RANK, "ordered_pairs": 16, "mixtures": 8, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True))
        return
    if any(path.exists() for path in (OUT, ARTIFACT, WEIGHTS, FIXTURE)):
        raise FileExistsError("output namespace already exists")
    signal.alarm(180)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    from fastload import load_model_fast

    discovery_rows, fresh_rows, contextual, v1_result, v1_artifact = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    live_weights = live.assembled.load_weights(model.state_dict(), "cuda")
    mlp = model.transformer.h[9].mlp
    left = mlp.Left.weight.detach().float()
    right = mlp.Right.weight.detach().float()
    down = mlp.Down.weight.detach().float()
    directions = contextual["input_directions"][STABLE_SLICE].cuda().float()
    epsilon = torch.finfo(torch.float32).eps
    covariance = torch.zeros(1152, 1152, dtype=torch.float64, device="cuda")
    symmetry = []
    state_replay = []
    started = time.perf_counter()
    for row in discovery_rows:
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, live_weights)
        _, stages = allocation.make_stages(model, native)
        response = v1.stage_responses(stages[0], native["z9"], directions).detach().float()
        flat = response.reshape(-1, 1152).double()
        covariance += flat.T @ flat
        symmetry.append(v1.ratio(response - response.transpose(0, 1), response))
        state_replay.append(v1.ratio(stages[0](native["z9"]) - native["h9"], native["h9"]))
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    spectrum = eigenvalues[order].clamp_min(0)
    retention = float(torch.sqrt(spectrum[:RANK].sum() / spectrum.sum().clamp_min(1e-30)))
    basis = eigenvectors[:, order[:RANK]].float()
    program = {"schema": "mlp9_contextual_dct_node_rank16_v2_weights", "left": left.cpu(), "right": right.cpu(), "projected_down": (basis.T @ down).cpu(), "basis": basis.cpu(), "directions": directions.cpu(), "left_directions": (directions @ left.T).cpu(), "right_directions": (directions @ right.T).cpu(), "epsilon": epsilon, "selected_rank": RANK}
    spec = importlib.util.spec_from_file_location("mlp9_contextual_dct_node_execute_v2", EXECUTOR)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    mixture_coefficients = v1_artifact["mixture_coefficients"].cuda().float()
    control_generator = torch.Generator().manual_seed(2026091641)
    controls = [torch.linalg.qr(torch.randn(1152, RANK, generator=control_generator, dtype=torch.float64), mode="reduced").Q.cuda().float() for _ in range(16)]

    records = []
    analytic_records = []
    mixture_records = []
    control_error_sq = [0.0] * len(controls)
    control_expected_sq = 0.0
    fixture = None
    for row_index, row in enumerate(fresh_rows):
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, live_weights)
        _, stages = allocation.make_stages(model, native)
        expected = v1.stage_responses(stages[0], native["z9"], directions).detach().float()
        analytic = v1.exact_formula(native["z9"], directions, left, right, down, epsilon).detach().float()
        predicted = executor.execute(native["z9"].cpu(), program).cuda().float()
        for first in range(4):
            for second in range(4):
                target = expected[first, second]
                estimate = predicted[first, second]
                exact = analytic[first, second]
                records.append({"row": row_index, "family": row["family"], "first": first, "second": second, "expected_sq": float(target.double().square().sum()), "error_sq": float((estimate - target).double().square().sum()), "predicted_sq": float(estimate.double().square().sum()), "dot": float((estimate.double() * target.double()).sum())})
                analytic_records.append({"row": row_index, "family": row["family"], "first": first, "second": second, "expected_sq": float(target.double().square().sum()), "error_sq": float((exact - target).double().square().sum()), "predicted_sq": float(exact.double().square().sum()), "dot": float((exact.double() * target.double()).sum())})
        flat = expected.reshape(-1, 1152)
        control_expected_sq += float(flat.double().square().sum())
        for index, control in enumerate(controls):
            projected = flat @ control @ control.T
            control_error_sq[index] += float((projected - flat).double().square().sum())
        for mixture in range(8):
            first_coeff = mixture_coefficients[mixture, 0]
            second_coeff = mixture_coefficients[mixture, 1]
            first_tangent = (first_coeff @ directions)[None, None, :].expand(1, native["z9"].shape[1], 1152)
            second_tangent = (second_coeff @ directions)[None, None, :].expand_as(first_tangent)
            direct = allocation.mixed_jvp(stages[0], native["z9"], first_tangent, second_tangent).detach().float()
            contracted = torch.einsum("i,j,ijbtd->btd", first_coeff, second_coeff, predicted)
            mixture_records.append({"row": row_index, "family": row["family"], "mixture": mixture, "expected_sq": float(direct.double().square().sum()), "error_sq": float((contracted - direct).double().square().sum()), "predicted_sq": float(contracted.double().square().sum()), "dot": float((contracted.double() * direct.double()).sum())})
        if fixture is None:
            fixture = {"schema": "mlp9_contextual_dct_node_rank16_v2_fixture", "state": native["z9"].detach().cpu(), "expected": predicted.detach().cpu()}

    overall = v1.aggregate(records, lambda record: True)
    families = [v1.aggregate(records, lambda record, family=family: record["family"] == family) for family in range(4)]
    pairs = [[v1.aggregate(records, lambda record, first=first, second=second: record["first"] == first and record["second"] == second) for second in range(4)] for first in range(4)]
    analytic_overall = v1.aggregate(analytic_records, lambda record: True)
    analytic_families = [v1.aggregate(analytic_records, lambda record, family=family: record["family"] == family) for family in range(4)]
    analytic_pairs = [[v1.aggregate(analytic_records, lambda record, first=first, second=second: record["first"] == first and record["second"] == second) for second in range(4)] for first in range(4)]
    mixtures = [v1.aggregate(mixture_records, lambda record, mixture=mixture: record["mixture"] == mixture) for mixture in range(8)]
    control_errors = [(value / max(control_expected_sq, 1e-30)) ** .5 for value in control_error_sq]
    control_median = float(torch.tensor(control_errors).median())

    atomic_torch(WEIGHTS, program)
    atomic_torch(FIXTURE, fixture)
    isolated_code = """import importlib.util,json,sys,torch
package,executor_path,weights_path,fixture_path=sys.argv[1:]
spec=importlib.util.spec_from_file_location('node',executor_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
program=torch.load(weights_path,weights_only=True,map_location='cpu');fixture=torch.load(fixture_path,weights_only=True,map_location='cpu')
actual=module.execute(fixture['state'],program).float();expected=fixture['expected'].float();error=float((actual-expected).norm()/expected.norm().clamp_min(1e-30))
print(json.dumps({'relative_l2':error,'shape':list(actual.shape),'passed':error<=2e-6},sort_keys=True));raise SystemExit(0 if error<=2e-6 else 1)
"""
    isolated = subprocess.run([sys.executable, "-I", "-c", isolated_code, str(PACKAGE), str(EXECUTOR), str(WEIGHTS), str(FIXTURE)], capture_output=True, text=True, check=True, timeout=60)
    isolated_report = json.loads(isolated.stdout)
    native_parameters = left.numel() + right.numel() + down.numel()
    package_parameters = sum(value.numel() for value in program.values() if torch.is_tensor(value))
    parameter_ratio = package_parameters / native_parameters
    v1_overall = v1_result["fresh_overall"]["relative_l2"]
    v1_max_pair = max(report["relative_l2"] for row in v1_result["fresh_pair_reports"] for report in row)
    max_pair = max(report["relative_l2"] for row in pairs for report in row)

    pred_a = bool(analytic_overall["relative_l2"] <= 2e-5 and max(report["relative_l2"] for report in analytic_families) <= 2e-5 and max(report["relative_l2"] for row in analytic_pairs for report in row) <= 2e-5 and max(symmetry) <= 2e-5 and max(state_replay) <= 2e-6)
    pred_b = bool(pred_a and retention >= .9999)
    pred_c = bool(pred_a and overall["relative_l2"] <= .015 and overall["cosine"] >= .999 and max(report["relative_l2"] for report in families) <= .015 and min(report["cosine"] for report in families) >= .999 and max_pair <= .15)
    pred_d = bool(pred_a and max(report["relative_l2"] for report in mixtures) <= .03 and min(report["cosine"] for report in mixtures) >= .999)
    pred_e = bool(pred_a and overall["relative_l2"] <= .5 * v1_overall and max_pair <= .5 * v1_max_pair and overall["relative_l2"] + .20 <= control_median)
    pred_f = bool(isolated_report["passed"] and isolated_report["relative_l2"] <= 2e-6 and parameter_ratio <= .70)
    predictions = {"pred_a_analytic_instrument": pred_a, "pred_b_registered_rank": pred_b, "pred_c_second_panel_prediction": pred_c, "pred_d_compositional_reuse": pred_d, "pred_e_correction_value": pred_e, "pred_f_standalone_extraction": pred_f}
    terminal = "mlp9_contextual_dct_rank16_extracted_compositional_node" if all(predictions.values()) else "valid_mlp9_contextual_dct_rank16_null" if pred_a else "invalid"

    artifact = {"schema": "mlp9_contextual_dct_node_rank16_fresh_v2_artifact", "rank": RANK, "retention": retention, "spectrum": spectrum.cpu(), "basis": basis.cpu(), "directions": directions.cpu(), "mixture_coefficients": mixture_coefficients.cpu(), "records": records, "analytic_records": analytic_records, "mixture_records": mixture_records}
    atomic_torch(ARTIFACT, artifact)
    result = {"schema": "mlp9_contextual_dct_node_rank16_fresh_v2_result", "terminal": terminal, "predictions": predictions, "rank": RANK, "discovery_response_retention": retention, "analytic_overall": analytic_overall, "analytic_family_reports": analytic_families, "analytic_pair_reports": analytic_pairs, "fresh_overall": overall, "fresh_family_reports": families, "fresh_pair_reports": pairs, "mixture_reports": mixtures, "symmetry_relative_errors": symmetry, "maximum_state_replay_relative_l2": max(state_replay), "random_rank16_relative_l2": control_errors, "random_rank16_median_relative_l2": control_median, "v1_comparison": {"v1_overall_relative_l2": v1_overall, "v2_over_v1_overall_error": overall["relative_l2"] / v1_overall, "v1_maximum_pair_relative_l2": v1_max_pair, "v2_maximum_pair_relative_l2": max_pair, "v2_over_v1_maximum_pair_error": max_pair / v1_max_pair}, "package": {"relative_directory": str(PACKAGE.relative_to(ROOT)), "weights_filename": WEIGHTS.name, "fixture_filename": FIXTURE.name, "weights_sha256": digest(WEIGHTS), "fixture_sha256": digest(FIXTURE), "executor_sha256": digest(EXECUTOR), "readme_sha256": digest(README), "isolated_check": isolated_report, "activation_ports": ["native pre-MLP9 z9"], "frozen_tensor_parameters": package_parameters, "native_bilinear_matrix_parameters": native_parameters, "parameter_ratio": parameter_ratio}, "artifact_relative": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": digest(ARTIFACT), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": {"checkpoint_loads": 1, "opened_discovery_prefixes": 48, "second_panel_fresh_prefixes": 64, "ordered_pair_mixed_jvps_per_prefix": 16, "mixture_mixed_jvps_per_fresh_prefix": 8, "eigendecompositions": 1, "random_rank16_controls": 16, "isolated_cpu_replays": 1, "suffix_calls": 0, "behavioral_outcomes": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Second-panel fresh cross-domain response test of the preregistered rank-16 standalone MLP9 contextual Hessian node; no suffix, behavior, selective removal, or complete-circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "retention": retention, "analytic": analytic_overall, "fresh": overall, "family_errors": [report["relative_l2"] for report in families], "max_pair": max_pair, "mixture_errors": [report["relative_l2"] for report in mixtures], "v1_comparison": result["v1_comparison"], "random_median": control_median, "parameter_ratio": parameter_ratio, "isolated": isolated_report, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
