#!/usr/bin/env python3
# BQGATE:112prefixes;16pairwise mixed JVPs +8mixtures fresh;180seconds;standalone extracted DCT node.
"""Export and fresh-test a compositional contextual-Hessian MLP9 node."""
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
CHECK = PACKAGE / "check.py"
README = PACKAGE / "README.md"
WEIGHTS = PACKAGE / "weights.pt"
FIXTURE = PACKAGE / "fixture.pt"
sys.path[:0] = [str(HERE), str(P), str(ROOT)]

import torch

import live_crossfirst_prefix_v1 as live
import run_crossfirst_per_layer_causal_hessian_v1 as allocation
from sparse_path_stability_atlas_v1 import digest

STEM = "MLP9_CONTEXTUAL_DCT_NODE_FRESH_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
DISCOVERY_ROWS = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ROWS.json"
FRESH_ROWS = P / f"{STEM}_ROWS.json"
CONTEXTUAL_ARTIFACT = P / "MLP9_CONTEXTUAL_DCT_RANGE_V1_ARTIFACT.pt"
RAW_DCT_ARTIFACT = P / "MLP9_DCT_UNSUPERVISED_READER_V1_ARTIFACT.pt"
RANKS = (8, 16, 32)
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


def ratio(numerator, denominator):
    return float(numerator.norm() / denominator.norm().clamp_min(1e-30))


def cosine(first, second):
    return float(torch.nn.functional.cosine_similarity(first.reshape(1, -1).double(), second.reshape(1, -1).double())[0])


def load_bound():
    binding = json.loads(BINDING.read_text())
    if not all(digest(path) == expected for path, expected in binding["files"].items()):
        raise ValueError("bound input changed")
    discovery = json.loads(DISCOVERY_ROWS.read_text())["rows"]
    fresh = json.loads(FRESH_ROWS.read_text())["rows"]
    if len(discovery) != 48 or len(fresh) != 64:
        raise ValueError("row count changed")
    return discovery, fresh, torch.load(CONTEXTUAL_ARTIFACT, weights_only=True), torch.load(RAW_DCT_ARTIFACT, weights_only=True)


def plan():
    discovery, fresh, _, _ = load_bound()
    return {"schema": "mlp9_contextual_dct_node_fresh_v1_plan", "discovery_prefixes": len(discovery), "fresh_prefixes": len(fresh), "directions": 4, "ordered_pairs": 16, "mixture_checks_per_fresh_prefix": 8, "rank_candidates": list(RANKS), "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def pair_tangents(directions, tokens):
    first = []
    second = []
    for left in range(4):
        for right in range(4):
            first.append(directions[left][None, :].expand(tokens, -1))
            second.append(directions[right][None, :].expand(tokens, -1))
    return torch.stack(first), torch.stack(second)


def exact_formula(state, directions, left, right, down, epsilon):
    left_state = state.float() @ left.T
    right_state = state.float() @ right.T
    q0 = (left_state * right_state) @ down.T
    s0 = state.float().square().mean(-1) + epsilon
    left_directions = directions @ left.T
    right_directions = directions @ right.T
    q_first = []
    s_first = []
    for index in range(4):
        q_first.append((left_directions[index] * right_state + left_state * right_directions[index]) @ down.T)
        s_first.append(2 * (state.float() * directions[index]).mean(-1))
    outputs = []
    for first in range(4):
        row = []
        for second in range(4):
            q_mixed = (left_directions[first] * right_directions[second] + left_directions[second] * right_directions[first]) @ down.T
            s_mixed = 2 * (directions[first] * directions[second]).mean()
            row.append(
                q_mixed / s0[..., None]
                - q_first[first] * s_first[second][..., None] / s0.square()[..., None]
                - q_first[second] * s_first[first][..., None] / s0.square()[..., None]
                + q0 * (2 * s_first[first] * s_first[second] / s0.pow(3) - s_mixed / s0.square())[..., None]
            )
        outputs.append(torch.stack(row))
    return torch.stack(outputs)


def stage_responses(stage, state, directions):
    tokens = state.shape[1]
    first, second = pair_tangents(directions, tokens)
    point = state.expand(16, -1, -1).contiguous()
    return allocation.mixed_jvp(stage, point, first, second).reshape(4, 4, 1, tokens, 1152)


def aggregate(records, selector):
    selected = [record for record in records if selector(record)]
    expected_sq = sum(record["expected_sq"] for record in selected)
    error_sq = sum(record["error_sq"] for record in selected)
    predicted_sq = sum(record["predicted_sq"] for record in selected)
    dot = sum(record["dot"] for record in selected)
    return {"relative_l2": (error_sq / max(expected_sq, 1e-30)) ** .5, "cosine": dot / max((expected_sq * predicted_sq) ** .5, 1e-30), "records": len(selected)}


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

    discovery_rows, fresh_rows, contextual, raw_dct = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    weights = live.assembled.load_weights(model.state_dict(), "cuda")
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
        native = live.prepare(model, ids, weights)
        _, stages = allocation.make_stages(model, native)
        response = stage_responses(stages[0], native["z9"], directions).detach().float()
        flat = response.reshape(-1, 1152).double()
        covariance += flat.T @ flat
        symmetry.append(ratio(response - response.transpose(0, 1), response))
        state_replay.append(ratio(stages[0](native["z9"]) - native["h9"], native["h9"]))

    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    spectrum = eigenvalues[order].clamp_min(0)
    candidates = {rank: float(torch.sqrt(spectrum[:rank].sum() / spectrum.sum().clamp_min(1e-30))) for rank in RANKS}
    selected_rank = next((rank for rank in RANKS if candidates[rank] >= .99), RANKS[-1])
    basis = eigenvectors[:, order[:selected_rank]].float()
    program = {
        "schema": "mlp9_contextual_dct_node_v1_weights",
        "left": left.cpu(),
        "right": right.cpu(),
        "projected_down": (basis.T @ down).cpu(),
        "basis": basis.cpu(),
        "directions": directions.cpu(),
        "left_directions": (directions @ left.T).cpu(),
        "right_directions": (directions @ right.T).cpu(),
        "epsilon": epsilon,
        "selected_rank": selected_rank,
    }
    spec = importlib.util.spec_from_file_location("mlp9_contextual_dct_node_execute", EXECUTOR)
    executor = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(executor)
    raw_basis = torch.linalg.qr(raw_dct["fits"][0]["output"].T.double(), mode="reduced").Q.cuda().float()
    control_generator = torch.Generator().manual_seed(2026091631)
    controls = [torch.linalg.qr(torch.randn(1152, selected_rank, generator=control_generator, dtype=torch.float64), mode="reduced").Q.cuda().float() for _ in range(16)]
    mixture_generator = torch.Generator(device="cuda").manual_seed(2026091632)
    mixture_coefficients = torch.randn(8, 2, 4, generator=mixture_generator, device="cuda", dtype=torch.float32)
    mixture_coefficients /= mixture_coefficients.norm(dim=-1, keepdim=True).clamp_min(1e-30)

    records = []
    analytic_records = []
    mixture_records = []
    comparator_error_squares = {"raw_weight_dct": 0.0, **{f"random_{index}": 0.0 for index in range(16)}}
    comparator_expected_sq = 0.0
    fixture = None
    for row_index, row in enumerate(fresh_rows):
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, weights)
        _, stages = allocation.make_stages(model, native)
        expected = stage_responses(stages[0], native["z9"], directions).detach().float()
        analytic = exact_formula(native["z9"], directions, left, right, down, epsilon).detach().float()
        predicted = executor.execute(native["z9"].cpu(), program).cuda().float()
        for first in range(4):
            for second in range(4):
                target = expected[first, second]
                estimate = predicted[first, second]
                exact = analytic[first, second]
                records.append({"row": row_index, "family": row["family"], "first": first, "second": second, "expected_sq": float(target.double().square().sum()), "error_sq": float((estimate - target).double().square().sum()), "predicted_sq": float(estimate.double().square().sum()), "dot": float((estimate.double() * target.double()).sum())})
                analytic_records.append({"row": row_index, "family": row["family"], "first": first, "second": second, "expected_sq": float(target.double().square().sum()), "error_sq": float((exact - target).double().square().sum()), "predicted_sq": float(exact.double().square().sum()), "dot": float((exact.double() * target.double()).sum())})
        flat_expected = expected.reshape(-1, 1152)
        comparator_expected_sq += float(flat_expected.double().square().sum())
        raw_prediction = flat_expected @ raw_basis @ raw_basis.T
        comparator_error_squares["raw_weight_dct"] += float((raw_prediction - flat_expected).double().square().sum())
        for index, control in enumerate(controls):
            control_prediction = flat_expected @ control @ control.T
            comparator_error_squares[f"random_{index}"] += float((control_prediction - flat_expected).double().square().sum())
        for mixture in range(8):
            first_coeff = mixture_coefficients[mixture, 0]
            second_coeff = mixture_coefficients[mixture, 1]
            first_tangent = (first_coeff @ directions)[None, None, :].expand(1, native["z9"].shape[1], 1152)
            second_tangent = (second_coeff @ directions)[None, None, :].expand_as(first_tangent)
            direct = allocation.mixed_jvp(stages[0], native["z9"], first_tangent, second_tangent).detach().float()
            contracted = torch.einsum("i,j,ijbtd->btd", first_coeff, second_coeff, predicted)
            mixture_records.append({"row": row_index, "family": row["family"], "mixture": mixture, "expected_sq": float(direct.double().square().sum()), "error_sq": float((contracted - direct).double().square().sum()), "predicted_sq": float(contracted.double().square().sum()), "dot": float((contracted.double() * direct.double()).sum())})
        if fixture is None:
            fixture = {"schema": "mlp9_contextual_dct_node_v1_fixture", "state": native["z9"].detach().cpu(), "expected": predicted.detach().cpu()}

    overall = aggregate(records, lambda record: True)
    families = [aggregate(records, lambda record, family=family: record["family"] == family) for family in range(4)]
    pairs = [[aggregate(records, lambda record, first=first, second=second: record["first"] == first and record["second"] == second) for second in range(4)] for first in range(4)]
    analytic_overall = aggregate(analytic_records, lambda record: True)
    analytic_families = [aggregate(analytic_records, lambda record, family=family: record["family"] == family) for family in range(4)]
    analytic_pairs = [[aggregate(analytic_records, lambda record, first=first, second=second: record["first"] == first and record["second"] == second) for second in range(4)] for first in range(4)]
    mixtures = [aggregate(mixture_records, lambda record, mixture=mixture: record["mixture"] == mixture) for mixture in range(8)]
    comparator_errors = {name: (error / max(comparator_expected_sq, 1e-30)) ** .5 for name, error in comparator_error_squares.items()}
    random_median = float(torch.tensor([value for name, value in comparator_errors.items() if name.startswith("random_")]).median())

    atomic_torch(WEIGHTS, program)
    atomic_torch(FIXTURE, fixture)
    isolated = subprocess.run([sys.executable, "-I", str(CHECK), str(PACKAGE)], capture_output=True, text=True, check=True, timeout=60)
    isolated_report = json.loads(isolated.stdout)
    native_parameters = left.numel() + right.numel() + down.numel()
    package_parameters = sum(value.numel() for value in program.values() if torch.is_tensor(value))
    parameter_ratio = package_parameters / native_parameters

    pred_a = bool(analytic_overall["relative_l2"] <= 2e-5 and max(report["relative_l2"] for report in analytic_families) <= 2e-5 and max(report["relative_l2"] for row in analytic_pairs for report in row) <= 2e-5 and max(symmetry) <= 2e-5 and max(state_replay) <= 2e-6)
    pred_b = bool(pred_a and selected_rank <= 16 and candidates[selected_rank] >= .99)
    pred_c = bool(pred_a and overall["relative_l2"] <= .05 and overall["cosine"] >= .995 and max(report["relative_l2"] for report in families) <= .05 and min(report["cosine"] for report in families) >= .995 and max(report["relative_l2"] for row in pairs for report in row) <= .10)
    pred_d = bool(pred_a and max(report["relative_l2"] for report in mixtures) <= .08 and min(report["cosine"] for report in mixtures) >= .99)
    pred_e = bool(pred_a and overall["relative_l2"] + .20 <= random_median and overall["relative_l2"] + .20 <= comparator_errors["raw_weight_dct"])
    pred_f = bool(isolated_report["passed"] and isolated_report["relative_l2"] <= 2e-6 and parameter_ratio <= .70)
    predictions = {"pred_a_analytic_instrument": pred_a, "pred_b_sparse_discovery": pred_b, "pred_c_fresh_response_prediction": pred_c, "pred_d_compositional_reuse": pred_d, "pred_e_compression_specificity": pred_e, "pred_f_standalone_extraction": pred_f}
    terminal = "mlp9_contextual_dct_extracted_compositional_node" if all(predictions.values()) else "valid_mlp9_contextual_dct_node_null" if pred_a else "invalid"

    artifact = {"schema": "mlp9_contextual_dct_node_fresh_v1_artifact", "selected_rank": selected_rank, "rank_response_retention": candidates, "spectrum": spectrum.cpu(), "basis": basis.cpu(), "directions": directions.cpu(), "mixture_coefficients": mixture_coefficients.cpu(), "records": records, "analytic_records": analytic_records, "mixture_records": mixture_records}
    atomic_torch(ARTIFACT, artifact)
    result = {"schema": "mlp9_contextual_dct_node_fresh_v1_result", "terminal": terminal, "predictions": predictions, "selected_rank": selected_rank, "rank_response_retention": candidates, "analytic_overall": analytic_overall, "analytic_family_reports": analytic_families, "analytic_pair_reports": analytic_pairs, "fresh_overall": overall, "fresh_family_reports": families, "fresh_pair_reports": pairs, "mixture_reports": mixtures, "symmetry_relative_errors": symmetry, "maximum_state_replay_relative_l2": max(state_replay), "comparator_relative_l2": comparator_errors, "random_comparator_median_relative_l2": random_median, "package": {"relative_directory": str(PACKAGE.relative_to(ROOT)), "weights_sha256": digest(WEIGHTS), "fixture_sha256": digest(FIXTURE), "executor_sha256": digest(EXECUTOR), "check_sha256": digest(CHECK), "readme_sha256": digest(README), "isolated_check": isolated_report, "activation_ports": ["native pre-MLP9 z9"], "frozen_tensor_parameters": package_parameters, "native_bilinear_matrix_parameters": native_parameters, "parameter_ratio": parameter_ratio}, "artifact_relative": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": digest(ARTIFACT), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": {"checkpoint_loads": 1, "opened_discovery_prefixes": 48, "fresh_score_blind_prefixes": 64, "ordered_pair_mixed_jvps_per_prefix": 16, "mixture_mixed_jvps_per_fresh_prefix": 8, "eigendecompositions": 1, "random_output_controls": 16, "isolated_cpu_replays": 1, "suffix_calls": 0, "behavioral_outcomes": 0, "fits_to_model_outputs": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Fresh cross-domain response-level test of a standalone weight/state-derived MLP9 contextual Hessian node with exact four-input bilinear composition; no behavior, suffix, selective removal, or complete-circuit claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "selected_rank": selected_rank, "rank_retention": candidates, "analytic": analytic_overall, "fresh": overall, "family_errors": [report["relative_l2"] for report in families], "maximum_pair_error": max(report["relative_l2"] for row in pairs for report in row), "mixture_errors": [report["relative_l2"] for report in mixtures], "comparators": comparator_errors, "parameter_ratio": parameter_ratio, "isolated": isolated_report, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
