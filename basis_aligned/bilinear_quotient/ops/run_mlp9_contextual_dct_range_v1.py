#!/usr/bin/env python3
# BQGATE:96prefixes;20batched double-JVP directions;96suffix gradients;180seconds;contextual DCT range.
"""Behavior-blind contextual MLP9 Hessian range and reader transfer test."""
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

STEM = "MLP9_CONTEXTUAL_DCT_RANGE_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
DISCOVERY_ROWS = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ROWS.json"
CONFIRMATION_ROWS = P / "CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_ROWS.json"
RAW_DCT = P / "MLP9_DCT_UNSUPERVISED_READER_V1_ARTIFACT.pt"
RANK_AUDIT = P / "MLP9_DCT_UNSUPERVISED_READER_RANK_STABILITY_V1_AUDIT.json"
RANK = 8
RANDOM_INPUTS = 16
STABLE_INDICES = (0, 1, 2, 7)


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
    discovery = json.loads(DISCOVERY_ROWS.read_text())["rows"]
    confirmation = json.loads(CONFIRMATION_ROWS.read_text())["rows"]
    validate(discovery)
    validate(confirmation)
    if len(discovery) != 48 or len(confirmation) != 48:
        raise ValueError("expected two 48-row panels")
    audit = json.loads(RANK_AUDIT.read_text())
    if tuple(pair[0] for pair in audit["stable_pairs_first_to_second"]) != STABLE_INDICES:
        raise ValueError("stable DCT indices changed")
    return discovery, confirmation, torch.load(RAW_DCT, weights_only=True)


def plan():
    discovery, confirmation, _ = load_bound()
    return {"schema": "mlp9_contextual_dct_range_v1_plan", "discovery_rows": len(discovery), "confirmation_rows": len(confirmation), "input_directions": RANDOM_INPUTS + len(STABLE_INDICES), "rank": RANK, "double_jvp_batches": 96, "suffix_gradients": 96, "physical_interventions": 0, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def make_input_directions(raw_dct, device):
    generator = torch.Generator().manual_seed(2026091621)
    random, _ = torch.linalg.qr(torch.randn(1152, RANDOM_INPUTS, generator=generator, dtype=torch.float64), mode="reduced")
    stable = raw_dct["fits"][0]["input"][list(STABLE_INDICES)].double()
    directions = torch.cat((random.T, stable), dim=0)
    return directions.to(device=device, dtype=torch.float32)


def top_basis(covariance, rank=RANK):
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvectors[:, order[:rank]], eigenvalues[order]


def response_retention(covariance, basis):
    retained = torch.trace(basis.T @ covariance.double() @ basis).clamp_min(0)
    return float(torch.sqrt(retained / torch.trace(covariance.double()).clamp_min(1e-30)))


def reader_coverage(reader, basis):
    flattened = reader.reshape(-1, 1152).double()
    return ratio(flattened @ basis @ basis.T, flattened)


def collect_panel(model, weights, rows, directions, symmetry_checks):
    combined = torch.zeros(1152, 1152, dtype=torch.float64, device="cuda")
    random_cov = torch.zeros_like(combined)
    stable_cov = torch.zeros_like(combined)
    readers = []
    state_replay = []
    symmetry = []
    for row_index, row in enumerate(rows):
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, weights)
        _, stages = allocation.make_stages(model, native)
        stage = stages[0]
        point = native["z9"].expand(len(directions), -1, -1).contiguous()
        tangent = directions[:, None, :].expand_as(point).contiguous()
        responses = allocation.mixed_jvp(stage, point, tangent, tangent).detach().float().reshape(-1, 1152)
        random_responses = responses[: RANDOM_INPUTS * native["z9"].shape[1]]
        # reshape first so direction, not flattened row order, selects the probe family
        by_direction = responses.reshape(len(directions), native["z9"].shape[1], 1152)
        random_responses = by_direction[:RANDOM_INPUTS].reshape(-1, 1152)
        stable_responses = by_direction[RANDOM_INPUTS:].reshape(-1, 1152)
        row_covariance = responses.T.double() @ responses.double()
        combined += row_covariance
        random_cov += random_responses.T.double() @ random_responses.double()
        stable_cov += stable_responses.T.double() @ stable_responses.double()

        h9 = stage(native["z9"]).detach().requires_grad_(True)
        state_replay.append(ratio(h9.detach() - native["h9"], native["h9"]))
        value = h9
        for downstream in stages[1:]:
            value = downstream(value)
        readout = allocation.make_readout(model, row)
        output = readout(value)
        readers.append(torch.autograd.grad(output[0] - output[1], h9)[0].detach().double().cpu())

        if len(symmetry) < symmetry_checks:
            first = tangent[0:1]
            second = tangent[1:2]
            origin = native["z9"]
            forward = allocation.mixed_jvp(stage, origin, first, second)
            reverse = allocation.mixed_jvp(stage, origin, second, first)
            symmetry.append(ratio(forward - reverse, .5 * (forward + reverse)))
    return {"combined_covariance": combined.cpu(), "random_covariance": random_cov.cpu(), "stable_covariance": stable_cov.cpu(), "readers": readers, "state_replay": state_replay, "symmetry": symmetry}


def panel_report(panel, rows, bases, raw_basis, controls):
    reader = torch.cat(panel["readers"], dim=1)
    report = {"response_retention": {}, "reader_coverage": {}, "family_reader_coverage": {}, "prompt_reader_coverage": {}}
    for name, basis in bases.items():
        report["response_retention"][name] = response_retention(panel["combined_covariance"], basis)
        report["reader_coverage"][name] = reader_coverage(reader, basis)
        report["family_reader_coverage"][name] = []
        report["prompt_reader_coverage"][name] = [reader_coverage(value, basis) for value in panel["readers"]]
        for family in range(4):
            family_reader = torch.cat([panel["readers"][index] for index, row in enumerate(rows) if row["family"] == family], dim=1)
            report["family_reader_coverage"][name].append(reader_coverage(family_reader, basis))
    report["reader_coverage"]["raw_weight_dct"] = reader_coverage(reader, raw_basis)
    report["random_control_reader_coverage"] = [reader_coverage(reader, basis) for basis in controls]
    return report


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

    discovery_rows, confirmation_rows, raw_dct = load_bound()
    model = load_model_fast().cuda().eval()
    model.requires_grad_(False)
    weights = live.assembled.load_weights(model.state_dict(), "cuda")
    directions = make_input_directions(raw_dct, "cuda")
    started = time.perf_counter()
    discovery = collect_panel(model, weights, discovery_rows, directions, 4)
    combined_basis, combined_spectrum = top_basis(discovery["combined_covariance"])
    random_basis, random_spectrum = top_basis(discovery["random_covariance"])
    stable_basis, stable_spectrum = top_basis(discovery["stable_covariance"])
    bases = {"combined": combined_basis, "random_input": random_basis, "stable_dct_input": stable_basis}
    raw_basis = torch.linalg.qr(raw_dct["fits"][0]["output"].T.double(), mode="reduced").Q
    control_generator = torch.Generator().manual_seed(2026091622)
    controls = [torch.linalg.qr(torch.randn(1152, RANK, generator=control_generator, dtype=torch.float64), mode="reduced").Q for _ in range(16)]
    discovery_report = panel_report(discovery, discovery_rows, bases, raw_basis, controls)
    confirmation = collect_panel(model, weights, confirmation_rows, directions, 0)
    confirmation_report = panel_report(confirmation, confirmation_rows, bases, raw_basis, controls)

    symmetry = discovery["symmetry"]
    state_replay = discovery["state_replay"] + confirmation["state_replay"]
    discovery_random_median = float(torch.tensor(discovery_report["random_control_reader_coverage"]).median())
    confirmation_random_max = max(confirmation_report["random_control_reader_coverage"])
    discovery_coverage = discovery_report["reader_coverage"]["combined"]
    confirmation_coverage = confirmation_report["reader_coverage"]["combined"]
    pred_a = bool(max(state_replay) <= 2e-6 and max(symmetry) <= 2e-5)
    pred_b = bool(pred_a and discovery_report["response_retention"]["combined"] >= .50 and confirmation_report["response_retention"]["combined"] >= .40)
    pred_c = bool(pred_a and discovery_coverage >= .30 and discovery_coverage >= discovery_random_median + .15 and discovery_coverage >= discovery_report["reader_coverage"]["raw_weight_dct"] + .15)
    pred_d = bool(pred_a and confirmation_coverage >= .25 and min(confirmation_report["family_reader_coverage"]["combined"]) >= .20 and confirmation_coverage > confirmation_random_max)
    pred_e = bool(pred_a and confirmation_coverage >= confirmation_report["reader_coverage"]["random_input"] + .03 and confirmation_coverage >= confirmation_report["reader_coverage"]["stable_dct_input"] + .03)
    pred_f = bool(all(value.shape == (1152, RANK) for value in bases.values()) and len(discovery["readers"]) == len(confirmation["readers"]) == 48)
    predictions = {"pred_a_instrument": pred_a, "pred_b_contextual_compression": pred_b, "pred_c_discovery_reader": pred_c, "pred_d_endpoint_transfer": pred_d, "pred_e_stable_dct_input_value": pred_e, "pred_f_complete_audit": pred_f}
    terminal = "mlp9_contextual_dct_reader_candidate" if all(predictions.values()) else "valid_mlp9_contextual_dct_range_null" if pred_a else "invalid"

    artifact = {"schema": "mlp9_contextual_dct_range_v1_artifact", "input_directions": directions.cpu(), "bases": bases, "spectra": {"combined": combined_spectrum, "random_input": random_spectrum, "stable_dct_input": stable_spectrum}, "discovery_report": discovery_report, "confirmation_report": confirmation_report}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {"schema": "mlp9_contextual_dct_range_v1_result", "terminal": terminal, "predictions": predictions, "symmetry_relative_errors": symmetry, "maximum_state_replay_relative_l2": max(state_replay), "discovery_report": discovery_report, "confirmation_report": confirmation_report, "covariance_spectrum_top16": {"combined": combined_spectrum[:16].tolist(), "random_input": random_spectrum[:16].tolist(), "stable_dct_input": stable_spectrum[:16].tolist()}, "artifact_relative": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": digest(ARTIFACT), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": {"checkpoint_loads": 1, "opened_prefixes": 96, "batched_double_jvp_calls": 96, "input_directions_per_call": 20, "suffix_gradients": 96, "symmetry_checks": 4, "covariance_eigendecompositions": 3, "random_output_controls": 16, "physical_interventions": 0, "behavioral_fits": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Behavior-blind contextual MLP9 Hessian response range fitted on one opened panel and evaluated on a disjoint opened new-endpoint panel; native z9/RMS contexts and exact suffix readers remain ports; no fresh OOD, causal extraction, removal, or sufficiency claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "symmetry_max": max(symmetry), "discovery_response": discovery_report["response_retention"], "confirmation_response": confirmation_report["response_retention"], "discovery_readers": discovery_report["reader_coverage"], "confirmation_readers": confirmation_report["reader_coverage"], "confirmation_families": confirmation_report["family_reader_coverage"]["combined"], "discovery_random_median": discovery_random_median, "confirmation_random_max": confirmation_random_max, "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
