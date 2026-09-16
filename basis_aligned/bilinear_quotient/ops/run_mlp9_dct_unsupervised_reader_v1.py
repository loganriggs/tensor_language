#!/usr/bin/env python3
# BQGATE:48prefixes;2implicit rank8 DCT fits;8nested JVP checks;240seconds;opened reader diagnostic.
"""Full-width weight-only MLP9 DCT factors versus the CrossFirst reader."""
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

STEM = "MLP9_DCT_UNSUPERVISED_READER_V1"
BINDING = P / f"{STEM}_BINDING.json"
OUT = P / f"{STEM}_RESULT.json"
ARTIFACT = P / f"{STEM}_ARTIFACT.pt"
ROWS = P / "CROSSFIRST_HESSIAN_TOP2_FRESH_V1_ROWS.json"
RANK = 8
SEEDS = (2026091601, 2026091602)
RESTARTS = 8
SWEEPS = 40
PROBES = 64


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
    return rows


def plan():
    rows = load_bound()
    return {"schema": "mlp9_dct_unsupervised_reader_v1_plan", "rows": len(rows), "rank": RANK, "fits": 2, "restarts": RESTARTS, "sweeps": SWEEPS, "tensor_probes": PROBES, "random_controls": 16, "model_loaded": False, "gpu_accessed": False, "queue_touched": False}


def tensor_vv(left, right, down, vector):
    return 2 * (down @ ((left @ vector) * (right @ vector)))


def tensor_uvw(left, right, down, output, first, second):
    hidden_reader = down.T @ output
    return torch.sum(hidden_reader * ((left @ first) * (right @ second) + (left @ second) * (right @ first)))


def matrix_apply(left, right, down, output, vectors):
    hidden_reader = (down.T @ output)[:, None]
    return left.T @ (hidden_reader * (right @ vectors)) + right.T @ (hidden_reader * (left @ vectors))


def canonical_rows(factors):
    factors = factors.clone()
    for index in range(factors.shape[0]):
        coordinate = int(factors[index].abs().argmax())
        if factors[index, coordinate] < 0:
            factors[index] *= -1
    return factors


@torch.no_grad()
def fit_implicit(left, right, down, seed):
    generator = torch.Generator(device=left.device).manual_seed(seed)
    input_factors = []
    output_factors = []
    amplitudes = []
    for component in range(RANK):
        vectors = torch.randn(1152, RESTARTS, generator=generator, device=left.device, dtype=torch.float64)
        if input_factors:
            basis = torch.stack(input_factors, dim=1)
            vectors -= basis @ (basis.T @ vectors)
        vectors /= vectors.norm(dim=0, keepdim=True).clamp_min(1e-30)
        for _ in range(SWEEPS):
            responses = torch.stack([tensor_vv(left, right, down, vectors[:, restart]) for restart in range(RESTARTS)], dim=1)
            if input_factors:
                prior_input = torch.stack(input_factors)
                prior_output = torch.stack(output_factors)
                prior_amplitude = torch.stack(amplitudes)
                overlaps = prior_input @ vectors
                responses -= prior_output.T @ (prior_amplitude[:, None] * overlaps.square())
            outputs = responses / responses.norm(dim=0, keepdim=True).clamp_min(1e-30)
            updated = torch.empty_like(vectors)
            for restart in range(RESTARTS):
                candidate = matrix_apply(left, right, down, outputs[:, restart], vectors[:, restart:restart + 1])[:, 0]
                if input_factors:
                    candidate -= prior_input.T @ (prior_amplitude * (prior_output @ outputs[:, restart]) * (prior_input @ vectors[:, restart]))
                    candidate -= basis @ (basis.T @ candidate)
                updated[:, restart] = candidate / candidate.norm().clamp_min(1e-30)
            vectors = updated
        scores = []
        for restart in range(RESTARTS):
            response = tensor_vv(left, right, down, vectors[:, restart])
            if input_factors:
                overlap = prior_input @ vectors[:, restart]
                response -= prior_output.T @ (prior_amplitude * overlap.square())
            scores.append(response.norm())
        best = int(torch.stack(scores).argmax())
        vector = vectors[:, best]
        response = tensor_vv(left, right, down, vector)
        if input_factors:
            overlap = prior_input @ vector
            response -= prior_output.T @ (prior_amplitude * overlap.square())
        amplitude = response.norm()
        output = response / amplitude.clamp_min(1e-30)
        input_factors.append(vector)
        output_factors.append(output)
        amplitudes.append(amplitude)
    raw_inputs = torch.stack(input_factors)
    inputs = canonical_rows(raw_inputs)
    outputs = torch.stack(output_factors)
    # A symmetric input factor enters as v⊗v, so v and -v are identical.
    # Canonicalizing v must therefore not change the paired output factor.
    return {"input": inputs.cpu(), "output": outputs.cpu(), "amplitude": torch.stack(amplitudes).cpu()}


@torch.no_grad()
def probe_residual(left, right, down, fit, probes):
    actual = torch.stack([tensor_vv(left, right, down, vector) for vector in probes])
    inputs = fit["input"].to(left.device)
    outputs = fit["output"].to(left.device)
    amplitudes = fit["amplitude"].to(left.device)
    predicted = (probes @ inputs.T).square() @ (amplitudes[:, None] * outputs)
    return ratio(actual - predicted, actual)


def orthonormal_span(rows):
    return torch.linalg.qr(rows.T.double(), mode="reduced").Q


def principal_cosines(first, second):
    return torch.linalg.svdvals(orthonormal_span(first).T @ orthonormal_span(second))


def coverage(reader, basis):
    flattened = reader.reshape(-1, reader.shape[-1]).double()
    return ratio(flattened @ basis @ basis.T, flattened)


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
    mlp = model.transformer.h[9].mlp
    left = mlp.Left.weight.detach().double()
    right = mlp.Right.weight.detach().double()
    down = mlp.Down.weight.detach().double()
    started = time.perf_counter()

    generator = torch.Generator(device="cuda").manual_seed(2026091603)
    contraction_errors = []
    for _ in range(8):
        point = torch.randn(1152, generator=generator, device="cuda", dtype=torch.float64)
        output = torch.randn(1152, generator=generator, device="cuda", dtype=torch.float64)
        first = torch.randn(1152, generator=generator, device="cuda", dtype=torch.float64)
        second = torch.randn(1152, generator=generator, device="cuda", dtype=torch.float64)
        function = lambda value: output @ (down @ ((left @ value) * (right @ value)) + mlp.Down_bias.double())
        observed = torch.func.jvp(lambda value: torch.func.jvp(function, (value,), (first,))[1], (point,), (second,))[1]
        expected = tensor_uvw(left, right, down, output, first, second)
        contraction_errors.append(float((observed - expected).abs() / expected.abs().clamp_min(1e-30)))

    fits = [fit_implicit(left, right, down, seed) for seed in SEEDS]
    probe_generator = torch.Generator(device="cuda").manual_seed(2026091604)
    probes = torch.randn(PROBES, 1152, generator=probe_generator, device="cuda", dtype=torch.float64)
    probes /= probes.norm(dim=1, keepdim=True)
    residuals = [probe_residual(left, right, down, fit, probes) for fit in fits]
    input_cosines = principal_cosines(fits[0]["input"], fits[1]["input"])
    output_cosines = principal_cosines(fits[0]["output"], fits[1]["output"])
    orthogonality = [float((fit["input"] @ fit["input"].T - torch.eye(RANK)).abs().max()) for fit in fits]
    output_basis = orthonormal_span(fits[0]["output"]).cuda()

    readers = []
    state_replay = []
    for row in rows:
        ids = torch.tensor([row["ids"]], device="cuda")
        native = live.prepare(model, ids, weights)
        _, stages = allocation.make_stages(model, native)
        readout = allocation.make_readout(model, row)
        h9 = stages[0](native["z9"]).detach().requires_grad_(True)
        state_replay.append(float((h9.detach() - native["h9"]).float().norm() / native["h9"].float().norm().clamp_min(1e-30)))
        value = h9
        for stage in stages[1:]:
            value = stage(value)
        contrast = readout(value)[0] - readout(value)[1]
        readers.append(torch.autograd.grad(contrast, h9)[0].detach().double().cpu())
    prompt_coverage = [coverage(reader, output_basis.cpu()) for reader in readers]
    pooled_reader = torch.cat(readers, dim=1)
    pooled_coverage = coverage(pooled_reader, output_basis.cpu())
    family_coverage = []
    for family in range(4):
        family_reader = torch.cat([readers[index] for index, row in enumerate(rows) if row["family"] == family], dim=1)
        family_coverage.append(coverage(family_reader, output_basis.cpu()))

    controls = []
    control_generator = torch.Generator().manual_seed(2026091605)
    for _ in range(16):
        basis, _ = torch.linalg.qr(torch.randn(1152, RANK, generator=control_generator, dtype=torch.float64), mode="reduced")
        controls.append(coverage(pooled_reader, basis))
    control_median = float(torch.tensor(controls, dtype=torch.float64).median())

    pred_a = bool(max(contraction_errors) <= 2e-5 and max(state_replay) <= 2e-6)
    pred_b = bool(pred_a and max(orthogonality) <= 2e-4 and all(value <= .98 for value in residuals))
    pred_c = bool(pred_a and min(input_cosines) >= .80 and min(output_cosines) >= .80)
    pred_d = bool(pred_a and pooled_coverage >= .50 and min(family_coverage) >= .35 and float(torch.tensor(prompt_coverage).median()) >= .40)
    pred_e = bool(pred_a and pooled_coverage >= control_median + .15 and pooled_coverage > max(controls))
    pred_f = bool(len(readers) == 48 and all(fit["input"].shape == (RANK, 1152) and fit["output"].shape == (RANK, 1152) for fit in fits))
    predictions = {"pred_a_tensor_instrument": pred_a, "pred_b_solver_health": pred_b, "pred_c_identifiability": pred_c, "pred_d_reader_coverage": pred_d, "pred_e_random_subspace_specificity": pred_e, "pred_f_complete_audit": pred_f}
    terminal = "mlp9_dct_unsupervised_reader_candidate" if all(predictions.values()) else "valid_mlp9_dct_unsupervised_reader_null" if pred_a else "invalid"

    artifact = {"schema": "mlp9_dct_unsupervised_reader_v1_artifact", "fits": fits, "readers": readers, "prompt_coverage": torch.tensor(prompt_coverage), "family_coverage": torch.tensor(family_coverage), "random_control_coverage": torch.tensor(controls), "probe_residual_relative_l2": torch.tensor(residuals), "input_principal_cosines": input_cosines, "output_principal_cosines": output_cosines}
    buffer = io.BytesIO()
    torch.save(artifact, buffer)
    atomic_bytes(ARTIFACT, buffer.getvalue())
    result = {"schema": "mlp9_dct_unsupervised_reader_v1_result", "terminal": terminal, "predictions": predictions, "maximum_contraction_relative_error": max(contraction_errors), "contraction_relative_errors": contraction_errors, "probe_residual_relative_l2": residuals, "input_orthogonality_maximum_absolute_error": orthogonality, "input_principal_cosines": input_cosines.tolist(), "output_principal_cosines": output_cosines.tolist(), "pooled_reader_coverage": pooled_coverage, "family_reader_coverage": family_coverage, "prompt_reader_coverage_median": float(torch.tensor(prompt_coverage).median()), "prompt_reader_coverage_minimum": min(prompt_coverage), "random_control_coverage": controls, "random_control_median": control_median, "maximum_state_replay_relative_l2": max(state_replay), "artifact_relative": str(ARTIFACT.relative_to(ROOT)), "artifact_sha256": digest(ARTIFACT), "runner_sha256": digest(RUNNER), "binding_sha256": digest(BINDING), "checkpoint_weights_sha256": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3", "seconds": time.perf_counter() - started, "price": {"checkpoint_loads": 1, "weight_only_rank8_fits": 2, "factor_restarts": RESTARTS, "factor_sweeps": SWEEPS, "tensor_probes": PROBES, "nested_jvp_checks": 8, "opened_prefixes": 48, "downstream_gradients": 48, "random_rank8_controls": 16, "physical_interventions": 0, "behavioral_fits": 0, "parameter_updates": 0}, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"), "scope": "Full-width weight-only raw-MLP9 symmetric-Hessian factor recovery followed by opened-panel CrossFirst reader-span diagnostic; no prompt enters factor recovery and no OOD, extraction, removal, or sufficiency claim."}
    atomic_json(OUT, result)
    print(json.dumps({"terminal": terminal, "predictions": predictions, "residuals": residuals, "input_min_cosine": float(input_cosines.min()), "output_min_cosine": float(output_cosines.min()), "pooled_coverage": pooled_coverage, "family_coverage": family_coverage, "prompt_median": result["prompt_reader_coverage_median"], "random_median": control_median, "random_maximum": max(controls), "seconds": result["seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
