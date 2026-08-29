#!/usr/bin/env python3
"""Outcome-blind contract and preflight for the native-Down behavioral port.

The pure functions here freeze support controls, fit-PCA gauge, simultaneous document
bootstrap, and the complete physical call plan.  Numerical execution remains fail-
closed until the separately audited fresh-row freezer and model adapter are added to
SOURCE_PATHS; ``run`` cannot create authority in this bounded slice.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import torch


ROOT = Path("/workspace/tensor_language")
HERE = ROOT / "basis_aligned" / "polynomial_causal"
PREREG = HERE / "BLOCK3_NATIVE_DOWN_BEHAVIORAL_PORT_V1_PREREGISTRATION.md"
ADDENDUM = HERE / "BLOCK3_NATIVE_DOWN_BEHAVIORAL_PORT_V1_EXECUTION_ADDENDUM.md"
RUNNER = HERE / "run_block3_native_down_behavioral_port_v1.py"
TEST = HERE / "test_run_block3_native_down_behavioral_port_v1.py"

AUTHORITY = HERE / "block3_native_down_behavioral_port_v1_authority.json"
DIRECTIONS = HERE / "block3_native_down_behavioral_port_v1_directions.pt"
DIRECTIONS_RECEIPT = HERE / "block3_native_down_behavioral_port_v1_directions_receipt.json"
ATTEMPT = HERE / "block3_native_down_behavioral_port_v1_final_attempt.json"
RESULTS = HERE / "block3_native_down_behavioral_port_v1_results.json"
MANIFEST = HERE / "block3_native_down_behavioral_port_v1_manifest.json"
RECEIPT = HERE / "block3_native_down_behavioral_port_v1_receipt.json"
FAILURE = HERE / "block3_native_down_behavioral_port_v1_failure.json"
LOCK = Path("/workspace/runs/.block3_native_down_behavioral_port_v1.lock")

FAMILY_AUTHORITY = HERE / "block3_consequence_family_f_v2_recovery_authority.json"
FAMILY_RESULTS = HERE / "block3_consequence_family_f_v2_recovery_results.json"
FAMILY_RECEIPT = HERE / "block3_consequence_family_f_v2_recovery_receipt.json"
FAMILY_PROGRAMS = HERE / "block3_consequence_family_f_v1_programs.pt"
PARENT_PINS = {
    str(FAMILY_AUTHORITY.relative_to(ROOT)): "ca759bebef883093ec2312249baf36c5e5e2175effc3e9af88b57d2471913dbd",
    str(FAMILY_RESULTS.relative_to(ROOT)): "18b03ccf3d6710813375bb7e09b1a3c313d5e7790e2ca3c9a9b683fbf91897c5",
    str(FAMILY_RECEIPT.relative_to(ROOT)): "e81673095c7b6202fdec293c6ad34924fb9acb15213d02ba4b203d5ff8c65a5a",
    str(FAMILY_PROGRAMS.relative_to(ROOT)): "d4af5bfbae03f8df9be8127e2e06c6f1a66b189be180ce72e5c74b6c7ac7a038",
}

WIDTH = 1152
N_GATES = 4608
BUDGET = 512
RANDOM_SEED = 2026082907
BOOTSTRAP_SEED = 2026082911
BOOTSTRAP_DRAWS = 2000
FRESH_DOCUMENTS = 192
BATCH_SIZE = 4
FIT_ROWS = 480
FIT_BATCHES = FIT_ROWS // BATCH_SIZE
FRESH_BATCHES = FRESH_DOCUMENTS // BATCH_SIZE
AMPLITUDES = (0.5, 1.0)
SIGNS = (-1, 1)
DIRECTION_COUNT = 4
SOURCE_PATHS = tuple(str(path.relative_to(ROOT)) for path in (
    PREREG, ADDENDUM, RUNNER, TEST,
    HERE / "recover_block3_consequence_family_f_v2.py",
    HERE / "fit_block3_consequence_family_f_v1.py",
    HERE / "block3_consequence_fit.py",
    HERE / "native_gate_subset.py",
    HERE / "collect_block3_native_gate_fit_v1.py",
    HERE / "bilin18_observed_model_facade.py",
    ROOT / "jacclust" / "tt_model.py", ROOT / "jacclust" / "__init__.py",
))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def random_support() -> torch.Tensor:
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    return torch.randperm(N_GATES, generator=generator)[:BUDGET].sort().values


def shifted_decoder_indices(candidate_support: torch.Tensor) -> torch.Tensor:
    validate_support(candidate_support)
    return torch.roll(candidate_support, -1).contiguous()


def validate_support(value: torch.Tensor) -> None:
    if value.dtype != torch.long or value.device.type != "cpu" or value.shape != (
        BUDGET,
    ) or len(torch.unique(value)) != BUDGET or int(value.min()) < 0 or int(
        value.max()
    ) >= N_GATES:
        raise ValueError("behavioral-port support is malformed")


@dataclass(frozen=True)
class DirectionReceipt:
    observations: int
    width: int
    fit_error_rms: float
    mean_sha256: str
    directions_sha256: str
    eigenvalues_sha256: str
    top5_eigenvalues: tuple[float, ...]
    adjacent_top5_gaps: tuple[float, ...]
    unit_rms_max_abs_error: float


def fit_error_directions(errors: torch.Tensor) -> tuple[torch.Tensor, DirectionReceipt]:
    if errors.dtype != torch.float64 or errors.device.type != "cpu" or errors.ndim != 2 or (
        errors.shape[1] != WIDTH
    ) or errors.shape[0] < 5 or not bool(torch.isfinite(errors).all()):
        raise ValueError("fit error matrix must be finite CPU float64 [n,1152]")
    mean = errors.mean(0)
    centered = errors - mean
    covariance = centered.T @ centered / errors.shape[0]
    values, vectors = torch.linalg.eigh(0.5 * (covariance + covariance.T))
    values, vectors = values.flip(0).contiguous(), vectors.flip(1).contiguous()
    scale = max(float(values[0]), 1.0)
    gaps = values[:4] - values[1:5]
    if bool((gaps <= 1e-10 * scale).any()):
        raise RuntimeError("fit-error top-four eigengap is unresolved")
    directions = vectors[:, :DIRECTION_COUNT].T.contiguous()
    for index in range(DIRECTION_COUNT):
        coordinate = int(torch.argmax(directions[index].abs()))
        if float(directions[index, coordinate]) < 0:
            directions[index].neg_()
        directions[index].mul_(math.sqrt(WIDTH) / torch.linalg.vector_norm(directions[index]))
    rms = directions.square().mean(1).sqrt()
    error_rms = float(errors.square().mean().sqrt())
    receipt = DirectionReceipt(
        observations=errors.shape[0], width=WIDTH, fit_error_rms=error_rms,
        mean_sha256=tensor_sha256(mean), directions_sha256=tensor_sha256(directions),
        eigenvalues_sha256=tensor_sha256(values),
        top5_eigenvalues=tuple(float(value) for value in values[:5]),
        adjacent_top5_gaps=tuple(float(value) for value in gaps),
        unit_rms_max_abs_error=float((rms - 1).abs().max()),
    )
    if not math.isfinite(error_rms) or error_rms <= 0 or receipt.unit_rms_max_abs_error > 1e-12:
        raise RuntimeError("fit-error direction scaling failed")
    return directions, receipt


def edit_cells() -> tuple[tuple[int, int, float], ...]:
    return tuple(
        (direction, sign, amplitude)
        for direction in range(DIRECTION_COUNT)
        for amplitude in AMPLITUDES for sign in SIGNS
    )


def error_secant_cells() -> tuple[tuple[int, float], ...]:
    return tuple((sign, amplitude) for amplitude in AMPLITUDES for sign in SIGNS)


def bootstrap_weights(documents: int = FRESH_DOCUMENTS) -> torch.Tensor:
    if type(documents) is not int or documents <= 1:
        raise ValueError("bootstrap document count must exceed one")
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    samples = torch.randint(documents, (BOOTSTRAP_DRAWS, documents), generator=generator)
    counts = torch.zeros(BOOTSTRAP_DRAWS, documents, dtype=torch.float64)
    counts.scatter_add_(1, samples, torch.ones_like(samples, dtype=torch.float64))
    return torch.cat((torch.ones(1, documents, dtype=torch.float64), counts), 0)


def ratio_series(
    numerators: torch.Tensor, denominators: torch.Tensor, weights: torch.Tensor,
) -> torch.Tensor:
    if numerators.ndim != 2 or denominators.shape != numerators.shape or weights.ndim != 2 or (
        weights.shape[1] != numerators.shape[1]
    ) or any(value.dtype != torch.float64 for value in (numerators, denominators, weights)):
        raise ValueError("document ratio series has invalid schema")
    denominator = weights @ denominators.T
    if not bool((denominator > 0).all()):
        raise RuntimeError("document ratio denominator is nonpositive")
    output = (weights @ numerators.T) / denominator
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError("document ratio series is nonfinite")
    return output


def simultaneous_bounds(series: torch.Tensor) -> dict[str, list[float]]:
    if series.dtype != torch.float64 or series.ndim != 2 or series.shape[0] != (
        BOOTSTRAP_DRAWS + 1
    ) or not bool(torch.isfinite(series).all()):
        raise ValueError("simultaneous series must be [point+2000,cells] float64")
    point, draws = series[0], series[1:]
    upper_radius = torch.quantile((draws - point).max(1).values, 0.95)
    lower_radius = torch.quantile((point - draws).max(1).values, 0.95)
    return {
        "point": point.tolist(),
        "simultaneous_q95_upper": (point + upper_radius).tolist(),
        "simultaneous_q05_lower": (point - lower_radius).tolist(),
    }


def centered_logits(value: torch.Tensor) -> torch.Tensor:
    if not value.is_floating_point() or value.ndim < 2 or not bool(torch.isfinite(value).all()):
        raise ValueError("logits are malformed")
    return value - value.mean(-1, keepdim=True)


def expected_call_plan() -> dict[str, Any]:
    student_per_batch = 5 + 3 + 2 * len(edit_cells())
    total_suffix_per_batch = 1 + student_per_batch
    return {
        "fit_prefixes": FIT_BATCHES,
        "fresh_prefixes": FRESH_BATCHES,
        "ordinary_full_model_replays": FRESH_BATCHES,
        "teacher_suffixes": FRESH_BATCHES,
        "student_suffixes_per_batch": student_per_batch,
        "student_suffixes": FRESH_BATCHES * student_per_batch,
        "all_suffixes": FRESH_BATCHES * total_suffix_per_batch,
        "physical_calls_sites_0_3_per_kind": FIT_BATCHES + FRESH_BATCHES + FRESH_BATCHES,
        "physical_calls_sites_4_17_per_kind": FRESH_BATCHES * total_suffix_per_batch + FRESH_BATCHES,
        "native_mlp3_calls": FIT_BATCHES + FRESH_BATCHES + FRESH_BATCHES,
        "compiled_student_native_mlp3_calls": 0,
        "error_candidate_plus_one_reused": True,
        "optimizer_calls": 0,
        "backward_calls": 0,
    }


def verify_parent_files() -> dict[str, str]:
    observed = {relative: file_sha256(ROOT / relative) for relative in PARENT_PINS}
    if observed != PARENT_PINS:
        raise RuntimeError("behavioral-port Family-F parent bytes changed")
    receipt = json.loads(FAMILY_RECEIPT.read_text())
    if receipt.get("authority_file_sha256") != PARENT_PINS[str(
        FAMILY_AUTHORITY.relative_to(ROOT)
    )] or receipt.get("results_file_sha256") != PARENT_PINS[str(
        FAMILY_RESULTS.relative_to(ROOT)
    )] or receipt.get("v1_programs_file_sha256") != PARENT_PINS[str(
        FAMILY_PROGRAMS.relative_to(ROOT)
    )] or receipt.get("authorized_for_validation") is not False:
        raise RuntimeError("behavioral-port Family-F parent joins changed")
    return observed


def source_closure() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    hashes = {}
    for relative in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if completed.returncode:
            raise RuntimeError(f"behavioral-port source is not committed: {relative}")
        digest = hashlib.sha256(completed.stdout).hexdigest()
        if file_sha256(ROOT / relative) != digest:
            raise RuntimeError(f"behavioral-port source differs from commit: {relative}")
        hashes[relative] = digest
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"],
                   cwd=ROOT, check=True)
    body = {"commit": commit, "paths": hashes}
    return {**body, "sha256": logical_sha256(body)}


def output_namespace() -> tuple[Path, ...]:
    return AUTHORITY, DIRECTIONS, DIRECTIONS_RECEIPT, ATTEMPT, RESULTS, MANIFEST, RECEIPT, FAILURE, LOCK


EXECUTION_BLOCKERS = (
    "canonical fresh-row freezer/receipt is not implemented in this bounded slice",
    "CUDA measurement adapter/result semantic validator is not implemented in this bounded slice",
)


def run() -> None:
    spent = [str(path) for path in output_namespace() if path.exists()]
    if spent:
        raise RuntimeError(f"behavioral-port namespace is spent: {spent}")
    verify_parent_files()
    source_closure()
    raise RuntimeError("behavioral-port execution remains NO-GO: " + "; ".join(EXECUTION_BLOCKERS))


if __name__ == "__main__":
    run()
