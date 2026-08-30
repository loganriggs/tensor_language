#!/usr/bin/env python3
"""Create-only synthetic production-shape CUDA timing for factorization v1.

This executable imports only the pure factorizer. It never opens the FIT bundle,
training input, validation role, model, corpus, or EVAL.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch

from causal_response_factorization_v1 import make_program_from_factors, predict_from_codes
from causal_response_factorization_v1_accelerated import (
    fit_shared_private_program_accelerated,
)


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = HERE / "causal_response_factorization_v1_cuda_benchmark_receipt.json"
SOURCE = Path(__file__).resolve()
OPTIMIZER_SOURCE = HERE / "causal_response_factorization_v1_accelerated.py"
GROUP_SIZES = (16, 13, 6, 5, 5, 4)
BENCHMARK_STEPS = 50


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def planted_gate() -> dict[str, float | bool]:
    generator = torch.Generator().manual_seed(88)
    groups = torch.tensor([0, 0, 1, 1], dtype=torch.int64)
    program = make_program_from_factors(
        tuple(
            torch.randn(shape, generator=generator, dtype=torch.float64)
            for shape in ((2, 1), (4, 1), (3, 1))
        ),
        tuple(
            tuple(
                torch.randn(shape, generator=generator, dtype=torch.float64)
                for shape in ((2, 1), (2, 1), (3, 1))
            )
            for _ in range(2)
        ),
        groups,
    )
    codes = torch.randn(
        (12, program.code_dimension), generator=generator, dtype=torch.float64
    )
    response = predict_from_codes(program.basis(), codes).reshape(2, 4, 3, 12)
    fitted = fit_shared_private_program_accelerated(
        response,
        torch.ones_like(response, dtype=torch.bool),
        groups,
        global_rank=1,
        private_rank=1,
        seed=2026083001,
        steps=2_000,
        learning_rate=0.04,
        optimizer_device="cuda",
    )
    passed = fitted.improvement_fraction > 0.9999 and fitted.final_mse < 1e-8
    if not passed:
        raise RuntimeError("CUDA planted recovery gate failed")
    return {
        "improvement_fraction": fitted.improvement_fraction,
        "canonical_cpu_float64_final_mse": fitted.final_mse,
        "passes": passed,
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("CUDA benchmark receipt namespace is already spent")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    groups = torch.cat(
        [torch.full((size,), group, dtype=torch.int64) for group, size in enumerate(GROUP_SIZES)]
    ).contiguous()
    if groups.numel() != 49:
        raise RuntimeError("synthetic owner groups do not have production size")
    generator = torch.Generator().manual_seed(2026083001)
    response = torch.randn(
        (2, 49, 49, 229), generator=generator, dtype=torch.float64
    ).contiguous()
    valid = torch.ones_like(response, dtype=torch.bool)

    planted = planted_gate()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    fitted = fit_shared_private_program_accelerated(
        response,
        valid,
        groups,
        global_rank=8,
        private_rank=2,
        seed=2026083001,
        steps=BENCHMARK_STEPS,
        learning_rate=0.03,
        optimizer_device="cuda",
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    device = torch.cuda.get_device_properties(0)
    receipt = {
        "schema": "causal_response_factorization_v1_cuda_benchmark_receipt",
        "status": "complete_synthetic_only",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": file_sha256(SOURCE),
        "optimizer_source_sha256": file_sha256(OPTIMIZER_SOURCE),
        "outcome_access": {
            "fit_bundle_opened": False,
            "training_input_opened": False,
            "validation_opened": False,
            "eval_opened": False,
            "model_opened": False,
        },
        "device": {
            "name": device.name,
            "total_memory_bytes": device.total_memory,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "production_shape_synthetic": {
            "shape": list(response.shape),
            "response_dtype": str(response.dtype),
            "valid_fraction": 1.0,
            "owner_group_sizes": list(GROUP_SIZES),
            "global_rank": 8,
            "private_rank_each_owner": 2,
            "steps": BENCHMARK_STEPS,
            "learning_rate": 0.03,
            "seed": 2026083001,
            "elapsed_seconds": elapsed,
            "seconds_per_step_including_setup_and_cpu_replay": elapsed / BENCHMARK_STEPS,
            "linear_2000_step_seconds_estimate": elapsed / BENCHMARK_STEPS * 2_000,
            "peak_cuda_memory_bytes": peak,
            "initial_mse": fitted.initial_mse,
            "canonical_cpu_float64_final_mse": fitted.final_mse,
            "improvement_fraction": fitted.improvement_fraction,
            "finite": all(
                torch.isfinite(torch.tensor(value))
                for value in (fitted.initial_mse, fitted.final_mse, fitted.improvement_fraction)
            ),
        },
        "planted_cuda_gate": planted,
    }
    raw = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with OUTPUT.open("x") as sink:
        sink.write(raw)
        sink.flush()
    replay = json.loads(OUTPUT.read_bytes())
    if replay != receipt:
        raise RuntimeError("CUDA benchmark receipt did not replay")
    print(raw, end="")


if __name__ == "__main__":
    main()
