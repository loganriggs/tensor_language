#!/usr/bin/env python3
"""Deterministically fit block-3 shared native-gate programs from sealed statistics."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import torch

_REPOSITORY_ROOT = Path("/workspace/tensor_language")
sys.path.insert(0, str(_REPOSITORY_ROOT))

import bilin18_observed_model_facade as facade
import collect_block3_native_gate_fit_v1 as collector
import native_gate_subset as subset


ROOT = collector.ROOT
HERE = collector.HERE
FIT_SOURCE = HERE / "fit_block3_native_gate_subset_v1.py"
FIT_AUTHORITY = HERE / "block3_native_gate_subset_v1_fit_authority.json"
PROGRAMS = HERE / "block3_native_gate_subset_v1_programs.pt"
RESULTS = HERE / "block3_native_gate_subset_v1_fit_results.json"
RECEIPT = HERE / "block3_native_gate_subset_v1_fit_receipt.json"
FAILURE = HERE / "block3_native_gate_subset_v1_fit_failure.json"
LOCK = Path("/workspace/runs/.block3_native_gate_subset_v1_fit.lock")
SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/fit_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/test_fit_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/test_collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/native_gate_subset.py",
    "basis_aligned/polynomial_causal/test_native_gate_subset.py",
    "basis_aligned/polynomial_causal/grouped_block_coefficient_screen.py",
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
)
BUDGETS = (256, 512)
OMP_BATCH = 16
RIDGE = 1e-6
RANDOM_SEED = 2026082907


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _collector_input_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): collector.file_sha256(path)
        for path in (collector.AUTHORITY, collector.PAYLOAD, collector.RECEIPT)
    }


def _verify_collector_inputs(expected: Mapping[str, str]) -> None:
    observed = _collector_input_hashes()
    if observed != dict(expected):
        raise RuntimeError("sealed collector inputs changed during deterministic fit")


def _source_closure() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, check=True,
    ).stdout.strip()
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if blob.returncode != 0:
            raise RuntimeError(f"fit source is not committed at frozen HEAD: {path}")
        committed_sha256 = hashlib.sha256(blob.stdout).hexdigest()
        if collector.file_sha256(ROOT / path) != committed_sha256:
            raise RuntimeError(f"live fit source differs from frozen commit: {path}")
        hashes[path] = committed_sha256
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True,
    )
    payload = {"commit": commit, "sha256s": hashes}
    return {**payload, "sha256": logical_sha256(payload)}


def _verify_source_closure(expected: Mapping[str, Any]) -> None:
    if not isinstance(expected, Mapping) or set(expected) != {
        "commit", "sha256s", "sha256",
    } or set(expected["sha256s"]) != set(SOURCE_PATHS) or logical_sha256({
        "commit": expected["commit"], "sha256s": expected["sha256s"],
    }) != expected["sha256"]:
        raise RuntimeError("fit source closure is malformed")
    for path, digest in expected["sha256s"].items():
        if collector.file_sha256(ROOT / path) != digest:
            raise RuntimeError(f"fit source changed during deterministic fit: {path}")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected["commit"], "origin/main"],
        cwd=ROOT, check=True,
    )


def _validate_payload() -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]
]:
    if not all(path.is_file() for path in (
        collector.AUTHORITY, collector.PAYLOAD, collector.RECEIPT,
    )) or any(path.exists() for path in (
        FIT_AUTHORITY, PROGRAMS, RESULTS, RECEIPT, FAILURE, LOCK,
    )):
        raise RuntimeError("sealed fit inputs are absent or fit output namespace is spent")
    input_hashes = _collector_input_hashes()
    authority = json.loads(collector.AUTHORITY.read_text())
    receipt = json.loads(collector.RECEIPT.read_text())
    if receipt.get("authority_file_sha256") != collector.file_sha256(
        collector.AUTHORITY
    ) or receipt.get("payload_file_sha256") != collector.file_sha256(
        collector.PAYLOAD
    ) or receipt.get("authority_sha256") != authority.get("authority_sha256") or (
        authority.get("status") != "frozen_before_any_block3_fit_tensor_outcome"
    ) or receipt.get("status") != (
        "fit_sufficient_statistics_complete_no_evaluation_opened"
    ):
        raise RuntimeError("fit collector authority/receipt join failed")
    payload = torch.load(collector.PAYLOAD, map_location="cpu", weights_only=True)
    expected_shapes = {
        "contribution_energy": (collector.GATES,),
        "prefilter_indices": (collector.PREFILTER,),
        "prefilter_gram": (collector.PREFILTER, collector.PREFILTER),
        "prefilter_cross": (collector.PREFILTER, collector.WIDTH),
        "prefilter_permuted_cross": (collector.PREFILTER, collector.WIDTH),
        "u_second_moment": (collector.WIDTH, collector.WIDTH),
        "v_second_moment": (collector.WIDTH, collector.WIDTH),
        "z_second_moment": (collector.WIDTH, collector.WIDTH),
    }
    if not isinstance(payload, dict) or payload.get("authority_sha256") != authority.get(
        "authority_sha256"
    ) or payload.get("count") != collector.ROW_COUNT * collector.TOKENS_PER_ROW or (
        payload.get("stacked_typed_count") != payload["count"] * len(subset.TERM_NAMES)
    ) or any(
        not torch.is_tensor(payload.get(name)) or tuple(payload[name].shape) != shape or not (
            torch.isfinite(payload[name]).all()
        ) for name, shape in expected_shapes.items()
    ) or not torch.is_tensor(payload.get("native_typed_write_energy")) or (
        payload["native_typed_write_energy"].numel() != 1
    ):
        raise RuntimeError("fit sufficient-statistic payload is malformed")
    indices = payload["prefilter_indices"]
    if indices.dtype != torch.long or len(torch.unique(indices)) != len(indices) or int(
        indices.min()
    ) < 0 or int(indices.max()) >= collector.GATES:
        raise RuntimeError("fit prefilter indices are malformed")
    _verify_collector_inputs(input_hashes)
    return authority, receipt, payload, input_hashes


def training_sse(
    gram: torch.Tensor,
    cross: torch.Tensor,
    decoder: torch.Tensor,
    write_energy: torch.Tensor | float,
) -> float:
    if gram.ndim != 2 or cross.shape != (gram.shape[0], decoder.shape[0]) or (
        decoder.shape[1] != gram.shape[0]
    ):
        raise ValueError("training objective shapes are incompatible")
    coefficient = decoder.T
    energy = torch.as_tensor(write_energy, dtype=gram.dtype, device=gram.device)
    sse = energy - 2 * (coefficient * cross).sum() + (
        coefficient * (gram @ coefficient)
    ).sum()
    tolerance = 1e-9 * max(float(energy), 1.0)
    if float(sse) < -tolerance:
        raise RuntimeError("computed ridge training SSE is materially negative")
    return max(float(sse), 0.0)


def _program_payload(program: subset.NativeGateSubsetProgram) -> dict[str, torch.Tensor]:
    return {
        "indices": program.indices.cpu().contiguous(),
        "left": program.left.cpu().contiguous(),
        "right": program.right.cpu().contiguous(),
        "decoder": program.decoder.cpu().contiguous(),
        "bias": program.bias.cpu().contiguous(),
    }


def deployed_polarization_replay(
    program: subset.NativeGateSubsetProgram,
) -> tuple[float, float]:
    """Execute the float32 K-product path and compare with its four typed banks."""

    if program.left.dtype != torch.float32:
        raise RuntimeError("deployed native-gate program is not float32")
    coordinates = torch.linspace(
        -1.0, 1.0, 3 * program.width, dtype=torch.float32,
    ).reshape(3, program.width)
    u = coordinates
    v = 0.375 * coordinates.flip(-1)
    direct = program.write(u + v)
    typed = sum(program.terms(u, v).values()) + program.bias
    absolute = float((direct - typed).abs().max())
    relative = absolute / max(float(typed.abs().max()), torch.finfo(torch.float32).tiny)
    if not torch.isfinite(direct).all() or relative > 2e-5:
        raise RuntimeError("float32 deployed K-product polarization replay failed")
    return absolute, relative


def _fit_one(
    *, left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor,
    prefilter_indices: torch.Tensor, gram: torch.Tensor, real_cross: torch.Tensor,
    fit_cross: torch.Tensor, local_indices: torch.Tensor, write_energy: torch.Tensor,
) -> tuple[subset.NativeGateSubsetProgram, dict[str, Any]]:
    selected_gram = gram[local_indices][:, local_indices]
    selected_real_cross = real_cross[local_indices]
    selected_fit_cross = fit_cross[local_indices]
    fitted_decoder = subset.fit_joint_decoder(
        selected_gram, selected_fit_cross, relative_ridge=RIDGE,
    )
    global_indices = prefilter_indices[local_indices]
    deployed_decoder = fitted_decoder.to(dtype=left.dtype)
    program = subset.build_program(
        left, right, bias, global_indices, deployed_decoder,
    )
    replay_absolute, replay_relative = deployed_polarization_replay(program)
    sse = training_sse(
        selected_gram, selected_real_cross, fitted_decoder, write_energy,
    )
    deployed_sse = training_sse(
        selected_gram, selected_real_cross,
        deployed_decoder.to(dtype=selected_gram.dtype), write_energy,
    )
    return program, {
        "gate_count": program.gates,
        "float_parameter_count": program.float_parameter_count,
        "float_byte_count": program.float_parameter_count * program.left.element_size(),
        "index_byte_count": program.indices.numel() * program.indices.element_size(),
        "total_literal_byte_count": (
            program.float_parameter_count * program.left.element_size()
            + program.indices.numel() * program.indices.element_size()
        ),
        "product_count_per_token": program.product_count_per_token,
        "linear_multiplies_per_token": 3 * program.width * program.gates,
        "deployed_dtype": str(program.left.dtype),
        "fit_write_nrmse": math.sqrt(sse / float(write_energy)),
        "deployed_float32_fit_write_nrmse": math.sqrt(
            deployed_sse / float(write_energy)
        ),
        "deployed_polarization_replay_max_abs": replay_absolute,
        "deployed_polarization_replay_relative": replay_relative,
        "selected_gate_sha256": collector.tensor_sha256(program.indices),
    }


def run() -> dict[str, Any]:
    authority, collector_receipt, payload, input_hashes = _validate_payload()
    claim = collector.acquire_claim(LOCK)
    started = time.time()
    try:
        source = _source_closure()
        checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
        if checkpoint.weights_sha256 != collector_receipt.get("checkpoint_weights_sha256"):
            raise RuntimeError("current checkpoint differs from sealed collector checkpoint")
        fit_authority_core = {
            "schema": "block3_native_gate_subset_v1_fit_authority",
            "status": "frozen_before_any_gate_selection_or_decoder_fit",
            "source_closure": source,
            "collector_authority_sha256": authority["authority_sha256"],
            "collector_input_file_sha256s": input_hashes,
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "budgets": list(BUDGETS),
            "omp_batch": OMP_BATCH,
            "ridge": RIDGE,
            "random_seed": RANDOM_SEED,
            "authorized_for_evaluation": False,
            "authorized_for_global_ledger_credit": False,
        }
        fit_authority_sha256 = logical_sha256(fit_authority_core)
        frozen_authority = {
            **fit_authority_core, "fit_authority_sha256": fit_authority_sha256,
        }
        claim.verify()
        collector.create_json(FIT_AUTHORITY, frozen_authority)
        if json.loads(FIT_AUTHORITY.read_text()) != frozen_authority:
            raise RuntimeError("published deterministic-fit authority did not replay")
        _verify_source_closure(source)
        _verify_collector_inputs(input_hashes)

        state = torch.load(
            Path(facade.DEFAULT_SNAPSHOT) / "pytorch_model.bin",
            map_location="cpu", weights_only=True, mmap=True,
        )
        prefix = f"transformer.h.{collector.LAYER}.mlp."
        left = state[prefix + "Left.weight"].detach().float()
        right = state[prefix + "Right.weight"].detach().float()
        bias = state[prefix + "Down_bias"].detach().float()
        prefilter_indices = payload["prefilter_indices"].long()
        gram = payload["prefilter_gram"].double()
        cross = payload["prefilter_cross"].double()
        permuted_cross = payload["prefilter_permuted_cross"].double()
        energy = payload["contribution_energy"].double()[prefilter_indices]
        write_energy = payload["native_typed_write_energy"].double()

        selected_max = subset.batch_simultaneous_omp(
            gram, cross, energy, budget=max(BUDGETS), prefilter=collector.PREFILTER,
            batch_size=OMP_BATCH, relative_ridge=RIDGE,
        )
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        random_order = torch.randperm(collector.PREFILTER, generator=generator)
        program_payload: dict[str, Any] = {}
        result_arms: dict[str, Any] = {}
        for budget in BUDGETS:
            arms = {
                "activation_selected": (selected_max[:budget], cross),
                "random_prefilter": (random_order[:budget], cross),
                "label_permutation": (selected_max[:budget], permuted_cross),
            }
            for family, (local, fit_cross) in arms.items():
                program, metrics = _fit_one(
                    left=left, right=right, bias=bias,
                    prefilter_indices=prefilter_indices, gram=gram,
                    real_cross=cross, fit_cross=fit_cross,
                    local_indices=local, write_energy=write_energy,
                )
                key = f"{family}_k{budget}"
                program_payload[key] = _program_payload(program)
                result_arms[key] = metrics

        programs = {
            "schema": "block3_native_gate_subset_v1_programs",
            "fit_authority_sha256": fit_authority_sha256,
            "programs": program_payload,
        }
        claim.verify()
        _verify_source_closure(source)
        _verify_collector_inputs(input_hashes)
        if json.loads(FIT_AUTHORITY.read_text()) != frozen_authority:
            raise RuntimeError("fit authority changed before program publication")
        collector.create_torch(PROGRAMS, programs)
        results = {
            **fit_authority_core,
            "fit_authority_sha256": fit_authority_sha256,
            "collector_receipt": dict(collector_receipt),
            "arms": result_arms,
            "selection_nested": True,
            "native_baseline_price": {
                "gate_count": collector.GATES,
                "float_parameter_count": 3 * collector.WIDTH * collector.GATES + collector.WIDTH,
                "float_byte_count": 4 * (
                    3 * collector.WIDTH * collector.GATES + collector.WIDTH
                ),
                "index_byte_count": 0,
                "product_count_per_token": collector.GATES,
                "linear_multiplies_per_token": 3 * collector.WIDTH * collector.GATES,
                "dtype": "torch.float32",
            },
            "evaluation_rows_loaded": 0,
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
        }
        collector.create_json(RESULTS, results)
        receipt = {
            "schema": "block3_native_gate_subset_v1_fit_receipt",
            "status": "deterministic_fit_complete_no_evaluation_opened",
            "fit_authority_sha256": fit_authority_sha256,
            "programs_file_sha256": collector.file_sha256(PROGRAMS),
            "results_file_sha256": collector.file_sha256(RESULTS),
            "collector_receipt_file_sha256": collector.file_sha256(collector.RECEIPT),
            "fit_authority_file_sha256": collector.file_sha256(FIT_AUTHORITY),
            "source_closure_sha256": source["sha256"],
            "elapsed_seconds": time.time() - started,
        }
        # Receipt is terminal. Revalidate all mutable inputs and every output it seals.
        claim.verify()
        _verify_source_closure(source)
        _verify_collector_inputs(input_hashes)
        if json.loads(FIT_AUTHORITY.read_text()) != frozen_authority or (
            collector.file_sha256(PROGRAMS) != receipt["programs_file_sha256"]
        ) or collector.file_sha256(RESULTS) != receipt["results_file_sha256"] or (
            facade.validate_snapshot(verify_weights_sha256=True) != checkpoint
        ):
            raise RuntimeError("terminal deterministic-fit integrity replay failed")
        collector.create_json(RECEIPT, receipt)
        return results
    except BaseException as error:
        if not FAILURE.exists() and not RECEIPT.exists():
            try:
                claim.verify()
                collector.create_json(FAILURE, {
                    "schema": "block3_native_gate_subset_v1_fit_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "fit_authority_exists": FIT_AUTHORITY.exists(),
                    "programs_exists": PROGRAMS.exists(),
                    "results_exists": RESULTS.exists(),
                    "receipt_exists": RECEIPT.exists(),
                    "elapsed_seconds": time.time() - started,
                })
            except BaseException:
                pass
        raise
    finally:
        claim.release()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
