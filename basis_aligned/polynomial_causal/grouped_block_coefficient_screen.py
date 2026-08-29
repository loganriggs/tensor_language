"""Gauge-balanced coefficient screen for an attention--bilinear-MLP supernode."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Mapping

import torch


ROOT = Path("/workspace/tensor_language")
HERE = ROOT / "basis_aligned" / "polynomial_causal"
DEFAULT_CHECKPOINT = Path(
    "/workspace/.hf_home/hub/"
    "models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/"
    "snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin"
)
DEFAULT_OUTPUT = HERE / "grouped_block_coefficient_screen_results.json"
PREREGISTRATION = HERE / "GROUPED_BLOCK_COEFFICIENT_SCREEN_V1_PREREGISTRATION.md"


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def balance_product_gauge(
    left: torch.Tensor, right: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if left.ndim != 2 or right.shape != left.shape or not (
        left.is_floating_point() and right.is_floating_point()
    ):
        raise ValueError("left/right product factors are incompatible")
    tiny = torch.finfo(left.dtype).tiny
    left_norm = torch.linalg.vector_norm(left, dim=1)
    right_norm = torch.linalg.vector_norm(right, dim=1)
    if bool((left_norm <= tiny).any() or (right_norm <= tiny).any()):
        raise ValueError("zero product-factor row has no unique positive balance")
    scale = torch.sqrt(right_norm / left_norm)
    return scale[:, None] * left, right / scale[:, None], scale


def weighted_interface(
    c_proj: torch.Tensor, left: torch.Tensor, right: torch.Tensor,
    down: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    if c_proj.ndim != 2 or c_proj.shape[0] != c_proj.shape[1] or (
        left.shape != right.shape
    ) or left.shape[1] != c_proj.shape[0] or down.shape != (
        c_proj.shape[0], left.shape[0]
    ):
        raise ValueError("supernode factor shapes are incompatible")
    balanced_left, balanced_right, scale = balance_product_gauge(left, right)
    weight = torch.linalg.vector_norm(down, dim=0)
    native_weighted_norm_sq = (
        (weight[:, None] * left).square().sum()
        + (weight[:, None] * right).square().sum()
    )
    balanced_weighted_norm_sq = (
        (weight[:, None] * balanced_left).square().sum()
        + (weight[:, None] * balanced_right).square().sum()
    )
    left_map = (weight[:, None] * balanced_left) @ c_proj
    right_map = (weight[:, None] * balanced_right) @ c_proj
    interface = torch.cat((left_map, right_map), dim=0)
    left_balanced_norm = torch.linalg.vector_norm(balanced_left, dim=1)
    right_balanced_norm = torch.linalg.vector_norm(balanced_right, dim=1)
    mismatch = (
        (left_balanced_norm - right_balanced_norm).abs()
        / torch.maximum(left_balanced_norm, right_balanced_norm)
    ).max()
    metadata = {
        "max_balanced_row_norm_relative_mismatch": float(mismatch),
        "balanced_to_native_weighted_factor_norm": float(
            torch.sqrt(balanced_weighted_norm_sq / native_weighted_norm_sq)
        ),
        "gauge_scale_min": float(scale.min()),
        "gauge_scale_median": float(scale.median()),
        "gauge_scale_max": float(scale.max()),
    }
    return interface, metadata


def spectrum_summary(interface: torch.Tensor) -> dict[str, object]:
    if interface.ndim != 2 or not interface.is_floating_point():
        raise ValueError("interface must be a floating matrix")
    gram = interface.T @ interface
    eigenvalues = torch.linalg.eigvalsh(gram.double()).clamp_min_(0).flip(0)
    total = eigenvalues.sum()
    if not bool(torch.isfinite(total)) or float(total) <= 0:
        raise ValueError("interface has no finite positive energy")
    cumulative = eigenvalues.cumsum(0) / total

    def energy_rank(threshold: float) -> int:
        return int(torch.searchsorted(cumulative, threshold).item() + 1)

    def relative_error(rank: int) -> float:
        kept = cumulative[min(rank, len(cumulative)) - 1]
        return float(torch.sqrt((1.0 - kept).clamp_min(0)))

    return {
        "singular_value_squared": [float(value) for value in eigenvalues],
        "frobenius_norm": float(torch.sqrt(total)),
        "spectral_norm": float(torch.sqrt(eigenvalues[0])),
        "stable_rank": float(total / eigenvalues[0]),
        "energy_ranks": {
            str(threshold): energy_rank(threshold)
            for threshold in (0.9, 0.95, 0.99, 0.999)
        },
        "relative_frobenius_error": {
            str(rank): relative_error(rank) for rank in (64, 128, 256, 512)
        },
    }


def block_factors(
    state: Mapping[str, torch.Tensor], layer: int,
) -> dict[str, torch.Tensor]:
    prefix = f"transformer.h.{layer}."
    names = {
        "c_proj": prefix + "attn.c_proj.weight",
        "left": prefix + "mlp.Left.weight",
        "right": prefix + "mlp.Right.weight",
        "down": prefix + "mlp.Down.weight",
    }
    missing = [name for name in names.values() if name not in state]
    if missing:
        raise KeyError(f"checkpoint lacks factors: {missing}")
    return {name: state[key].detach().float().cpu() for name, key in names.items()}


def run(checkpoint: Path, output: Path, *, layer: int) -> dict[str, object]:
    if not PREREGISTRATION.is_file() or not checkpoint.is_file() or output.exists():
        raise RuntimeError("preregistration/checkpoint missing or output namespace spent")
    started = time.time()
    resolved = checkpoint.resolve()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    factors = block_factors(state, layer)
    interface, balance = weighted_interface(**factors)
    spectrum = spectrum_summary(interface)
    promising = int(spectrum["energy_ranks"]["0.95"]) <= 256
    result = {
        "schema": "grouped_block_coefficient_screen_v1",
        "status": "descriptive_coefficient_screen_no_ledger_credit",
        "layer": layer,
        "decision": {
            "criterion": "95pct_energy_rank_le_256",
            "promising": promising,
            "next_if_pass": "extend frozen screen to blocks 4-8 and collect typed vector responses",
            "next_if_fail": "prune raw coefficient HOSVD; collect activation/consequence-weighted typed responses",
        },
        "balance": balance,
        "spectrum": spectrum,
        "integrity": {
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "checkpoint_requested": str(checkpoint),
            "checkpoint_resolved": str(resolved),
            "checkpoint_blob_name": resolved.name,
            "factor_sha256": {
                name: tensor_sha256(value) for name, value in factors.items()
            },
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "pid": os.getpid(),
        },
        "elapsed_seconds": time.time() - started,
    }
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(temporary, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    result = run(args.checkpoint, args.output, layer=args.layer)
    print(json.dumps({
        "layer": result["layer"],
        "decision": result["decision"],
        "balance": result["balance"],
        "energy_ranks": result["spectrum"]["energy_ranks"],
        "relative_frobenius_error": result["spectrum"]["relative_frobenius_error"],
        "stable_rank": result["spectrum"]["stable_rank"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()

