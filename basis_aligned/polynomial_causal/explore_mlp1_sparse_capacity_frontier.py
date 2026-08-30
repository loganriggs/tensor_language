#!/usr/bin/env python3
"""Discovery-only capacity check for storage-bounded sparse MLP1 Down programs.

This deliberately reuses the already-opened FIT and SELECT roles from the C512 run.
It never opens FINAL, and its output is therefore useful for choosing a later fresh
confirmation experiment, not as confirmatory evidence itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch import nn
import torch.nn.functional as F

import run_mlp1_sparse_c512_continue_factorial_v1_fit as base


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
OUTPUT = HERE / "mlp1_sparse_capacity_frontier_discovery.json"

GATE_DIM = 4608
OUTPUT_DIM = 1152
ACTIVE_ATOMS = 32
STEPS = 2400
BATCH_SIZE = 1024
LEARNING_RATE = 0.003
CURVE_EVERY = 200


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def price(dictionary_size: int) -> dict[str, float | int]:
    stored = dictionary_size * (GATE_DIM + OUTPUT_DIM) + OUTPUT_DIM
    native_down = GATE_DIM * OUTPUT_DIM
    native_full_mlp = 15_926_400
    saved = native_down - stored
    score_multiplies = dictionary_size * GATE_DIM
    sparse_decode_multiplies = ACTIVE_ATOMS * OUTPUT_DIM
    executed_multiplies = score_multiplies + sparse_decode_multiplies
    return {
        "dictionary_size": dictionary_size,
        "stored_reals": stored,
        "native_down_reals": native_down,
        "down_storage_fraction": stored / native_down,
        "full_mlp_storage_saved_reals": saved,
        "full_mlp_storage_saved_fraction": saved / native_full_mlp,
        "score_multiplies_per_token": score_multiplies,
        "sparse_decode_multiplies_per_token": sparse_decode_multiplies,
        "executed_down_multiplies_per_token": executed_multiplies,
        "down_multiply_fraction": executed_multiplies / native_down,
        "full_mlp_dense_map_multiply_saved_fraction": (
            native_down - executed_multiplies
        ) / native_full_mlp,
    }


def topk_relu(scores: torch.Tensor, k: int = ACTIVE_ATOMS) -> torch.Tensor:
    if scores.ndim != 2 or not 0 < k <= scores.shape[1]:
        raise ValueError("invalid TopK input")
    values, indices = scores.topk(k, dim=1, largest=True, sorted=False)
    result = torch.zeros_like(scores)
    result.scatter_(1, indices, values.relu())
    return result


class SparseProgram(nn.Module):
    def __init__(
        self, encoder: torch.Tensor, decoder: torch.Tensor, intercept: torch.Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("encoder", encoder)
        self.register_buffer("decoder", decoder)
        self.register_buffer("intercept", intercept)

    def forward(self, gate: torch.Tensor) -> torch.Tensor:
        shape = gate.shape[:-1]
        flat = gate.float().reshape(-1, GATE_DIM)
        output = topk_relu(flat @ self.encoder.T) @ self.decoder.T + self.intercept
        return output.reshape(*shape, OUTPUT_DIM).to(gate.dtype)


@torch.no_grad()
def r2(
    encoder: torch.Tensor, decoder: torch.Tensor, intercept: torch.Tensor,
    gates: torch.Tensor, targets: torch.Tensor, device: torch.device,
) -> float:
    residual = 0.0
    centered = float((targets.double() - targets.double().mean(0)).square().sum())
    for start in range(0, len(gates), 2048):
        gate = gates[start:start + 2048].to(device)
        target = targets[start:start + 2048].to(device)
        prediction = topk_relu(gate @ encoder.T) @ decoder.T + intercept
        residual += float((target.double() - prediction.double()).square().sum())
    return 1.0 - residual / max(centered, 1e-30)


def train(
    dictionary_size: int, seed: int, fit_gate: torch.Tensor, fit_target: torch.Tensor,
    select_gate: torch.Tensor, select_target: torch.Tensor, device: torch.device,
) -> tuple[SparseProgram, list[dict[str, float | int]]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(10_000 + seed)
    encoder = torch.randn(dictionary_size, GATE_DIM, device=device) / math.sqrt(GATE_DIM)
    encoder /= encoder.norm(dim=1, keepdim=True)
    decoder = torch.randn(OUTPUT_DIM, dictionary_size, device=device) / math.sqrt(OUTPUT_DIM)
    intercept = fit_target.mean(0).to(device)
    encoder.requires_grad_(True); decoder.requires_grad_(True); intercept.requires_grad_(True)
    optimizer = torch.optim.Adam([encoder, decoder, intercept], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    curve: list[dict[str, float | int]] = []
    for step in range(1, STEPS + 1):
        indices = torch.randint(len(fit_gate), (BATCH_SIZE,), generator=generator)
        gate = fit_gate.index_select(0, indices).to(device)
        target = fit_target.index_select(0, indices).to(device)
        prediction = topk_relu(gate @ encoder.T) @ decoder.T + intercept
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
        with torch.no_grad():
            encoder /= encoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if step % CURVE_EVERY == 0:
            with torch.no_grad():
                value = r2(encoder, decoder, intercept, select_gate, select_target, device)
            curve.append({"step": step, "train_mse": float(loss), "select_r2": value})
            print(f"P={dictionary_size} step={step} select_r2={value:.6f}", flush=True)
    program = SparseProgram(encoder.detach(), decoder.detach(), intercept.detach()).eval()
    return program, curve


def load_opened_role(entry: dict[str, object]) -> torch.Tensor:
    path = Path(str(entry["path"]))
    if file_sha256(path) != entry["file_sha256"]:
        raise RuntimeError(f"opened role changed: {path}")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value.dtype != torch.long or list(value.shape) != entry["shape"]:
        raise RuntimeError(f"opened role shape/dtype changed: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.dictionary_size <= ACTIVE_ATOMS:
        raise ValueError("dictionary size must exceed active atoms")
    if price(args.dictionary_size)["full_mlp_storage_saved_reals"] <= 0:
        raise ValueError("discovery point is not storage-smaller than native Down")

    started = time.time()
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit = load_opened_role(receipt["entries"]["FIT"])
    select = load_opened_role(receipt["entries"]["SELECT"])
    device = torch.device("cuda")
    model, checkpoint = base.facade.load_bilin18(device=device, dtype=torch.bfloat16)
    fit_gate, fit_target, fit_calls = base.capture_gate_action(model, fit, device)
    select_gate, select_target, select_calls = base.capture_gate_action(model, select, device)
    program, curve = train(
        args.dictionary_size, args.seed, fit_gate, fit_target,
        select_gate, select_target, device,
    )
    select_ce, score_calls = base.score_select_ce(model, select, program, device)
    output = {
        "schema": "mlp1_sparse_capacity_frontier_discovery_v1",
        "status": "discovery_complete",
        "claim_boundary": (
            "Reuses previously opened FIT/SELECT and never opens FINAL; may choose a "
            "fresh confirmation point but is not confirmatory or terminal evidence."
        ),
        "dictionary_size": args.dictionary_size,
        "active_atoms": ACTIVE_ATOMS,
        "seed": args.seed,
        "steps": STEPS,
        "documents": {"FIT": len(fit), "SELECT": len(select), "FINAL_opened": 0},
        "curve": curve,
        "select_ce": select_ce,
        "price": price(args.dictionary_size),
        "calls": {"FIT": fit_calls, "SELECT_capture": select_calls, "SELECT_score": score_calls},
        "checkpoint": checkpoint.__dict__,
        "parents": {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "rows_receipt_sha256": file_sha256(ROWS_RECEIPT),
            "p512_result_sha256": file_sha256(
                HERE / "mlp1_sparse_c512_continue_factorial_v2_fit_result.json"
            ),
        },
        "runtime_seconds": time.time() - started,
    }
    OUTPUT.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
