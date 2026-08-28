#!/usr/bin/env python3
"""Prospective multi-intervention causal validation for shared-QK ranks 512 and 640."""

from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import weakref

import torch

import bilin18_observed_model_facade as facade
import early_mlp_suffix_transport_v1_rows as row_authority
import tensor_attention_projection_frontier as frontier
import tensor_bilin18_shared_qk_whole_program as base
from tensor_bilin18_program import TensorBilin18Program
from tensor_preserving_mlp import TensorMLPBank


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
OUTPUT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
PREREG = HERE / "TENSOR_BILIN18_CAUSAL_INTERVENTION_BANK_PREREGISTRATION.md"
RANKS = (512, 640)
EXPECTED_TOTALS = {512: 503_436_726, 640: 516_707_766}
DENSE_TOTAL = 545_904_054
ROLE_PATHS = {
    "cross_task_skip31000": BQ / ".rowcache_compiler_v2/fineweb_n192_skip31000.pt",
    "cross_task_skip35000": BQ / ".rowcache_compiler_v2/fineweb_n192_skip35000.pt",
}
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20_260_828
EVAL_BATCH = 2
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_bilin18_shared_qk_whole_program.py",
    HERE / "tensor_bilin18_program.py",
    HERE / "tensor_attention_projection_frontier.py",
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_mlp.py",
    HERE / "test_tensor_bilin18_causal_intervention_bank.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("causal-bank publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def validate_role(path: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    expected_file, expected_raw, payload_key = row_authority.CANONICAL_ROW_TENSORS[path]
    if payload_key is not None:
        raise RuntimeError("causal-bank role unexpectedly uses a nested tensor payload")
    observed_file = row_authority.file_sha256(path)
    rows = torch.load(path, map_location="cpu", weights_only=True)
    observed_raw = row_authority.tensor_raw_sha256(rows)
    if (
        observed_file != expected_file or observed_raw != expected_raw
        or tuple(rows.shape) != (192, 513) or rows.dtype != torch.long
    ):
        raise RuntimeError(f"causal-bank row authority failed for {path}")
    receipt = {
        "path": str(path.resolve()), "shape": list(rows.shape),
        "serialized_sha256": observed_file, "tensor_raw_sha256": observed_raw,
    }
    return rows, receipt


def synthetic_tokens(family: int) -> torch.Tensor:
    slopes = (65_537, 99_991)
    offsets = (4_093, 12_289)
    index = torch.arange(frontier.T, dtype=torch.long)
    return (index * slopes[family] + offsets[family]) % facade.TOKENIZER_VOCAB


def make_fixture(
    name: str, tokens: torch.Tensor, position: int, delta: int, source: str,
) -> dict[str, Any]:
    original = int(tokens[position])
    changed = tokens.clone()
    changed[position] = (changed[position] + delta) % facade.TOKENIZER_VOCAB
    return {
        "name": name, "tokens": tokens.clone(), "changed": changed,
        "position": position, "delta": delta, "source": source,
        "original_token": original, "replacement_token": int(changed[position]),
    }


def build_fixture_bank() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    authority: dict[str, Any] = {}
    natural_deltas = (1, 257, 4_093, 17)
    for role_index, (role, path) in enumerate(ROLE_PATHS.items()):
        rows, authority[role] = validate_role(path)
        for row_index in (0, 1):
            tokens = rows[row_index, : frontier.T]
            for pos_index, position in enumerate((16, 96)):
                offset_index = role_index * 2 + row_index
                delta = natural_deltas[(offset_index + pos_index) % len(natural_deltas)]
                fixtures.append(make_fixture(
                    f"natural_{role}_row{row_index}_pos{position}", tokens,
                    position, delta, f"{role}:row{row_index}",
                ))
    synthetic_deltas = (17, 257, 4_093, 8_191)
    for family in range(2):
        tokens = synthetic_tokens(family)
        for position, delta in zip((8, 32, 96, 160), synthetic_deltas, strict=True):
            fixtures.append(make_fixture(
                f"synthetic_family{family}_pos{position}", tokens, position,
                delta, f"affine_family_{family}",
            ))
    names = [fixture["name"] for fixture in fixtures]
    signatures = {
        (fixture["source"], fixture["position"], fixture["original_token"],
         fixture["replacement_token"])
        for fixture in fixtures
    }
    if len(fixtures) != 16 or len(names) != len(set(names)) or len(signatures) != 16:
        raise RuntimeError("causal intervention bank is not the frozen 16 unique fixtures")
    return fixtures, authority


def native_forward(model: torch.nn.Module, tokens: torch.Tensor) -> torch.Tensor:
    def attention(event: facade.AttentionEvent):
        return event.block.attn(event.state, event.first_value)

    def mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    return facade.forward_with_dispatch(model, tokens, attention, mlp)


@torch.no_grad()
def collect_deltas(
    forward, fixtures: list[dict[str, Any]], device: torch.device,
) -> dict[str, torch.Tensor]:
    deltas: dict[str, torch.Tensor] = {}
    for start in range(0, len(fixtures), EVAL_BATCH):
        chunk = fixtures[start : start + EVAL_BATCH]
        tokens = torch.stack([row["tokens"] for row in chunk]).to(device)
        changed = torch.stack([row["changed"] for row in chunk]).to(device)
        base_logits = forward(tokens)
        changed_logits = forward(changed)
        for index, fixture in enumerate(chunk):
            position = int(fixture["position"])
            deltas[fixture["name"]] = (
                changed_logits[index, position + 1 :] - base_logits[index, position + 1 :]
            ).detach().cpu()
        del base_logits, changed_logits, tokens, changed
    return deltas


def delta_metrics(native_delta: torch.Tensor, program_delta: torch.Tensor) -> dict[str, float]:
    native = native_delta.double().reshape(-1)
    program = program_delta.double().reshape(-1)
    signal = float(torch.dot(native, native))
    error = float(torch.dot(program - native, program - native))
    native_norm = math.sqrt(signal)
    program_norm = float(torch.linalg.vector_norm(program))
    cosine = (
        float(torch.dot(native, program)) / (native_norm * program_norm)
        if native_norm > 0 and program_norm > 0 else 0.0
    )
    return {
        "native_max_abs": float(native.abs().max()),
        "program_max_abs": float(program.abs().max()),
        "recovery": 1.0 - error / signal if signal > 0 else float("-inf"),
        "cosine": cosine,
        "norm_ratio": program_norm / native_norm if native_norm > 0 else float("inf"),
    }


def bootstrap_lcb(values: list[float]) -> float:
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator(device="cpu").manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(
        len(values), (N_BOOTSTRAP, len(values)), generator=generator,
    )
    means = tensor[indices].mean(dim=1)
    return float(torch.quantile(means, 0.05))


def summarize(metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    recoveries = [row["recovery"] for row in metrics.values()]
    cosines = [row["cosine"] for row in metrics.values()]
    pass_count = sum(
        row["recovery"] >= 0.90 and row["cosine"] >= 0.95
        for row in metrics.values()
    )
    return {
        "fixtures": len(metrics),
        "mean_recovery": sum(recoveries) / len(recoveries),
        "median_recovery": float(torch.tensor(recoveries).median()),
        "minimum_recovery": min(recoveries),
        "recovery_bootstrap_95_lcb": bootstrap_lcb(recoveries),
        "mean_cosine": sum(cosines) / len(cosines),
        "median_cosine": float(torch.tensor(cosines).median()),
        "minimum_cosine": min(cosines),
        "cosine_bootstrap_95_lcb": bootstrap_lcb(cosines),
        "individual_joint_pass_count": pass_count,
        "individual_joint_pass_fraction": pass_count / len(metrics),
        "all_native_and_program_signals_nonzero": all(
            row["native_max_abs"] > 0 and row["program_max_abs"] > 0
            for row in metrics.values()
        ),
    }


def robust_gate(summary: dict[str, Any]) -> bool:
    return bool(
        summary["recovery_bootstrap_95_lcb"] >= 0.90
        and summary["cosine_bootstrap_95_lcb"] >= 0.95
        and summary["individual_joint_pass_fraction"] >= 0.75
        and summary["all_native_and_program_signals_nonzero"]
    )


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("causal intervention bank result is create-only and already exists")
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixtures, authority = build_fixture_bank()
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    model_reference = weakref.ref(model)
    fit_rows = frontier.load_rows(frontier.FIT_ROWS, 480)
    native_deltas = collect_deltas(lambda x: native_forward(model, x), fixtures, device)

    candidates: dict[str, Any] = {}
    for rank in RANKS:
        base.RANK = rank
        attention_bank, fit_receipt = base.compile_shared_bank(model, fit_rows)
        mlp_bank = TensorMLPBank.from_model(model)
        blocks = tuple(model.transformer.h)
        program = TensorBilin18Program(
            token_embedding=model.transformer.wte.weight.detach(),
            residual_lambdas=torch.stack([block.lambdas.detach() for block in blocks]),
            unembedding=model.lm_head.weight.detach(),
            attention_bank=attention_bank,
            mlp_bank=mlp_bank,
        )
        native_ptrs = base.pointer_set(tuple(model.parameters()) + tuple(model.buffers()))
        program_ptrs = base.pointer_set(tuple(program.parameters()) + tuple(program.buffers()))
        storage_disjoint = native_ptrs.isdisjoint(program_ptrs)
        native_module_references = [
            f"{module.__class__.__module__}.{module.__class__.__qualname__}"
            for module in program.modules()
            if module.__class__.__module__.startswith("jacclust.tt_model")
        ]
        program_deltas = collect_deltas(program, fixtures, device)
        fixture_metrics = {
            fixture["name"]: delta_metrics(
                native_deltas[fixture["name"]], program_deltas[fixture["name"]],
            )
            for fixture in fixtures
        }
        summary = summarize(fixture_metrics)
        cost = program.cost_receipt()
        ownership = bool(
            cost["total_stored_values"] == EXPECTED_TOTALS[rank]
            and cost["native_calls_per_forward"] == 0
            and cost["fitted_lookup_table_values"] == 0
            and cost["total_input_support"] and storage_disjoint
            and not native_module_references
        )
        candidates[str(rank)] = {
            "rank": rank,
            "fixtures": fixture_metrics,
            "summary": summary,
            "robust_gate": robust_gate(summary),
            "ownership_and_price_gate": ownership,
            "cost": {
                **cost,
                "dense_reference_stored_values": DENSE_TOTAL,
                "stored_values_saved": DENSE_TOTAL - int(cost["total_stored_values"]),
                "stored_fraction_of_dense": int(cost["total_stored_values"]) / DENSE_TOTAL,
            },
            "fit": fit_receipt,
            "execution": {
                "native_program_storage_disjoint": storage_disjoint,
                "native_module_references": native_module_references,
                "program_native_calls": 0,
            },
        }
        del program_deltas, program, attention_bank, mlp_bank, blocks
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    rank512_summary = candidates["512"]["summary"]
    rank640_summary = candidates["640"]["summary"]
    recovery_gain = rank640_summary["mean_recovery"] - rank512_summary["mean_recovery"]
    cosine_gain = rank640_summary["mean_cosine"] - rank512_summary["mean_cosine"]
    nonregression_fraction = sum(
        candidates["640"]["fixtures"][name]["recovery"]
        >= candidates["512"]["fixtures"][name]["recovery"]
        for name in candidates["512"]["fixtures"]
    ) / len(fixtures)
    paired = {
        "mean_recovery_gain_640_minus_512": recovery_gain,
        "mean_cosine_gain_640_minus_512": cosine_gain,
        "rank640_recovery_nonregression_fraction": nonregression_fraction,
    }
    predictions = {
        "A_fixture_and_authority_gate": len(fixtures) == 16 and len(authority) == 2,
        "B_ownership_and_price_gate": all(
            row["ownership_and_price_gate"] for row in candidates.values()
        ),
        "C_rank512_robust_gate": candidates["512"]["robust_gate"],
        "D_rank640_rescue_gate": candidates["640"]["robust_gate"],
        "E_paired_capacity_evidence": (
            recovery_gain >= 0.01 and cosine_gain >= 0.005
            and nonregression_fraction >= 0.75
        ),
    }
    if predictions["C_rank512_robust_gate"]:
        status = "rank512_robust_pass"
    elif predictions["D_rank640_rescue_gate"] and predictions["E_paired_capacity_evidence"]:
        status = "rank640_robust_pass"
    else:
        status = "measured_gate_failure"

    del model
    gc.collect()
    if model_reference() is not None:
        raise RuntimeError("checkpoint survives causal-bank construction boundary")
    fixture_receipt = [
        {key: value for key, value in fixture.items() if key not in ("tokens", "changed")}
        for fixture in fixtures
    ]
    result = {
        "status": status,
        "scope": "prospective 16-fixture causal bank for complete shared-QK programs",
        "checkpoint": asdict(checkpoint),
        "fixtures": fixture_receipt,
        "candidates": candidates,
        "paired_comparison": paired,
        "predictions": predictions,
        "execution": {"checkpoint_model_collected_before_publication": True},
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "roles": authority,
            "fit": sha256_file(frontier.FIT_ROWS),
        },
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({
        "status": outcome["status"],
        "summaries": {
            rank: row["summary"] for rank, row in outcome["candidates"].items()
        },
        "paired_comparison": outcome["paired_comparison"],
        "predictions": outcome["predictions"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
