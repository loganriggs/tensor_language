#!/usr/bin/env python3
"""Complete standalone shared-QK-384 composition and context gate."""

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
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import tensor_attention_projection_frontier as frontier
from tensor_bilin18_program import TensorBilin18Program
from tensor_preserving_attention import TensorAttentionBank
from tensor_preserving_attention_identity import deterministic_tokens
from tensor_preserving_mlp import TensorMLPBank


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_bilin18_shared_qk_whole_program_results.json"
PREREG = HERE / "TENSOR_BILIN18_SHARED_QK_WHOLE_PROGRAM_PREREGISTRATION.md"
ATTENTION_PARENT = HERE / "tensor_attention_projection_frontier_results.json"
EXACT_PARENT = HERE / "tensor_bilin18_standalone_identity_results.json"
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_bilin18_program.py",
    HERE / "tensor_attention_projection_frontier.py",
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_mlp.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_tensor_bilin18_shared_qk_whole_program.py",
)
RANK = 384
LAYERS = 18
BATCH = 4


class _CapturedTarget(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer_set(values) -> set[int]:
    return {value.untyped_storage().data_ptr() for value in values if value.numel()}


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("shared-QK whole-program publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def empty_score() -> dict[str, dict[str, float | int]]:
    return {
        name: {"loss_sum": 0.0, "positions": 0}
        for name in ("all", "seen_current", "unseen_current")
    }


@torch.no_grad()
def add_score(
    accumulator: dict[str, dict[str, float | int]], logits: torch.Tensor,
    rows: torch.Tensor, seen: torch.Tensor, *, score_start: int = frontier.SCORE_START,
) -> None:
    targets = rows[:, 1 : logits.shape[1] + 1].to(logits.device)
    inputs = rows[:, : logits.shape[1]].to(logits.device)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).reshape_as(targets)[:, score_start:]
    covered = seen[inputs[:, score_start:]]
    masks = {
        "all": torch.ones_like(covered),
        "seen_current": covered,
        "unseen_current": ~covered,
    }
    for name, mask in masks.items():
        accumulator[name]["loss_sum"] = float(accumulator[name]["loss_sum"]) + float(
            losses[mask].double().sum()
        )
        accumulator[name]["positions"] = int(accumulator[name]["positions"]) + int(
            mask.sum()
        )


def finalize_score(accumulator: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    result = {}
    for name, row in accumulator.items():
        count = int(row["positions"])
        result[name] = {
            "ce": float(row["loss_sum"]) / count if count else None,
            "positions": count,
        }
    return result


@torch.no_grad()
def compile_shared_bank(model: torch.nn.Module, fit_rows: torch.Tensor):
    programs = []
    receipt: dict[str, Any] = {}
    device = next(model.parameters()).device
    blocks = tuple(model.transformer.h)
    batch = frontier.FIT_BATCH
    spec = frontier.ArmSpec(qk_rank=RANK, value_rank=None, shared_qk=True)

    for target in range(LAYERS):
        covariance = torch.zeros(
            frontier.D, frontier.D, dtype=torch.float64, device=device,
        )
        positions = 0
        for start in range(0, len(fit_rows), batch):
            tokens = fit_rows[start : start + batch, : frontier.T].to(device)
            if len(tokens) != batch:
                raise RuntimeError("fit role is not divisible by fit batch")

            def attention(event: facade.AttentionEvent):
                nonlocal positions
                if event.site < target:
                    return programs[event.site](event.state, event.first_value)
                if event.site != target:
                    raise RuntimeError("shared-QK fit dispatcher passed target")
                state = event.state.reshape(-1, frontier.D).double()
                covariance.addmm_(state.T, state)
                positions += state.shape[0]
                raise _CapturedTarget

            def mlp(event: facade.EarlyMLPEvent):
                return event.block.mlp(event.state)

            try:
                facade.forward_with_dispatch(
                    model, tokens, attention, mlp, require_production=False,
                )
            except _CapturedTarget:
                pass
            else:
                raise RuntimeError("shared-QK fit target was not captured")

        expected = len(fit_rows) * frontier.T
        if positions != expected:
            raise RuntimeError("shared-QK fit position ledger changed")
        normalized = covariance / positions
        program = frontier.compile_site(blocks[target].attn, normalized, spec)
        programs.append(program)
        receipt[str(target)] = {
            "positions": positions,
            "covariance_trace": float(torch.trace(normalized)),
            "cost": asdict(program.cost_receipt()),
        }
        print(f"fit shared-QK site {target + 1}/{LAYERS}", flush=True)
    return TensorAttentionBank(programs), receipt


@torch.no_grad()
def score_native(model, rows: torch.Tensor, seen: torch.Tensor) -> dict[str, Any]:
    accumulator = empty_score()
    calls = {site: 0 for site in range(LAYERS)}
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        tokens = batch[:, : frontier.T].to(next(model.parameters()).device)

        def attention(event: facade.AttentionEvent):
            calls[event.site] += 1
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            return event.block.mlp(event.state)

        add_score(
            accumulator,
            facade.forward_with_dispatch(model, tokens, attention, mlp),
            batch, seen,
        )
    expected = len(rows) // BATCH
    if set(calls.values()) != {expected}:
        raise RuntimeError("native role call ledger changed")
    result = finalize_score(accumulator)
    result["native_attention_calls"] = calls
    return result


@torch.no_grad()
def score_program(
    program: TensorBilin18Program, rows: torch.Tensor, seen: torch.Tensor,
) -> dict[str, Any]:
    accumulator = empty_score()
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        tokens = batch[:, : frontier.T].to(program.token_embedding.device)
        add_score(accumulator, program(tokens), batch, seen)
    return finalize_score(accumulator)


@torch.no_grad()
def build_reference_and_program(device: torch.device):
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    fit_rows = frontier.load_rows(frontier.FIT_ROWS, 480)
    mask_rows = frontier.load_rows(frontier.MASK_ROWS, 96)
    role_rows = {
        name: frontier.load_rows(path, 192) for name, path in frontier.EVAL_ROLES.items()
    }
    seen = frontier.seen_token_mask(mask_rows, device)
    shared_bank, fit_receipt = compile_shared_bank(model, fit_rows)
    mlp_bank = TensorMLPBank.from_model(model)
    blocks = tuple(model.transformer.h)
    program = TensorBilin18Program(
        token_embedding=model.transformer.wte.weight.detach(),
        residual_lambdas=torch.stack([block.lambdas.detach() for block in blocks]),
        unembedding=model.lm_head.weight.detach(),
        attention_bank=shared_bank,
        mlp_bank=mlp_bank,
    )

    native_roles = {
        name: score_native(model, rows, seen) for name, rows in role_rows.items()
    }
    tokens = deterministic_tokens(device)
    changed = tokens.clone()
    changed[:, 32] = (changed[:, 32] + 1) % facade.TOKENIZER_VOCAB

    def native_attention(event: facade.AttentionEvent):
        return event.block.attn(event.state, event.first_value)

    def native_mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    native_base = facade.forward_with_dispatch(
        model, tokens, native_attention, native_mlp,
    )
    native_changed = facade.forward_with_dispatch(
        model, changed, native_attention, native_mlp,
    )
    native_ptrs = pointer_set(tuple(model.parameters()) + tuple(model.buffers()))
    program_ptrs = pointer_set(tuple(program.parameters()) + tuple(program.buffers()))
    storage_disjoint = native_ptrs.isdisjoint(program_ptrs)
    if not storage_disjoint:
        raise RuntimeError("compressed program aliases checkpoint storage")
    native_module_references = [
        f"{module.__class__.__module__}.{module.__class__.__qualname__}"
        for module in program.modules()
        if module.__class__.__module__.startswith("jacclust.tt_model")
    ]
    if native_module_references:
        raise RuntimeError("compressed program retains checkpoint modules")
    model_reference = weakref.ref(model)
    return {
        "program": program, "checkpoint": checkpoint,
        "fit_rows": fit_rows, "mask_rows": mask_rows, "role_rows": role_rows,
        "seen": seen, "fit_receipt": fit_receipt, "native_roles": native_roles,
        "tokens": tokens, "changed": changed,
        "native_base": native_base, "native_changed": native_changed,
        "storage_disjoint": storage_disjoint,
        "native_module_references": native_module_references,
        "model_reference": model_reference,
    }


def context_metrics(
    native_base: torch.Tensor, native_changed: torch.Tensor,
    program_base: torch.Tensor, program_changed: torch.Tensor,
) -> dict[str, float | bool]:
    native_delta = (
        native_changed[:, 33:] - native_base[:, 33:]
    ).double().reshape(-1)
    program_delta = (
        program_changed[:, 33:] - program_base[:, 33:]
    ).double().reshape(-1)
    signal = float(torch.dot(native_delta, native_delta))
    error = float(torch.dot(program_delta - native_delta, program_delta - native_delta))
    program_norm = float(torch.linalg.vector_norm(program_delta))
    native_norm = math.sqrt(signal)
    cosine = (
        float(torch.dot(native_delta, program_delta)) / (native_norm * program_norm)
        if native_norm > 0 and program_norm > 0 else 0.0
    )
    return {
        "downstream_current_tokens_fixed": True,
        "native_max_abs": float(native_delta.abs().max()),
        "program_max_abs": float(program_delta.abs().max()),
        "context_delta_recovery": 1.0 - error / signal,
        "context_delta_cosine": cosine,
        "context_delta_norm_ratio": program_norm / native_norm,
    }


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("shared-QK whole-program result is create-only and already exists")
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = build_reference_and_program(device)
    program = built.pop("program")
    model_reference = built.pop("model_reference")
    gc.collect()
    if model_reference() is not None:
        raise RuntimeError("checkpoint survives compressed-program construction boundary")

    program_roles = {
        name: score_program(program, rows, built["seen"])
        for name, rows in built["role_rows"].items()
    }
    program_base = program(built["tokens"])
    program_changed = program(built["changed"])
    context = context_metrics(
        built["native_base"], built["native_changed"], program_base, program_changed,
    )
    parent = json.loads(ATTENTION_PARENT.read_text())
    comparisons: dict[str, Any] = {}
    for role in built["native_roles"]:
        native = built["native_roles"][role]
        measured = program_roles[role]
        comparisons[role] = {
            "all_ce_harm": measured["all"]["ce"] - native["all"]["ce"],
            "covered_ce_harm": (
                measured["seen_current"]["ce"] - native["seen_current"]["ce"]
            ),
            "unseen_ce_harm": (
                measured["unseen_current"]["ce"] - native["unseen_current"]["ce"]
            ),
            "parent_shared_covered_ce_abs_error": abs(
                measured["seen_current"]["ce"]
                - parent["roles"][role]["arms"]["shared_qk384"]["ce"]
            ),
        }
    cost = program.cost_receipt()
    expected_total = 490_165_686
    dense_total = 545_904_054
    harms = [row["covered_ce_harm"] for row in comparisons.values()]
    predictions = {
        "A_complete_price_and_ownership": (
            cost["total_stored_values"] == expected_total
            and cost["native_calls_per_forward"] == 0
            and cost["fitted_lookup_table_values"] == 0
            and cost["total_input_support"]
            and built["storage_disjoint"]
            and not built["native_module_references"]
        ),
        "B_covered_ce_harm_at_most_0.03": all(
            row["covered_ce_harm"] <= 0.03
            and row["parent_shared_covered_ce_abs_error"] <= 0.003
            for row in comparisons.values()
        ),
        "C_all_position_ce_harm_at_most_0.05": all(
            row["all_ce_harm"] <= 0.05 for row in comparisons.values()
        ),
        "D_context_transport": (
            context["program_max_abs"] > 0
            and context["context_delta_recovery"] >= 0.90
            and context["context_delta_cosine"] >= 0.95
        ),
        "E_role_harm_replication_within_0.01": max(harms) - min(harms) <= 0.01,
    }
    result = {
        "status": "pass" if all(predictions.values()) else "measured_gate_failure",
        "scope": "shared-QK-384 attention plus exact MLPs in complete standalone program",
        "checkpoint": asdict(built["checkpoint"]),
        "roles": {
            name: {"native": built["native_roles"][name], "program": program_roles[name]}
            for name in built["native_roles"]
        },
        "comparisons": comparisons,
        "context_gate": context,
        "cost": {
            **cost,
            "dense_reference_stored_values": dense_total,
            "stored_values_saved": dense_total - int(cost["total_stored_values"]),
            "stored_fraction_of_dense": int(cost["total_stored_values"]) / dense_total,
        },
        "operations_production_forward": program.operation_receipt(
            batch=BATCH, sequence=frontier.T,
        ),
        "execution": {
            "checkpoint_model_collected_before_scoring": True,
            "native_program_storage_disjoint": built["storage_disjoint"],
            "native_module_references": built["native_module_references"],
            "program_native_calls": 0,
        },
        "predictions": predictions,
        "fit": built["fit_receipt"],
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "parents": {
                "attention": sha256_file(ATTENTION_PARENT),
                "exact_complete_program": sha256_file(EXACT_PARENT),
            },
            "roles": {
                "fit": sha256_file(frontier.FIT_ROWS),
                "mask": sha256_file(frontier.MASK_ROWS),
                **{name: sha256_file(path) for name, path in frontier.EVAL_ROLES.items()},
            },
        },
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({
        "status": outcome["status"],
        "comparisons": outcome["comparisons"],
        "context_gate": outcome["context_gate"],
        "cost": {
            key: outcome["cost"][key] for key in (
                "total_stored_values", "stored_values_saved", "stored_fraction_of_dense",
            )
        },
        "predictions": outcome["predictions"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
