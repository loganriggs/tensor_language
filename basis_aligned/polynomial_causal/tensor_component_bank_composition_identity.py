#!/usr/bin/env python3
"""Role-free identity gate for simultaneous owned attention and MLP banks."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from tensor_preserving_attention import PROJECTION_NAMES, TensorAttentionBank
from tensor_preserving_attention_identity import (
    AttentionNativePoison, deterministic_tokens, tensor_sha256,
)
from tensor_preserving_mlp import TensorMLPBank
from tensor_preserving_mlp_identity import MLPNativePoison


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_component_bank_composition_identity_results.json"
ATTENTION_PARENT = HERE / "tensor_preserving_attention_identity_results.json"
MLP_PARENT = HERE / "tensor_preserving_mlp_identity_results.json"
PREREG = HERE / "TENSOR_COMPONENT_BANK_COMPOSITION_PREREGISTRATION.md"
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_mlp.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_tensor_component_bank_composition_identity.py",
)
LAYERS = 18
UNOWNED_EXACT_INTERFACES = (
    "transformer.wte token embedding",
    "18 block residual lambda pairs and x0 skip",
    "whole-state RMSNorm calls at embedding/attention/MLP/final interfaces",
    "lm_head unembedding",
    "30*tanh(logits/30) output softcap",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pointer_set(values) -> set[int]:
    return {
        value.untyped_storage().data_ptr() for value in values if value.numel()
    }


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("composition publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.no_grad()
def run_identity() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("component composition result is create-only and already exists")
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    tokens = deterministic_tokens(device)
    attention_bank = TensorAttentionBank.from_model(
        model, ranks={name: None for name in PROJECTION_NAMES},
    )
    mlp_bank = TensorMLPBank.from_model(model)

    native_tensors = tuple(model.parameters()) + tuple(model.buffers())
    attention_tensors = tuple(attention_bank.buffers())
    mlp_tensors = tuple(mlp_bank.buffers())
    native_ptrs = pointer_set(native_tensors)
    attention_ptrs = pointer_set(attention_tensors)
    mlp_ptrs = pointer_set(mlp_tensors)
    mutually_disjoint = (
        native_ptrs.isdisjoint(attention_ptrs)
        and native_ptrs.isdisjoint(mlp_ptrs)
        and attention_ptrs.isdisjoint(mlp_ptrs)
    )
    if not mutually_disjoint:
        raise RuntimeError("component-bank storage aliases native or peer storage")

    native_attn_calls = {site: 0 for site in range(LAYERS)}
    native_mlp_calls = {site: 0 for site in range(LAYERS)}
    native_attn_writes = {}
    native_buses = {}
    native_mlp_writes = {}

    def native_attention(event: facade.AttentionEvent):
        native_attn_calls[event.site] += 1
        write, bus = event.block.attn(event.state, event.first_value)
        native_attn_writes[event.site] = write.detach().cpu().clone()
        native_buses[event.site] = bus.detach().cpu().clone()
        return write, bus

    def native_mlp(event: facade.EarlyMLPEvent):
        native_mlp_calls[event.site] += 1
        write = event.block.mlp(event.state)
        native_mlp_writes[event.site] = write.detach().cpu().clone()
        return write

    native_logits = facade.forward_with_dispatch(
        model, tokens, native_attention, native_mlp,
    )
    expected = {site: 1 for site in range(LAYERS)}
    if native_attn_calls != expected or native_mlp_calls != expected:
        raise RuntimeError("native composition reference call ledger failed")

    program_attn_calls = {site: 0 for site in range(LAYERS)}
    program_mlp_calls = {site: 0 for site in range(LAYERS)}
    attn_diffs = {}
    bus_diffs = {}
    mlp_diffs = {}
    blocks = tuple(model.transformer.h)
    attention_poison = AttentionNativePoison(model)
    mlp_poison = MLPNativePoison(model)
    with attention_bank.begin(blocks) as attention_transaction:
        with mlp_bank.begin(blocks) as mlp_transaction:
            def program_attention(event: facade.AttentionEvent):
                program_attn_calls[event.site] += 1
                write, bus = attention_transaction(event)
                attn_diffs[str(event.site)] = float((
                    write.float() - native_attn_writes[event.site].to(write.device).float()
                ).abs().max())
                bus_diffs[str(event.site)] = float((
                    bus.float() - native_buses[event.site].to(bus.device).float()
                ).abs().max())
                return write, bus

            def program_mlp(event: facade.EarlyMLPEvent):
                program_mlp_calls[event.site] += 1
                write = mlp_transaction(event)
                mlp_diffs[str(event.site)] = float((
                    write.float() - native_mlp_writes[event.site].to(write.device).float()
                ).abs().max())
                return write

            with attention_poison.scope():
                with mlp_poison.scope():
                    program_logits = facade.forward_with_dispatch(
                        model, tokens, program_attention, program_mlp,
                    )

    ledgers_pass = (
        program_attn_calls == expected and program_mlp_calls == expected
        and not any(attention_poison.calls.values()) and not any(mlp_poison.calls.values())
        and attention_poison.restored and attention_poison.inert
        and mlp_poison.restored and mlp_poison.inert
    )
    if not ledgers_pass:
        raise RuntimeError("simultaneous component execution ledger failed")
    logit_diff = float((program_logits - native_logits).abs().max())
    native_hash = tensor_sha256(native_logits)
    program_hash = tensor_sha256(program_logits)
    if max(attn_diffs.values()) or max(bus_diffs.values()) or max(mlp_diffs.values()) or (
        logit_diff or native_hash != program_hash
    ):
        raise RuntimeError("simultaneous component banks failed bitwise identity")
    targets = torch.roll(tokens, shifts=-1, dims=1)
    native_ce = float(F.cross_entropy(
        native_logits[:, 64:].reshape(-1, native_logits.shape[-1]),
        targets[:, 64:].reshape(-1),
    ))
    program_ce = float(F.cross_entropy(
        program_logits[:, 64:].reshape(-1, program_logits.shape[-1]),
        targets[:, 64:].reshape(-1),
    ))

    attention_cost = attention_bank.cost_receipt()
    mlp_cost = mlp_bank.cost_receipt()
    total = attention_cost["total_stored_values"] + mlp_cost["total_stored_values"]
    if total != 430_003_602:
        raise RuntimeError("combined dense component denominator changed")
    result = {
        "status": "pass",
        "scope": "36 attention/MLP components; facade interfaces listed separately",
        "checkpoint": asdict(checkpoint),
        "fixture": {"shape": list(tokens.shape), "sha256": tensor_sha256(tokens)},
        "numerical": {
            "attention_write_max_abs": attn_diffs,
            "bus_max_abs": bus_diffs,
            "mlp_write_max_abs": mlp_diffs,
            "logit_max_abs": logit_diff,
            "native_logit_sha256": native_hash,
            "program_logit_sha256": program_hash,
            "native_ce": native_ce, "program_ce": program_ce,
        },
        "execution": {
            "native_attention_calls": native_attn_calls,
            "native_mlp_calls": native_mlp_calls,
            "program_attention_calls": program_attn_calls,
            "program_mlp_calls": program_mlp_calls,
            "literal_native_attention_calls": sum(attention_poison.calls.values()),
            "literal_native_mlp_calls": sum(mlp_poison.calls.values()),
            "attention_closure": asdict(attention_transaction.closure),
            "mlp_closure": asdict(mlp_transaction.closure),
            "mutually_disjoint_storage": mutually_disjoint,
        },
        "cost": {
            "attention": attention_cost, "mlp": mlp_cost,
            "total_component_stored_values": total,
        },
        "unowned_exact_facade_interfaces": list(UNOWNED_EXACT_INTERFACES),
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "parent_identities": {
                "attention": sha256_file(ATTENTION_PARENT),
                "mlp": sha256_file(MLP_PARENT),
            },
        },
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run_identity()
    print(json.dumps({
        "status": outcome["status"], "numerical": outcome["numerical"],
        "total_component_stored_values": outcome["cost"]["total_component_stored_values"],
        "unowned_exact_facade_interfaces": outcome["unowned_exact_facade_interfaces"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
