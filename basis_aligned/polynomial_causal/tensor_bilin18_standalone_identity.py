#!/usr/bin/env python3
"""Role-free identity gate for the complete standalone bilin18 tensor program."""

from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
import weakref

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from tensor_bilin18_program import TensorBilin18Program
from tensor_preserving_attention_identity import deterministic_tokens, tensor_sha256


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_bilin18_standalone_identity_results.json"
PARENT = HERE / "tensor_component_bank_composition_identity_results.json"
PREREG = HERE / "TENSOR_BILIN18_STANDALONE_IDENTITY_PREREGISTRATION.md"
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_bilin18_program.py",
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_mlp.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_tensor_bilin18_program.py",
    HERE / "test_tensor_bilin18_standalone_identity.py",
)
LAYERS = 18


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
                raise OSError("standalone publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.no_grad()
def build_reference_and_program(device: torch.device):
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    tokens = deterministic_tokens(device)
    changed = tokens.clone()
    changed[:, 32] = (changed[:, 32] + 1) % facade.TOKENIZER_VOCAB
    calls = {
        "attention": {site: 0 for site in range(LAYERS)},
        "mlp": {site: 0 for site in range(LAYERS)},
    }

    def native_attention(event: facade.AttentionEvent):
        calls["attention"][event.site] += 1
        return event.block.attn(event.state, event.first_value)

    def native_mlp(event: facade.EarlyMLPEvent):
        calls["mlp"][event.site] += 1
        return event.block.mlp(event.state)

    native_logits = facade.forward_with_dispatch(
        model, tokens, native_attention, native_mlp,
    )
    changed_native_logits = facade.forward_with_dispatch(
        model, changed, native_attention, native_mlp,
    )
    expected = {site: 2 for site in range(LAYERS)}
    if calls["attention"] != expected or calls["mlp"] != expected:
        raise RuntimeError("native standalone reference call ledger failed")

    program = TensorBilin18Program.from_model(model)
    native_ptrs = pointer_set(tuple(model.parameters()) + tuple(model.buffers()))
    program_ptrs = pointer_set(tuple(program.parameters()) + tuple(program.buffers()))
    storage_disjoint = native_ptrs.isdisjoint(program_ptrs)
    if not storage_disjoint:
        raise RuntimeError("standalone tensor storage aliases native checkpoint storage")
    native_type_prefix = "jacclust.tt_model"
    native_module_references = [
        f"{module.__class__.__module__}.{module.__class__.__qualname__}"
        for module in program.modules()
        if module.__class__.__module__.startswith(native_type_prefix)
    ]
    if native_module_references:
        raise RuntimeError("standalone program retains native checkpoint modules")
    model_reference = weakref.ref(model)
    return {
        "program": program,
        "tokens": tokens,
        "changed": changed,
        "native_logits": native_logits,
        "changed_native_logits": changed_native_logits,
        "checkpoint": checkpoint,
        "calls": calls,
        "storage_disjoint": storage_disjoint,
        "native_module_references": native_module_references,
        "model_reference": model_reference,
    }


@torch.no_grad()
def run_identity() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("standalone result is create-only and already exists")
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    built = build_reference_and_program(device)
    program = built.pop("program")
    model_reference = built.pop("model_reference")
    gc.collect()
    if model_reference() is not None:
        raise RuntimeError("checkpoint model survives standalone construction boundary")

    program_logits = program(built["tokens"])
    changed_program_logits = program(built["changed"])
    base_max_abs = float((program_logits - built["native_logits"]).abs().max())
    changed_max_abs = float((
        changed_program_logits - built["changed_native_logits"]
    ).abs().max())
    base_native_hash = tensor_sha256(built["native_logits"])
    base_program_hash = tensor_sha256(program_logits)
    changed_native_hash = tensor_sha256(built["changed_native_logits"])
    changed_program_hash = tensor_sha256(changed_program_logits)
    if base_max_abs or changed_max_abs or base_native_hash != base_program_hash or (
        changed_native_hash != changed_program_hash
    ):
        raise RuntimeError("standalone program failed bitwise logit identity")

    later = slice(33, None)
    native_context_max_abs = float((
        built["changed_native_logits"][:, later] - built["native_logits"][:, later]
    ).abs().max())
    program_context_max_abs = float((
        changed_program_logits[:, later] - program_logits[:, later]
    ).abs().max())
    if native_context_max_abs <= 0 or program_context_max_abs != native_context_max_abs:
        raise RuntimeError("standalone program failed prefix-context transport identity")

    targets = torch.roll(built["tokens"], shifts=-1, dims=1)
    native_ce = float(F.cross_entropy(
        built["native_logits"][:, 64:].reshape(-1, built["native_logits"].shape[-1]),
        targets[:, 64:].reshape(-1),
    ))
    program_ce = float(F.cross_entropy(
        program_logits[:, 64:].reshape(-1, program_logits.shape[-1]),
        targets[:, 64:].reshape(-1),
    ))
    if native_ce != program_ce:
        raise RuntimeError("standalone program failed exact CE identity")

    cost = program.cost_receipt()
    if int(cost["total_stored_values"]) != 545_904_054 or int(
        cost["shell"]["total_shell_stored_values"]
    ) != 115_900_452 or int(
        cost["attention"]["total_stored_values"]
    ) + int(cost["mlp"]["total_stored_values"]) != 430_003_602:
        raise RuntimeError("complete standalone storage denominator changed")
    if cost["native_calls_per_forward"] or cost["fitted_lookup_table_values"] or not (
        cost["total_input_support"]
    ):
        raise RuntimeError("standalone ownership receipt failed")

    result = {
        "status": "pass",
        "scope": "complete exact standalone bilin18 tensor program",
        "checkpoint": asdict(built["checkpoint"]),
        "fixture": {
            "shape": list(built["tokens"].shape),
            "base_sha256": tensor_sha256(built["tokens"]),
            "changed_sha256": tensor_sha256(built["changed"]),
            "prefix_changed_position": 32,
            "downstream_start_position": 33,
        },
        "numerical": {
            "base_logit_max_abs": base_max_abs,
            "changed_logit_max_abs": changed_max_abs,
            "base_native_logit_sha256": base_native_hash,
            "base_program_logit_sha256": base_program_hash,
            "changed_native_logit_sha256": changed_native_hash,
            "changed_program_logit_sha256": changed_program_hash,
            "native_ce": native_ce,
            "program_ce": program_ce,
        },
        "context_gate": {
            "downstream_current_tokens_fixed": True,
            "native_context_max_abs": native_context_max_abs,
            "program_context_max_abs": program_context_max_abs,
            "bitwise_context_transport_identity": True,
        },
        "execution": {
            "native_reference_calls": built["calls"],
            "program_native_calls": 0,
            "checkpoint_model_collected_before_program_execution": True,
            "native_module_references": built["native_module_references"],
            "native_program_storage_disjoint": built["storage_disjoint"],
        },
        "cost": cost,
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCES},
            "parent_component_identity_sha256": sha256_file(PARENT),
        },
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run_identity()
    print(json.dumps({
        "status": outcome["status"],
        "numerical": outcome["numerical"],
        "context_gate": outcome["context_gate"],
        "execution": outcome["execution"],
        "total_stored_values": outcome["cost"]["total_stored_values"],
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
