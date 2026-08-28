#!/usr/bin/env python3
"""Source-closed 18-layer identity gate for the tensor attention program.

This is a numerical/execution identity check on deterministic synthetic tokens.  It
opens no FineWeb or evaluation role and performs no fitting or selection.  The program
pass poisons every literal native attention forward before calling the source-closed
dispatcher surface, so a post-forward hook or fallback cannot pass the call ledger.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterator

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from tensor_preserving_attention import (
    PROJECTION_NAMES, TensorAttentionBank,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_preserving_attention_identity_results.json"
SOURCE_FILES = (
    Path(__file__).resolve(),
    HERE / "tensor_preserving_attention.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "TENSOR_PRESERVING_ATTENTION_PREREGISTRATION.md",
    HERE / "test_tensor_preserving_attention.py",
    HERE / "test_tensor_preserving_attention_identity.py",
    HERE / "test_bilin18_observed_model_facade.py",
    HERE.parents[1] / "jacclust" / "tt_model.py",
)
BATCH = 4
SEQUENCE = 256
LAYERS = 18


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    header = json.dumps({
        "shape": list(contiguous.shape), "dtype": str(contiguous.dtype),
    }, sort_keys=True).encode()
    return hashlib.sha256(header + contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def deterministic_tokens(device: torch.device) -> torch.Tensor:
    """A role-free production-shaped fixture spanning common and boundary IDs."""

    tokens = (
        torch.arange(BATCH * SEQUENCE, dtype=torch.long).reshape(BATCH, SEQUENCE)
        * 7919 + 104729
    ) % facade.TOKENIZER_VOCAB
    tokens[0, 0] = 0
    tokens[0, 1] = facade.TOKENIZER_VOCAB - 1
    facade.validate_tokens(tokens)
    return tokens.to(device)


class AttentionNativePoison:
    """Replace all native attention objects, then restore object identity exactly."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.blocks = tuple(model.transformer.h)
        self.snapshots = {site: block.attn for site, block in enumerate(self.blocks)}
        self.installed: dict[int, torch.nn.Module] = {}
        self.calls = {site: 0 for site in range(len(self.blocks))}
        self.restored = False
        self.inert = False

    @contextmanager
    def scope(self) -> Iterator[None]:
        if self.installed:
            raise RuntimeError("attention poison is one-use")
        calls = self.calls

        class ForbiddenAttention(torch.nn.Module):
            def __init__(self, site: int) -> None:
                super().__init__()
                self.site = site

            def forward(self, *_args, **_kwargs):
                calls[self.site] += 1
                raise RuntimeError(f"literal native attention{self.site} call is forbidden")

        for site, block in enumerate(self.blocks):
            installed = ForbiddenAttention(site)
            self.installed[site] = installed
            block.attn = installed
        try:
            yield
        finally:
            for site, block in enumerate(self.blocks):
                block.attn = self.snapshots[site]
            self.restored = all(
                block.attn is self.snapshots[site]
                for site, block in enumerate(self.blocks)
            )
            self.inert = self.restored and all(
                block.attn is not self.installed[site]
                for site, block in enumerate(self.blocks)
            )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _ce(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    targets = torch.roll(tokens, shifts=-1, dims=1)
    return float(F.cross_entropy(
        logits[:, 64:].reshape(-1, logits.shape[-1]),
        targets[:, 64:].reshape(-1),
    ))


def bank_tensor_manifest(bank: TensorAttentionBank) -> dict[str, object]:
    layers: dict[str, object] = {}
    for site, program in enumerate(bank.programs):
        tensors: dict[str, object] = {}
        for name, projection in program.projections.items():
            for field in ("weight", "input_factor", "output_factor"):
                value = getattr(projection, field)
                if value is not None:
                    tensors[f"{name}.{field}"] = {
                        "shape": list(value.shape), "dtype": str(value.dtype),
                        "numel": value.numel(),
                        "bytes": value.numel() * value.element_size(),
                        "sha256": tensor_sha256(value),
                    }
        for name in ("lamb", "inv_freq"):
            value = getattr(program, name)
            tensors[name] = {
                "shape": list(value.shape), "dtype": str(value.dtype),
                "numel": value.numel(), "bytes": value.numel() * value.element_size(),
                "sha256": tensor_sha256(value),
            }
        layers[str(site)] = tensors
    return layers


def native_and_program_storage_disjoint(
    model: torch.nn.Module, bank: TensorAttentionBank,
) -> bool:
    native = {
        tensor.untyped_storage().data_ptr()
        for block in model.transformer.h
        for tensor in tuple(block.attn.parameters()) + tuple(block.attn.buffers())
        if tensor.numel()
    }
    program = {
        tensor.untyped_storage().data_ptr()
        for tensor in bank.buffers()
        if tensor.numel()
    }
    return native.isdisjoint(program)


@torch.no_grad()
def run_identity() -> dict[str, Any]:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    tokens = deterministic_tokens(device)

    bank = TensorAttentionBank.from_model(
        model, ranks={name: None for name in PROJECTION_NAMES},
    )
    if len(bank.programs) != LAYERS:
        raise RuntimeError("attention program stack is incomplete")
    storage_disjoint = native_and_program_storage_disjoint(model, bank)
    if not storage_disjoint:
        raise RuntimeError("attention bank aliases native attention storage")

    native_calls = {site: 0 for site in range(LAYERS)}
    native_writes: dict[int, torch.Tensor] = {}
    native_buses: dict[int, torch.Tensor] = {}
    native_states: dict[int, torch.Tensor] = {}
    native_mlp_calls = {site: 0 for site in range(LAYERS)}

    def native_attention(event: facade.AttentionEvent):
        expected = sum(native_calls.values())
        if event.site != expected:
            raise RuntimeError("native attention dispatch order changed")
        native_calls[event.site] += 1
        native_states[event.site] = event.state.detach().cpu().clone()
        write, bus = event.block.attn(event.state, event.first_value)
        native_writes[event.site] = write.detach().cpu().clone()
        native_buses[event.site] = bus.detach().cpu().clone()
        return write, bus

    def reference_mlp(event: facade.EarlyMLPEvent):
        native_mlp_calls[event.site] += 1
        return event.block.mlp(event.state)

    native_logits = facade.forward_with_dispatch(
        model, tokens, native_attention, reference_mlp,
    )
    if tuple(native_calls.items()) != tuple((site, 1) for site in range(LAYERS)):
        raise RuntimeError("native reference did not dispatch every attention site once")

    offline_write_diffs: dict[str, float] = {}
    offline_bus_diffs: dict[str, float] = {}
    root_bus = native_buses[0].to(device)
    for site, program in enumerate(bank.programs):
        incoming = None if site == 0 else root_bus
        write, bus = program(native_states[site].to(device), incoming)
        offline_write_diffs[str(site)] = _max_abs(
            write, native_writes[site].to(device),
        )
        offline_bus_diffs[str(site)] = _max_abs(bus, native_buses[site].to(device))

    program_calls = {site: 0 for site in range(LAYERS)}
    write_diffs: dict[str, float] = {}
    bus_diffs: dict[str, float] = {}

    program_mlp_calls = {site: 0 for site in range(LAYERS)}

    def program_mlp(event: facade.EarlyMLPEvent):
        program_mlp_calls[event.site] += 1
        return event.block.mlp(event.state)

    poison = AttentionNativePoison(model)
    blocks = tuple(model.transformer.h)
    with bank.begin(blocks) as transaction:
        def program_attention(event: facade.AttentionEvent):
            write, bus = transaction(event)
            program_calls[event.site] += 1
            write_diffs[str(event.site)] = _max_abs(
                write, native_writes[event.site].to(write.device),
            )
            bus_diffs[str(event.site)] = _max_abs(
                bus, native_buses[event.site].to(bus.device),
            )
            return write, bus

        with poison.scope():
            program_logits = facade.forward_with_dispatch(
                model, tokens, program_attention, program_mlp,
            )
    bank_closure = transaction.closure
    expected_calls = tuple((site, 1) for site in range(LAYERS))
    if tuple(program_calls.items()) != expected_calls:
        raise RuntimeError("program did not dispatch every attention site once")
    if tuple(native_mlp_calls.items()) != expected_calls or tuple(
        program_mlp_calls.items()
    ) != expected_calls:
        raise RuntimeError("native MLP policy was not identical between arms")
    if any(poison.calls.values()) or not poison.restored or not poison.inert:
        raise RuntimeError("native attention poison did not close cleanly")

    logit_max_abs = _max_abs(program_logits, native_logits)
    exact_logits = bool(torch.equal(program_logits, native_logits))
    exact_writes = all(value == 0.0 for value in write_diffs.values())
    exact_buses = all(value == 0.0 for value in bus_diffs.values())
    exact_offline_writes = all(value == 0.0 for value in offline_write_diffs.values())
    exact_offline_buses = all(value == 0.0 for value in offline_bus_diffs.values())
    native_ce = _ce(native_logits.float(), tokens)
    program_ce = _ce(program_logits.float(), tokens)
    costs = bank.cost_receipt()
    total_values = int(costs["total_stored_values"])
    identity_pass = (
        exact_logits and exact_writes and exact_buses
        and exact_offline_writes and exact_offline_buses
        and tuple(program_logits.shape) == (BATCH, SEQUENCE, facade.LOGIT_VOCAB)
        and native_ce == program_ce and total_values > 0 and storage_disjoint
        and bank_closure.ordered and bank_closure.block_identity
        and bank_closure.first_value_identity and bank_closure.closed
    )
    payload = {
        "status": "pass" if identity_pass else "fail",
        "scope": "role-free deterministic numerical and zero-native-call identity gate",
        "checkpoint": asdict(checkpoint),
        "source_sha256": {str(path.relative_to(HERE.parents[1])): sha256_file(path)
                          for path in SOURCE_FILES},
        "fixture": {
            "construction": "((arange(4*256)*7919+104729)%50257), with IDs 0 and 50256 pinned",
            "shape": list(tokens.shape),
            "sha256": tensor_sha256(tokens),
            "opens_data_role": False,
        },
        "execution": {
            "native_reference_calls": native_calls,
            "program_calls": program_calls,
            "native_reference_mlp_calls": native_mlp_calls,
            "program_arm_mlp_calls": program_mlp_calls,
            "literal_native_attention_calls_during_program": poison.calls,
            "poison_restored": poison.restored,
            "poison_inert": poison.inert,
            "native_mlp_policy": "all 18 MLPs remain native in both identity arms",
            "bank_closure": asdict(bank_closure),
            "native_program_storage_disjoint": storage_disjoint,
            "native_attention_objects_replaced_during_program": True,
        },
        "numerics": {
            "exact_logits": exact_logits,
            "exact_attention_writes": exact_writes,
            "exact_first_value_bus": exact_buses,
            "exact_same_input_attention_writes": exact_offline_writes,
            "exact_same_input_first_value_bus": exact_offline_buses,
            "logit_max_abs": logit_max_abs,
            "write_max_abs_by_site": write_diffs,
            "bus_max_abs_by_site": bus_diffs,
            "same_input_write_max_abs_by_site": offline_write_diffs,
            "same_input_bus_max_abs_by_site": offline_bus_diffs,
            "native_ce": native_ce,
            "program_ce": program_ce,
            "native_logits_sha256": tensor_sha256(native_logits),
            "program_logits_sha256": tensor_sha256(program_logits),
        },
        "cost": {
            **costs,
            "note": "dense identity price for the replaced attention stack only; native MLP, embedding, and unembedding remain outside this component receipt",
        },
        "tensor_manifest": bank_tensor_manifest(bank),
        "runtime_s": round(time.time() - started, 1),
    }
    if not identity_pass:
        raise RuntimeError(f"tensor-preserving attention identity failed: {payload['numerics']}")
    return payload


def main() -> None:
    payload = run_identity()
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=1) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, OUTPUT)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        "TENSOR ATTENTION IDENTITY PASS | exact logits/writes/v1 | "
        f"0 native attention calls | {payload['cost']['total_stored_values']:,} values | "
        f"{payload['runtime_s']}s",
        flush=True,
    )
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
