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
from pathlib import Path
import time
from types import MethodType
from typing import Any, Iterator

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from tensor_preserving_attention import (
    PROJECTION_NAMES, TensorPreservingSquaredAttention,
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
    """Forbid literal attention forwards and restore instance dispatch exactly."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.modules = tuple(block.attn for block in model.transformer.h)
        self.snapshots: dict[int, tuple[bool, Any]] = {}
        self.installed: dict[int, Any] = {}
        self.calls = {site: 0 for site in range(len(self.modules))}
        self.restored = False
        self.inert = False

    @contextmanager
    def scope(self) -> Iterator[None]:
        if self.snapshots:
            raise RuntimeError("attention poison is one-use")
        for site, module in enumerate(self.modules):
            had_instance_forward = "forward" in module.__dict__
            previous = module.__dict__.get("forward")
            self.snapshots[site] = (had_instance_forward, previous)

            def poison(_module, *_args, _site=site, **_kwargs):
                self.calls[_site] += 1
                raise RuntimeError(f"literal native attention{_site} call is forbidden")

            installed = MethodType(poison, module)
            self.installed[site] = installed
            module.forward = installed
        try:
            yield
        finally:
            for site, module in enumerate(self.modules):
                had_instance_forward, previous = self.snapshots[site]
                if had_instance_forward:
                    module.__dict__["forward"] = previous
                else:
                    module.__dict__.pop("forward", None)
            self.restored = all(
                (("forward" in module.__dict__) == had)
                and (not had or module.__dict__.get("forward") is previous)
                for site, module in enumerate(self.modules)
                for had, previous in (self.snapshots[site],)
            )
            self.inert = self.restored and all(
                module.__dict__.get("forward") is not self.installed[site]
                for site, module in enumerate(self.modules)
            )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def _ce(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    targets = torch.roll(tokens, shifts=-1, dims=1)
    return float(F.cross_entropy(
        logits[:, 64:].reshape(-1, logits.shape[-1]),
        targets[:, 64:].reshape(-1),
    ))


@torch.no_grad()
def run_identity() -> dict[str, Any]:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    tokens = deterministic_tokens(device)

    programs = torch.nn.ModuleList([
        TensorPreservingSquaredAttention.from_native(
            block.attn, ranks={name: None for name in PROJECTION_NAMES},
        )
        for block in model.transformer.h
    ])
    if len(programs) != LAYERS:
        raise RuntimeError("attention program stack is incomplete")

    native_calls = {site: 0 for site in range(LAYERS)}
    native_writes: dict[int, torch.Tensor] = {}
    native_buses: dict[int, torch.Tensor] = {}

    def native_attention(event: facade.AttentionEvent):
        expected = sum(native_calls.values())
        if event.site != expected:
            raise RuntimeError("native attention dispatch order changed")
        native_calls[event.site] += 1
        write, bus = event.block.attn(event.state, event.first_value)
        native_writes[event.site] = write.detach().cpu().clone()
        native_buses[event.site] = bus.detach().cpu().clone()
        return write, bus

    def native_mlp(event: facade.EarlyMLPEvent):
        return event.block.mlp(event.state)

    native_logits = facade.forward_with_dispatch(
        model, tokens, native_attention, native_mlp,
    )
    if tuple(native_calls.items()) != tuple((site, 1) for site in range(LAYERS)):
        raise RuntimeError("native reference did not dispatch every attention site once")

    program_calls = {site: 0 for site in range(LAYERS)}
    write_diffs: dict[str, float] = {}
    bus_diffs: dict[str, float] = {}

    def program_attention(event: facade.AttentionEvent):
        expected = sum(program_calls.values())
        if event.site != expected:
            raise RuntimeError("program attention dispatch order changed")
        program_calls[event.site] += 1
        write, bus = programs[event.site](event.state, event.first_value)
        write_diffs[str(event.site)] = _max_abs(write, native_writes[event.site].to(write.device))
        bus_diffs[str(event.site)] = _max_abs(bus, native_buses[event.site].to(bus.device))
        return write, bus

    poison = AttentionNativePoison(model)
    with poison.scope():
        program_logits = facade.forward_with_dispatch(
            model, tokens, program_attention, native_mlp,
        )
    if tuple(program_calls.items()) != tuple((site, 1) for site in range(LAYERS)):
        raise RuntimeError("program did not dispatch every attention site once")
    if any(poison.calls.values()) or not poison.restored or not poison.inert:
        raise RuntimeError("native attention poison did not close cleanly")

    logit_max_abs = _max_abs(program_logits, native_logits)
    exact_logits = bool(torch.equal(program_logits, native_logits))
    exact_writes = all(value == 0.0 for value in write_diffs.values())
    exact_buses = all(value == 0.0 for value in bus_diffs.values())
    native_ce = _ce(native_logits.float(), tokens)
    program_ce = _ce(program_logits.float(), tokens)
    costs = [asdict(program.cost_receipt()) for program in programs]
    total_values = sum(int(receipt["total_stored_values"]) for receipt in costs)
    identity_pass = (
        exact_logits and exact_writes and exact_buses
        and tuple(program_logits.shape) == (BATCH, SEQUENCE, facade.LOGIT_VOCAB)
        and native_ce == program_ce and total_values > 0
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
            "literal_native_attention_calls_during_program": poison.calls,
            "poison_restored": poison.restored,
            "poison_inert": poison.inert,
            "native_mlp_policy": "all 18 MLPs remain native in both identity arms",
        },
        "numerics": {
            "exact_logits": exact_logits,
            "exact_attention_writes": exact_writes,
            "exact_first_value_bus": exact_buses,
            "logit_max_abs": logit_max_abs,
            "write_max_abs_by_site": write_diffs,
            "bus_max_abs_by_site": bus_diffs,
            "native_ce": native_ce,
            "program_ce": program_ce,
            "native_logits_sha256": tensor_sha256(native_logits),
            "program_logits_sha256": tensor_sha256(program_logits),
        },
        "cost": {
            "per_layer": costs,
            "total_stored_values": total_values,
            "token_table_values": sum(int(x["token_table_values"]) for x in costs),
            "native_calls_per_forward": sum(int(x["native_calls_per_forward"]) for x in costs),
            "total_input_support": all(bool(x["total_input_support"]) for x in costs),
            "note": "dense identity price for the replaced attention stack only; native MLP, embedding, and unembedding remain outside this component receipt",
        },
        "runtime_s": round(time.time() - started, 1),
    }
    if not identity_pass:
        raise RuntimeError(f"tensor-preserving attention identity failed: {payload['numerics']}")
    return payload


def main() -> None:
    payload = run_identity()
    OUTPUT.write_text(json.dumps(payload, indent=1) + "\n")
    print(
        "TENSOR ATTENTION IDENTITY PASS | exact logits/writes/v1 | "
        f"0 native attention calls | {payload['cost']['total_stored_values']:,} values | "
        f"{payload['runtime_s']}s",
        flush=True,
    )
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
