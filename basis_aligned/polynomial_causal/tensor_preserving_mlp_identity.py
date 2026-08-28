#!/usr/bin/env python3
"""Role-free source-closed identity gate for the owned bilinear MLP bank."""

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

import bilin18_observed_model_facade as facade
from tensor_preserving_attention_identity import deterministic_tokens, tensor_sha256
from tensor_preserving_mlp import TensorMLPBank


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_preserving_mlp_identity_results.json"
PREREG = HERE / "TENSOR_PRESERVING_MLP_IDENTITY_PREREGISTRATION.md"
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_preserving_mlp.py",
    HERE / "tensor_preserving_attention_identity.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_tensor_preserving_mlp.py",
    HERE / "test_tensor_preserving_mlp_identity.py",
    HERE.parents[1] / "jacclust" / "tt_model.py",
)
LAYERS = 18


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MLPNativePoison:
    def __init__(self, model: torch.nn.Module) -> None:
        self.blocks = tuple(model.transformer.h)
        self.snapshots = {site: block.mlp for site, block in enumerate(self.blocks)}
        self.installed: dict[int, torch.nn.Module] = {}
        self.calls = {site: 0 for site in range(len(self.blocks))}
        self.restored = False
        self.inert = False

    @contextmanager
    def scope(self) -> Iterator[None]:
        if self.installed:
            raise RuntimeError("MLP poison is one-use")
        calls = self.calls

        class ForbiddenMLP(torch.nn.Module):
            def __init__(self, site: int) -> None:
                super().__init__()
                self.site = site

            def forward(self, *_args, **_kwargs):
                calls[self.site] += 1
                raise RuntimeError(f"literal native MLP{self.site} call is forbidden")

        for site, block in enumerate(self.blocks):
            forbidden = ForbiddenMLP(site)
            self.installed[site] = forbidden
            block.mlp = forbidden
        try:
            yield
        finally:
            for site, block in enumerate(self.blocks):
                block.mlp = self.snapshots[site]
            self.restored = all(
                block.mlp is self.snapshots[site] for site, block in enumerate(self.blocks)
            )
            self.inert = self.restored and all(
                block.mlp is not self.installed[site] for site, block in enumerate(self.blocks)
            )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max())


def storage_disjoint(model: torch.nn.Module, bank: TensorMLPBank) -> bool:
    native = {
        value.untyped_storage().data_ptr()
        for block in model.transformer.h
        for value in tuple(block.mlp.parameters()) + tuple(block.mlp.buffers())
        if value.numel()
    }
    program = {
        value.untyped_storage().data_ptr() for value in bank.buffers() if value.numel()
    }
    return native.isdisjoint(program)


def bank_manifest(bank: TensorMLPBank) -> dict[str, Any]:
    layers = {}
    for site, program in enumerate(bank.programs):
        tensors = {}
        for name in ("left", "right", "down"):
            projection = getattr(program, name)
            for field in ("weight", "input_factor", "output_factor"):
                value = getattr(projection, field)
                if value is not None:
                    tensors[f"{name}.{field}"] = {
                        "shape": list(value.shape), "dtype": str(value.dtype),
                        "numel": value.numel(), "bytes": value.numel() * value.element_size(),
                        "sha256": tensor_sha256(value),
                    }
        tensors["down_bias"] = {
            "shape": list(program.down_bias.shape), "dtype": str(program.down_bias.dtype),
            "numel": program.down_bias.numel(),
            "bytes": program.down_bias.numel() * program.down_bias.element_size(),
            "sha256": tensor_sha256(program.down_bias),
        }
        layers[str(site)] = tensors
    return layers


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("identity publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.no_grad()
def run_identity() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("MLP identity result is create-only and already exists")
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    tokens = deterministic_tokens(device)
    bank = TensorMLPBank.from_model(model)
    if not storage_disjoint(model, bank):
        raise RuntimeError("MLP bank aliases native storage")

    native_mlp_calls = {site: 0 for site in range(LAYERS)}
    native_attn_calls = {site: 0 for site in range(LAYERS)}
    native_states = {}
    native_writes = {}

    def native_attention(event: facade.AttentionEvent):
        native_attn_calls[event.site] += 1
        return event.block.attn(event.state, event.first_value)

    def native_mlp(event: facade.EarlyMLPEvent):
        native_mlp_calls[event.site] += 1
        native_states[event.site] = event.state.detach().cpu().clone()
        write = event.block.mlp(event.state)
        native_writes[event.site] = write.detach().cpu().clone()
        return write

    native_logits = facade.forward_with_dispatch(
        model, tokens, native_attention, native_mlp,
    )
    expected = {site: 1 for site in range(LAYERS)}
    if native_mlp_calls != expected or native_attn_calls != expected:
        raise RuntimeError("native identity reference call ledger failed")

    offline_diffs = {}
    for site, program in enumerate(bank.programs):
        write = program(native_states[site].to(device))
        offline_diffs[str(site)] = _max_abs(write, native_writes[site].to(device))

    program_mlp_calls = {site: 0 for site in range(LAYERS)}
    program_attn_calls = {site: 0 for site in range(LAYERS)}
    trajectory_diffs = {}

    def program_attention(event: facade.AttentionEvent):
        program_attn_calls[event.site] += 1
        return event.block.attn(event.state, event.first_value)

    poison = MLPNativePoison(model)
    blocks = tuple(model.transformer.h)
    with bank.begin(blocks) as transaction:
        def program_mlp(event: facade.EarlyMLPEvent):
            program_mlp_calls[event.site] += 1
            write = transaction(event)
            trajectory_diffs[str(event.site)] = _max_abs(
                write, native_writes[event.site].to(write.device),
            )
            return write

        with poison.scope():
            program_logits = facade.forward_with_dispatch(
                model, tokens, program_attention, program_mlp,
            )
    closure = transaction.closure
    if program_mlp_calls != expected or program_attn_calls != expected or any(
        poison.calls.values()
    ) or not poison.restored or not poison.inert:
        raise RuntimeError("program identity execution ledger failed")

    logit_error = _max_abs(program_logits, native_logits)
    native_hash = tensor_sha256(native_logits)
    program_hash = tensor_sha256(program_logits)
    if max(offline_diffs.values()) != 0 or max(trajectory_diffs.values()) != 0 or (
        logit_error != 0 or native_hash != program_hash
    ):
        raise RuntimeError("dense MLP tensor program failed bitwise identity")
    targets = torch.roll(tokens, shifts=-1, dims=1)
    native_ce = float(torch.nn.functional.cross_entropy(
        native_logits[:, 64:].reshape(-1, native_logits.shape[-1]),
        targets[:, 64:].reshape(-1),
    ))
    program_ce = float(torch.nn.functional.cross_entropy(
        program_logits[:, 64:].reshape(-1, program_logits.shape[-1]),
        targets[:, 64:].reshape(-1),
    ))
    receipt = bank.cost_receipt()
    if receipt["total_stored_values"] != 286_675_200:
        raise RuntimeError("dense MLP storage denominator changed")

    result = {
        "status": "pass",
        "checkpoint": asdict(checkpoint),
        "fixture": {"shape": list(tokens.shape), "sha256": tensor_sha256(tokens)},
        "numerical": {
            "offline_write_max_abs": offline_diffs,
            "trajectory_write_max_abs": trajectory_diffs,
            "logit_max_abs": logit_error,
            "native_logit_sha256": native_hash,
            "program_logit_sha256": program_hash,
            "native_ce": native_ce,
            "program_ce": program_ce,
        },
        "execution": {
            "native_mlp_calls": native_mlp_calls,
            "program_mlp_calls": program_mlp_calls,
            "native_attention_calls": native_attn_calls,
            "program_attention_calls": program_attn_calls,
            "literal_native_mlp_calls_in_program_arm": sum(poison.calls.values()),
            "replacement_restored": poison.restored,
            "replacement_inert": poison.inert,
            "bank_closure": asdict(closure),
            "storage_disjoint": True,
        },
        "cost": receipt,
        "tensor_manifest": bank_manifest(bank),
        "provenance": {str(path): sha256_file(path) for path in SOURCES},
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run_identity()
    print(json.dumps({
        "status": outcome["status"], "numerical": outcome["numerical"],
        "cost": {"total_stored_values": outcome["cost"]["total_stored_values"]},
        "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
