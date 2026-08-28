#!/usr/bin/env python3
"""Matched weighted independent-versus-shared QK executable control."""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch

import bilin18_observed_model_facade as facade
import tensor_attention_projection_frontier as frontier
from tensor_preserving_attention import (
    QK_NAMES, SharedInputLinearBank, StoredLinear, TensorAttentionBank,
    TensorPreservingSquaredAttention,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "tensor_attention_shared_qk_matched_control_results.json"
PARENT = HERE / "tensor_attention_projection_frontier_results.json"
PREREG = HERE / "TENSOR_ATTENTION_SHARED_QK_MATCHED_CONTROL_PREREGISTRATION.md"
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_attention_projection_frontier.py",
    HERE / "tensor_preserving_attention.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "test_tensor_attention_shared_qk_matched_control.py",
)
RANK = 384
NAMES = ("independent_weighted384", "shared_qk384_replay")


class _CapturedTarget(RuntimeError):
    pass


def _covariance_roots(covariance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    covariance = (covariance.double() + covariance.double().T) * 0.5
    width = covariance.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = torch.finfo(covariance.dtype).eps * width * eigenvalues[-1].clamp_min(1.0)
    supported = eigenvalues.clamp_min(floor)
    square_root = (eigenvectors * supported.sqrt()) @ eigenvectors.T
    inverse_square_root = (eigenvectors * supported.rsqrt()) @ eigenvectors.T
    return square_root, inverse_square_root


def independent_activation_weighted_linear(
    covariance: torch.Tensor, weight: torch.Tensor, rank: int,
) -> StoredLinear:
    """Weighted Eckart--Young fit for one registered ridge coefficient."""

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or (
        weight.shape != covariance.shape or not 0 < rank < covariance.shape[0]
    ):
        raise ValueError("independent weighted projection topology changed")
    covariance = (covariance.double() + covariance.double().T) * 0.5
    square_root, inverse_square_root = _covariance_roots(covariance)
    coefficient = frontier._registered_coefficient(covariance, weight)
    whitened = square_root @ coefficient
    gram = whitened @ whitened.T
    _, left = torch.linalg.eigh((gram + gram.T) * 0.5)
    left = left[:, -rank:].flip(1)
    encoder = inverse_square_root @ left
    decoder = left.T @ whitened
    return StoredLinear(
        input_factor=encoder.T.to(weight.dtype),
        output_factor=decoder.T.to(weight.dtype),
    )


def compile_independent_site(
    native: torch.nn.Module, covariance: torch.Tensor,
) -> TensorPreservingSquaredAttention:
    projections = {
        name: independent_activation_weighted_linear(
            covariance, getattr(native, f"c_{name}").weight.detach(), RANK,
        )
        for name in QK_NAMES
    }
    projections["v"] = StoredLinear.from_weight(native.c_v.weight.detach())
    projections["proj"] = StoredLinear.from_weight(native.c_proj.weight.detach())
    return TensorPreservingSquaredAttention(
        projections, lamb=native.lamb.detach(),
        inv_freq=native.rotary.inv_freq.detach(), n_head=int(native.n_head),
    ).to(device=native.c_q.weight.device, dtype=native.c_q.weight.dtype)


@torch.no_grad()
def compile_banks(
    model: torch.nn.Module, fit_rows: torch.Tensor,
) -> tuple[dict[str, TensorAttentionBank], dict[str, Any]]:
    programs: dict[str, list[TensorPreservingSquaredAttention]] = {
        name: [] for name in NAMES
    }
    receipt: dict[str, Any] = {}
    device = next(model.parameters()).device
    blocks = tuple(model.transformer.h)
    batch = frontier.FIT_BATCH

    for target in range(frontier.LAYERS):
        covariance = {
            name: torch.zeros(frontier.D, frontier.D, dtype=torch.float64, device=device)
            for name in NAMES
        }
        positions = {name: 0 for name in NAMES}
        for start in range(0, len(fit_rows), batch):
            base = fit_rows[start : start + batch, : frontier.T].to(device)
            if len(base) != batch:
                raise RuntimeError("fit role is not divisible by fit batch")
            tokens = torch.cat((base, base), dim=0)

            def attention(event: facade.AttentionEvent):
                if event.site < target:
                    writes = []
                    buses = []
                    for index, name in enumerate(NAMES):
                        section = slice(index * batch, (index + 1) * batch)
                        incoming = None if event.first_value is None else event.first_value[section]
                        write, bus = programs[name][event.site](event.state[section], incoming)
                        writes.append(write)
                        buses.append(bus)
                    return torch.cat(writes), torch.cat(buses)
                if event.site != target:
                    raise RuntimeError("fit dispatcher passed its target")
                for index, name in enumerate(NAMES):
                    section = slice(index * batch, (index + 1) * batch)
                    state = event.state[section].reshape(-1, frontier.D).double()
                    covariance[name].addmm_(state.T, state)
                    positions[name] += state.shape[0]
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
                raise RuntimeError("fit target was not captured")

        expected = len(fit_rows) * frontier.T
        if set(positions.values()) != {expected}:
            raise RuntimeError("fit position ledger changed")
        normalized = {name: covariance[name] / expected for name in NAMES}
        independent = compile_independent_site(blocks[target].attn, normalized[NAMES[0]])
        shared = frontier.compile_site(
            blocks[target].attn, normalized[NAMES[1]],
            frontier.ArmSpec(qk_rank=RANK, value_rank=None, shared_qk=True),
        )
        programs[NAMES[0]].append(independent)
        programs[NAMES[1]].append(shared)
        receipt[str(target)] = {
            name: {
                "positions": positions[name],
                "covariance_trace": float(torch.trace(normalized[name])),
                "cost": asdict(program.cost_receipt()),
            }
            for name, program in ((NAMES[0], independent), (NAMES[1], shared))
        }
        print(f"fit site {target + 1}/{frontier.LAYERS}", flush=True)

    return {name: TensorAttentionBank(programs[name]) for name in NAMES}, receipt


def publish_create_only(value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("result publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("matched control result is create-only and already exists")
    started = time.time()
    parent = json.loads(PARENT.read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    fit = frontier.load_rows(frontier.FIT_ROWS, 480)
    mask = frontier.load_rows(frontier.MASK_ROWS, 96)
    evaluations = {
        name: frontier.load_rows(path, 192) for name, path in frontier.EVAL_ROLES.items()
    }
    seen = frontier.seen_token_mask(mask, device)
    constants = torch.load(frontier.CONSTANTS, map_location="cpu")
    banks, fit_receipt = compile_banks(model, fit)
    prices = {name: frontier.bank_price(bank) for name, bank in banks.items()}

    roles: dict[str, Any] = {}
    for role, rows in evaluations.items():
        native = frontier.score_native(model, rows, seen)
        constant = frontier.score_constant(model, rows, seen, constants)
        stake = constant["ce"] - native["ce"]
        arms = {}
        for name, bank in banks.items():
            measured = frontier.score_bank(model, bank, rows, seen)
            measured["harm_vs_native"] = measured["ce"] - native["ce"]
            measured["normalized_recovery"] = (
                constant["ce"] - measured["ce"]
            ) / stake
            arms[name] = measured
            print(
                f"{role} {name}: CE {measured['ce']:.6f} "
                f"R {measured['normalized_recovery']:.6f}", flush=True,
            )
        roles[role] = {"native": native, "constant": constant, "stake": stake, "arms": arms}

    parent_roles = parent["roles"]
    replay_errors = {
        role: abs(
            roles[role]["arms"][NAMES[1]]["normalized_recovery"]
            - parent_roles[role]["arms"]["shared_qk384"]["normalized_recovery"]
        )
        for role in roles
    }
    sharing_deltas = {
        role: (
            roles[role]["arms"][NAMES[1]]["normalized_recovery"]
            - roles[role]["arms"][NAMES[0]]["normalized_recovery"]
        )
        for role in roles
    }
    predictions = {
        "replay_within_0.003": all(value <= 0.003 for value in replay_errors.values()),
        "sharing_fidelity_free_within_0.005": all(
            value >= -0.005 for value in sharing_deltas.values()
        ),
        "shared_complete_cost_dominates": (
            all(value >= -0.005 for value in sharing_deltas.values())
            and prices[NAMES[1]]["total_stored_values"]
            < prices[NAMES[0]]["total_stored_values"]
            and prices[NAMES[1]]["multiply_adds_per_production_forward"]
            < prices[NAMES[0]]["multiply_adds_per_production_forward"]
        ),
    }
    result = {
        "status": "discovery_only_matched_control",
        "checkpoint": asdict(checkpoint),
        "rank": RANK,
        "roles": roles,
        "prices": prices,
        "comparisons": {
            "parent_shared_replay_abs_errors": replay_errors,
            "shared_minus_independent_recovery": sharing_deltas,
            "predictions": predictions,
        },
        "fit": fit_receipt,
        "provenance": {
            "parent": {"path": str(PARENT), "sha256": frontier.sha256_file(PARENT)},
            "sources": {str(path): frontier.sha256_file(path) for path in SOURCES},
            "roles": {
                "fit": frontier.sha256_file(frontier.FIT_ROWS),
                "mask": frontier.sha256_file(frontier.MASK_ROWS),
                **{name: frontier.sha256_file(path) for name, path in frontier.EVAL_ROLES.items()},
            },
        },
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(outcome["comparisons"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT} ({outcome['runtime_s']:.1f}s)", flush=True)
