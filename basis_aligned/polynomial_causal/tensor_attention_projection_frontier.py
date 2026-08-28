#!/usr/bin/env python3
"""Bottom-up executable frontier for tensor-preserving bilin18 attention.

This runner implements the discovery protocol frozen in
``TENSOR_PRESERVING_ATTENTION_PREREGISTRATION.md``.  It never imports a
historical hook/compiler runner.  Fitting is accelerated by concatenating the
five registered arm trajectories into one batch at each target depth; the
covariance and program state remain separate for every arm.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
from tensor_preserving_attention import (
    PROJECTION_NAMES,
    QK_NAMES,
    SharedInputLinearBank,
    StoredLinear,
    TensorAttentionBank,
    TensorPreservingSquaredAttention,
)
from tensor_preserving_attention_identity import AttentionNativePoison


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
OUTPUT = HERE / "tensor_attention_projection_frontier_results.json"
FIT_ROWS = BQ / ".rowcache/fineweb_n480_skip80.pt"
MASK_ROWS = BQ / ".rowcache/fineweb_n96_skip80.pt"
EVAL_ROLES = {
    "heldout_skip7000": BQ / ".rowcache/fineweb_n192_skip7000.pt",
    "replication_skip11000": BQ / ".rowcache/fineweb_n192_skip11000.pt",
}
CONSTANTS = BQ / "opt_ablation_consts_all.pt"
SOURCE_FILES = (
    Path(__file__).resolve(),
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_attention_identity.py",
    HERE / "bilin18_observed_model_facade.py",
    HERE / "TENSOR_PRESERVING_ATTENTION_PREREGISTRATION.md",
    HERE / "test_tensor_attention_projection_frontier.py",
)

D = 1152
T = 256
LAYERS = 18
BATCH = 4
SCORE_START = 64
RIDGE = 1e-3


@dataclass(frozen=True)
class ArmSpec:
    qk_rank: int | None
    value_rank: int | None
    shared_qk: bool = False


ARM_SPECS: Mapping[str, ArmSpec] = {
    "routing384": ArmSpec(qk_rank=384, value_rank=None),
    "value384": ArmSpec(qk_rank=None, value_rank=384),
    "joint384": ArmSpec(qk_rank=384, value_rank=384),
    "joint512": ArmSpec(qk_rank=512, value_rank=512),
    "shared_qk384": ArmSpec(qk_rank=384, value_rank=None, shared_qk=True),
}


class _CapturedTarget(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path, expected: int) -> torch.Tensor:
    value = torch.load(path, map_location="cpu")
    rows = value["rows"] if isinstance(value, dict) else value
    rows = rows[:, : T + 1].long().contiguous()
    if tuple(rows.shape) != (expected, T + 1):
        raise RuntimeError(f"row role {path.name} changed shape")
    if int(rows.min()) < 0 or int(rows.max()) >= facade.TOKENIZER_VOCAB:
        raise RuntimeError(f"row role {path.name} left tokenizer support")
    return rows


def activation_weighted_linear(
    covariance: torch.Tensor, weight: torch.Tensor, rank: int,
    *, ridge_fraction: float = RIDGE,
) -> StoredLinear:
    """Replay the registered ridge-map then ordinary-SVD estimator.

    ``weight`` is stored in torch Linear orientation.  The fitted coefficient
    maps row vectors as ``x @ coefficient``; StoredLinear factors translate
    that orientation without materializing a dense truncated map.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be square")
    width = covariance.shape[0]
    if weight.shape != (width, width) or not 0 < rank < width:
        raise ValueError("compressed projection topology changed")
    covariance = covariance.double()
    scale = torch.diag(covariance).mean()
    regularized = covariance + ridge_fraction * scale * torch.eye(
        width, dtype=covariance.dtype, device=covariance.device,
    )
    coefficient = torch.linalg.solve(
        regularized, covariance @ weight.double().T,
    )
    u, singular, vh = torch.linalg.svd(coefficient, full_matrices=False)
    return StoredLinear(
        input_factor=u[:, :rank].T.to(weight.dtype),
        output_factor=(vh[:rank].T * singular[:rank]).to(weight.dtype),
    )


def _registered_coefficient(
    covariance: torch.Tensor, weight: torch.Tensor,
    *, ridge_fraction: float = RIDGE,
) -> torch.Tensor:
    width = covariance.shape[0]
    scale = torch.diag(covariance).mean()
    regularized = covariance + ridge_fraction * scale * torch.eye(
        width, dtype=covariance.dtype, device=covariance.device,
    )
    return torch.linalg.solve(regularized, covariance @ weight.double().T)


def shared_activation_weighted_bank(
    covariance: torch.Tensor, weights: Mapping[str, torch.Tensor], rank: int,
) -> SharedInputLinearBank:
    """Optimal shared rank-r coefficient stack in the activation metric.

    If ``C_j`` are the four registered ridge coefficients, this minimizes

        sum_j || A^(1/2) (C_j - E D_j) ||_F^2

    over one encoder ``E`` and four typed decoders ``D_j``.  It is the weighted
    Eckart--Young solution: concatenate the whitened maps, take their leading
    common left singular subspace, then unwhiten the encoder.
    """

    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1] or not (
        0 < rank < covariance.shape[0]
    ) or set(weights) != set(QK_NAMES):
        raise ValueError("shared QK fit is malformed")
    covariance = (covariance.double() + covariance.double().T) * 0.5
    width = covariance.shape[0]
    if any(weight.shape != (width, width) for weight in weights.values()):
        raise ValueError("shared QK source topology changed")
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    floor = torch.finfo(covariance.dtype).eps * width * eigenvalues[-1].clamp_min(1.0)
    supported = eigenvalues.clamp_min(floor)
    square_root = (eigenvectors * supported.sqrt()) @ eigenvectors.T
    inverse_square_root = (eigenvectors * supported.rsqrt()) @ eigenvectors.T
    coefficients = {
        name: _registered_coefficient(covariance, weights[name]) for name in QK_NAMES
    }
    whitened = square_root @ torch.cat(
        [coefficients[name] for name in QK_NAMES], dim=1,
    )
    # Left singular vectors from the smaller symmetric Gram matrix avoid
    # materializing a 1152 x 4608 Vh.
    gram = whitened @ whitened.T
    _, left = torch.linalg.eigh((gram + gram.T) * 0.5)
    left = left[:, -rank:].flip(1)
    encoder = inverse_square_root @ left
    decoded = left.T @ whitened
    output_factors = {
        name: decoded[:, index * width : (index + 1) * width].T.float()
        for index, name in enumerate(QK_NAMES)
    }
    return SharedInputLinearBank(encoder.T.float(), output_factors)


def compile_site(
    native: torch.nn.Module, covariance: torch.Tensor, spec: ArmSpec,
) -> TensorPreservingSquaredAttention:
    sources = {
        "q": native.c_q.weight.detach(),
        "k": native.c_k.weight.detach(),
        "q2": native.c_q2.weight.detach(),
        "k2": native.c_k2.weight.detach(),
        "v": native.c_v.weight.detach(),
        "proj": native.c_proj.weight.detach(),
    }
    projections: dict[str, StoredLinear] = {}
    shared = None
    if spec.shared_qk:
        if spec.qk_rank is None:
            raise ValueError("shared QK arm has no rank")
        shared = shared_activation_weighted_bank(
            covariance, {name: sources[name] for name in QK_NAMES}, spec.qk_rank,
        )
    else:
        for name in QK_NAMES:
            projections[name] = (
                StoredLinear.from_weight(sources[name])
                if spec.qk_rank is None
                else activation_weighted_linear(covariance, sources[name], spec.qk_rank)
            )
    projections["v"] = (
        StoredLinear.from_weight(sources["v"])
        if spec.value_rank is None
        else activation_weighted_linear(covariance, sources["v"], spec.value_rank)
    )
    projections["proj"] = StoredLinear.from_weight(sources["proj"])
    return TensorPreservingSquaredAttention(
        projections,
        lamb=native.lamb.detach(),
        inv_freq=native.rotary.inv_freq.detach(),
        n_head=int(native.n_head),
        shared_qk=shared,
    ).to(device=sources["q"].device, dtype=sources["q"].dtype)


@torch.no_grad()
def compile_arms_jointly(
    model: torch.nn.Module, fit_rows: torch.Tensor,
) -> tuple[dict[str, TensorAttentionBank], dict[str, Any]]:
    """Fit distinct bottom-up trajectories with one concatenated model pass."""

    names = tuple(ARM_SPECS)
    programs: dict[str, list[TensorPreservingSquaredAttention]] = {
        name: [] for name in names
    }
    per_site: dict[str, Any] = {}
    device = next(model.parameters()).device
    blocks = tuple(model.transformer.h)

    for target_site in range(LAYERS):
        covariance = {
            name: torch.zeros(D, D, dtype=torch.float64, device=device)
            for name in names
        }
        positions = {name: 0 for name in names}

        for start in range(0, len(fit_rows), BATCH):
            base = fit_rows[start : start + BATCH, :T].to(device)
            if len(base) != BATCH:
                raise RuntimeError("fit role is not divisible by the production batch")
            tokens = torch.cat([base for _ in names], dim=0)

            def attention_dispatch(event: facade.AttentionEvent):
                if event.site < target_site:
                    writes = []
                    buses = []
                    for arm_index, name in enumerate(names):
                        sl = slice(arm_index * BATCH, (arm_index + 1) * BATCH)
                        incoming = None if event.first_value is None else event.first_value[sl]
                        write, bus = programs[name][event.site](event.state[sl], incoming)
                        writes.append(write)
                        buses.append(bus)
                    return torch.cat(writes, dim=0), torch.cat(buses, dim=0)
                if event.site != target_site:
                    raise RuntimeError("fit dispatcher passed the target site")
                for arm_index, name in enumerate(names):
                    sl = slice(arm_index * BATCH, (arm_index + 1) * BATCH)
                    state = event.state[sl].reshape(-1, D).double()
                    covariance[name].addmm_(state.T, state)
                    positions[name] += state.shape[0]
                raise _CapturedTarget

            def native_mlp(event: facade.EarlyMLPEvent):
                return event.block.mlp(event.state)

            try:
                facade.forward_with_dispatch(
                    model, tokens, attention_dispatch, native_mlp,
                    require_production=False,
                )
            except _CapturedTarget:
                pass
            else:
                raise RuntimeError("fit forward did not stop at its target site")

        layer_receipt: dict[str, Any] = {}
        for name in names:
            if positions[name] != len(fit_rows) * T:
                raise RuntimeError("fit covariance position count changed")
            normalized = covariance[name] / positions[name]
            program = compile_site(blocks[target_site].attn, normalized, ARM_SPECS[name])
            programs[name].append(program)
            layer_receipt[name] = {
                "positions": positions[name],
                "trace": float(torch.trace(normalized)),
                "program_cost": asdict(program.cost_receipt()),
            }
        per_site[str(target_site)] = layer_receipt
        print(f"fit site {target_site + 1}/{LAYERS}", flush=True)

    return (
        {name: TensorAttentionBank(programs[name]) for name in names},
        per_site,
    )


def seen_token_mask(mask_rows: torch.Tensor, device: torch.device) -> torch.Tensor:
    counts = torch.zeros(facade.TOKENIZER_VOCAB, dtype=torch.long, device=device)
    values = mask_rows[:, :T].to(device).reshape(-1)
    counts.index_add_(0, values, torch.ones_like(values))
    return counts > 0


@torch.no_grad()
def score_logits(
    logits: torch.Tensor, rows: torch.Tensor, seen: torch.Tensor,
) -> tuple[float, int]:
    targets = rows[:, 1 : T + 1].to(logits.device)
    inputs = rows[:, :T].to(logits.device)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none",
    ).reshape_as(targets)[:, SCORE_START:]
    covered = seen[inputs[:, SCORE_START:]]
    return float(losses[covered].sum()), int(covered.sum())


@torch.no_grad()
def score_native(
    model: torch.nn.Module, rows: torch.Tensor, seen: torch.Tensor,
) -> dict[str, Any]:
    total = 0.0
    count = 0
    calls = [0] * LAYERS
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        tokens = batch[:, :T].to(next(model.parameters()).device)

        def attention(event: facade.AttentionEvent):
            calls[event.site] += 1
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            return event.block.mlp(event.state)

        loss, n = score_logits(
            facade.forward_with_dispatch(model, tokens, attention, mlp), batch, seen,
        )
        total += loss
        count += n
    return {"ce": total / count, "positions": count, "attention_calls": calls}


@torch.no_grad()
def score_constant(
    model: torch.nn.Module, rows: torch.Tensor, seen: torch.Tensor,
    constants: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    total = 0.0
    count = 0
    for start in range(0, len(rows), BATCH):
        batch = rows[start : start + BATCH]
        tokens = batch[:, :T].to(next(model.parameters()).device)

        def attention(event: facade.AttentionEvent):
            write = constants[f"attn{event.site}"].to(
                device=event.state.device, dtype=event.state.dtype,
            ).reshape(1, 1, D).expand_as(event.state)
            if event.site == 0:
                bus = event.block.attn.c_v(event.state).view(
                    BATCH, T, int(event.block.attn.n_head), D // int(event.block.attn.n_head),
                )
            else:
                bus = event.first_value
            return write, bus

        def mlp(event: facade.EarlyMLPEvent):
            return event.block.mlp(event.state)

        loss, n = score_logits(
            facade.forward_with_dispatch(model, tokens, attention, mlp), batch, seen,
        )
        total += loss
        count += n
    return {
        "ce": total / count,
        "positions": count,
        "diagnostic_native_projection_calls_per_batch": 1,
        "note": "constant-write denominator mints only the shape-correct site0 value bus",
    }


@torch.no_grad()
def score_bank(
    model: torch.nn.Module, bank: TensorAttentionBank,
    rows: torch.Tensor, seen: torch.Tensor,
) -> dict[str, Any]:
    total = 0.0
    count = 0
    calls = [0] * LAYERS
    closures = []
    poison = AttentionNativePoison(model)
    blocks = tuple(model.transformer.h)
    with poison.scope():
        for start in range(0, len(rows), BATCH):
            batch = rows[start : start + BATCH]
            tokens = batch[:, :T].to(next(model.parameters()).device)
            with bank.begin(blocks) as transaction:
                def attention(event: facade.AttentionEvent):
                    calls[event.site] += 1
                    return transaction(event)

                def mlp(event: facade.EarlyMLPEvent):
                    return event.block.mlp(event.state)

                logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            closures.append(asdict(transaction.closure))
            loss, n = score_logits(logits, batch, seen)
            total += loss
            count += n
    expected = len(rows) // BATCH
    if calls != [expected] * LAYERS or any(poison.calls.values()) or not (
        poison.restored and poison.inert
    ):
        raise RuntimeError("executable attention call ledger failed")
    if not all(
        row["ordered"] and row["block_identity"] and row["first_value_identity"]
        and row["closed"] for row in closures
    ):
        raise RuntimeError("attention transaction closure failed")
    return {
        "ce": total / count,
        "positions": count,
        "program_calls": calls,
        "literal_native_attention_calls": sum(poison.calls.values()),
        "transactions_closed": len(closures),
    }


def bank_price(bank: TensorAttentionBank) -> dict[str, Any]:
    receipt = bank.cost_receipt()
    buffers = tuple(bank.buffers())
    stored_bits = sum(value.numel() * value.element_size() * 8 for value in buffers)
    multiply_adds = sum(
        program.multiply_adds(batch=BATCH, sequence=T) for program in bank.programs
    )
    return {
        **receipt,
        "stored_bits": stored_bits,
        "multiply_adds_per_production_forward": multiply_adds,
        "buffer_values": sum(value.numel() for value in buffers),
    }


def publish_create_only(value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        written = 0
        while written < len(payload):
            advanced = os.write(descriptor, payload[written:])
            if advanced <= 0:
                raise OSError("create-only result publication made no progress")
            written += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("frontier result is create-only and already exists")
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    fit = load_rows(FIT_ROWS, 480)
    mask = load_rows(MASK_ROWS, 96)
    evaluations = {name: load_rows(path, 192) for name, path in EVAL_ROLES.items()}
    seen = seen_token_mask(mask, device)
    constants = torch.load(CONSTANTS, map_location="cpu")

    dense = TensorAttentionBank.from_model(
        model, ranks={name: None for name in PROJECTION_NAMES},
    )
    compiled, fit_receipt = compile_arms_jointly(model, fit)
    banks = {"dense_identity": dense, **compiled}
    prices = {name: bank_price(bank) for name, bank in banks.items()}
    if not prices["shared_qk384"]["total_stored_values"] < prices["routing384"][
        "total_stored_values"
    ]:
        raise RuntimeError("shared QK arm did not reduce complete storage")

    roles: dict[str, Any] = {}
    for role, rows in evaluations.items():
        native = score_native(model, rows, seen)
        constant = score_constant(model, rows, seen, constants)
        stake = constant["ce"] - native["ce"]
        if stake <= 0:
            raise RuntimeError("constant-attention denominator has no positive CE stake")
        arms = {}
        for name, bank in banks.items():
            measured = score_bank(model, bank, rows, seen)
            measured["harm_vs_native"] = measured["ce"] - native["ce"]
            measured["normalized_recovery"] = (
                constant["ce"] - measured["ce"]
            ) / stake
            arms[name] = measured
            print(
                f"{role} {name}: CE {measured['ce']:.6f} "
                f"R {measured['normalized_recovery']:.4f}", flush=True,
            )
        roles[role] = {
            "native": native,
            "constant_attention": constant,
            "stake": stake,
            "arms": arms,
        }

    primary = roles["heldout_skip7000"]
    harms = {name: row["harm_vs_native"] for name, row in primary["arms"].items()}
    composition_margin = harms["joint384"] - (
        harms["routing384"] + harms["value384"]
    )
    executable = {
        name: (
            primary["arms"][name]["normalized_recovery"] >= 0.90
            and prices[name]["stored_bits"] < prices["dense_identity"]["stored_bits"]
            and prices[name]["multiply_adds_per_production_forward"]
            < prices["dense_identity"]["multiply_adds_per_production_forward"]
            and prices[name]["total_input_support"]
            and primary["arms"][name]["literal_native_attention_calls"] == 0
        )
        for name in ARM_SPECS
    }
    controls = {
        "dense_identity_max_abs_ce_error": max(
            abs(role["arms"]["dense_identity"]["ce"] - role["native"]["ce"])
            for role in roles.values()
        ),
        "joint384_composition_margin_nat": composition_margin,
        "joint384_composition_pass": composition_margin <= 0.10,
        "executable_compression": executable,
        "replication_same_pass_labels": {
            name: (
                primary["arms"][name]["normalized_recovery"] >= 0.90
            ) == (
                roles["replication_skip11000"]["arms"][name]["normalized_recovery"] >= 0.90
            )
            for name in ARM_SPECS
        },
    }
    if controls["dense_identity_max_abs_ce_error"] > 1e-6:
        raise RuntimeError("dense executable identity failed on corpus evaluation")

    result = {
        "status": "discovery_only",
        "protocol": "tensor-preserving attention projection frontier",
        "checkpoint": asdict(checkpoint),
        "arms": {name: asdict(spec) for name, spec in ARM_SPECS.items()},
        "roles": roles,
        "prices": prices,
        "controls": controls,
        "fit": {
            "rows": len(fit),
            "positions_per_arm_per_site": len(fit) * T,
            "ridge_fraction": RIDGE,
            "joint_batch_optimization": (
                "five distinct arm trajectories concatenated only along batch; "
                "separate covariance and bottom-up programs"
            ),
            "sites": fit_receipt,
        },
        "provenance": {
            "sources": {str(path): sha256_file(path) for path in SOURCE_FILES},
            "data": {
                "fit": {"path": str(FIT_ROWS), "sha256": sha256_file(FIT_ROWS)},
                "mask": {"path": str(MASK_ROWS), "sha256": sha256_file(MASK_ROWS)},
                **{
                    name: {"path": str(path), "sha256": sha256_file(path)}
                    for name, path in EVAL_ROLES.items()
                },
                "constants": {"path": str(CONSTANTS), "sha256": sha256_file(CONSTANTS)},
            },
        },
        "runtime_s": time.time() - started,
    }
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps(outcome["controls"], indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT} ({outcome['runtime_s']:.1f}s)", flush=True)
