"""Reusable arm dispatcher, ledgers, and scorer for MLP2 CMR v1 validation."""

from __future__ import annotations

from dataclasses import asdict, fields
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
from mlp2_cmr_v1_physical_program import (
    PhysicalRetainedBilinearMLP, zero_mlp_write,
)
import mlp2_cmr_v1_suffix_math as suffix_math
import mlp2_cmr_v1_validation_statistics as statistics


SITE = 2
DOCUMENTS = 192
BATCH = 4
CALLS = 48
PREFIXES = (48, 96, 192)
PHYSICAL_ARMS = ("SUFFIX", "LOCAL", "RMS", "MASS", "DERANGED", "HASH_RANDOM")
SIGNED_T = {
    "minus_0p25": -0.25,
    "minus_0p10": -0.10,
    "plus_0p10": 0.10,
    "plus_0p25": 0.25,
}
ALL_ARMS = ("NATIVE", "ZERO", *PHYSICAL_ARMS, *SIGNED_T)
FLOAT_FIELDS = (
    "native_nll_sum", "candidate_nll_sum", "teacher_kl_sum",
    "centered_logit_sse", "native_centered_logit_energy", "raw_logit_sse",
)
INTEGER_FIELDS = (
    "native_correct_count", "candidate_correct_count",
    "native_top1_agreement_count",
)
DERANGEMENT_SEED = 2026090209
HASH_RANDOM_SEED = 20260829


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def support_sha256(value: torch.Tensor) -> str:
    value = torch.as_tensor(value, dtype=torch.long).cpu().contiguous()
    return tensor_sha256(value)


def validate_supports(supports: Mapping[str, torch.Tensor]) -> dict[str, str]:
    if set(supports) != set(PHYSICAL_ARMS):
        raise ValueError("physical support family changed")
    hashes = {}
    for arm in PHYSICAL_ARMS:
        support = supports[arm]
        if not torch.is_tensor(support) or support.shape != (512,) or (
            support.dtype != torch.long or torch.unique(support).numel() != 512
            or int(support.min()) < 0 or int(support.max()) >= 4608
        ):
            raise ValueError(f"{arm} support is malformed")
        hashes[arm] = support_sha256(support)
    return hashes


def build_physical_programs(
    native: torch.nn.Module, mean: torch.Tensor,
    supports: Mapping[str, torch.Tensor],
) -> tuple[dict[str, PhysicalRetainedBilinearMLP], dict[str, dict[str, int]]]:
    validate_supports(supports)
    if mean.shape != (4608,) or mean.dtype != torch.float64 or not bool(
        torch.isfinite(mean).all()
    ) or native.Left.weight.shape != (4608, 1152) or native.Right.weight.shape != (
        4608, 1152
    ) or native.Down.weight.shape != (1152, 4608) or native.Down_bias.shape != (
        1152,
    ) or any(value.dtype != torch.bfloat16 for value in (
        native.Left.weight, native.Right.weight, native.Down.weight, native.Down_bias,
    )):
        raise RuntimeError("production MLP2 coefficients or FIT mean changed")
    programs = {
        arm: PhysicalRetainedBilinearMLP.from_native(native, mean, supports[arm])
        for arm in PHYSICAL_ARMS
    }
    receipts = {arm: program.receipt_dict() for arm, program in programs.items()}
    expected = {
        "input_width": 1152,
        "output_width": 1152,
        "native_products": 4608,
        "retained_products": 512,
        "stored_scalar_values": 1_770_624,
        "support_index_values": 512,
        "bilinear_products_per_token": 512,
        "native_mlp_calls_per_forward": 0,
    }
    if any(receipt != expected for receipt in receipts.values()):
        raise RuntimeError("physical MLP2 price receipt changed")
    return programs, receipts


def physical_materialization_replay(
    programs: Mapping[str, PhysicalRetainedBilinearMLP], *, device: torch.device,
) -> dict[str, float | bool]:
    generator = torch.Generator(device="cpu").manual_seed(2026082908)
    state = torch.randn(2, 3, 1152, generator=generator).to(
        device=device, dtype=torch.bfloat16,
    )
    errors = {}
    with torch.inference_mode():
        for arm in PHYSICAL_ARMS:
            program = programs[arm]
            actual = program(state)
            reference = F.linear(
                F.linear(state, program.left) * F.linear(state, program.right),
                program.down, program.folded_bias,
            )
            errors[arm] = float((actual - reference).abs().max())
    return {
        "maximum_absolute_error": max(errors.values()),
        "bit_exact": all(value == 0 for value in errors.values()),
        "per_arm_maximum_absolute_error": errors,
    }


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _select_top(scores: torch.Tensor, count: int) -> tuple[int, ...]:
    if scores.ndim != 1 or scores.dtype != torch.float64 or not bool(
        torch.isfinite(scores).all()
    ) or type(count) is not int or not 0 < count <= scores.numel():
        raise ValueError("selector replay scores changed")
    values = scores.tolist()
    return tuple(sorted(
        range(len(values)), key=lambda index: (-values[index], index),
    )[:count])


def selector_gauge_permutation_replay(
    mean: torch.Tensor, variance: torch.Tensor, down: torch.Tensor,
    suffix_score: torch.Tensor, suffix_support: torch.Tensor,
) -> dict[str, Any]:
    """Recompute the frozen selector gauge audit from live coefficients."""

    hidden = 4608
    if mean.shape != (hidden,) or variance.shape != (hidden,) or down.shape != (
        1152, hidden,
    ) or suffix_score.shape != (hidden,) or suffix_support.shape != (512,) or any(
        value.dtype != torch.float64
        for value in (mean, variance, down, suffix_score)
    ) or suffix_support.dtype != torch.long:
        raise ValueError("selector gauge replay inputs changed")
    std, orientation, permutation = suffix_math.canonical_derangement(
        mean, variance, down, DERANGEMENT_SEED,
    )
    random_support = suffix_math.canonical_hash_random_support(
        mean, variance, down, 512, HASH_RANDOM_SEED,
    )
    pattern = torch.tensor(
        [2.0, -4.0, 0.5, -0.25, 8.0, -2.0, 0.125], dtype=torch.float64,
    )
    scales = pattern.repeat((hidden + len(pattern) - 1) // len(pattern))[:hidden]
    std2, orientation2, permutation2 = suffix_math.canonical_derangement(
        mean * scales, variance * scales.square(), down / scales,
        DERANGEMENT_SEED,
    )
    random2 = suffix_math.canonical_hash_random_support(
        mean * scales, variance * scales.square(), down / scales, 512,
        HASH_RANDOM_SEED,
    )
    canonical_down = down * std[None, :] * orientation[None, :]
    canonical_down2 = down / scales * std2[None, :] * orientation2[None, :]
    general_generator = torch.Generator().manual_seed(2026090211)
    general_scales = torch.exp(
        3 * torch.randn(hidden, generator=general_generator, dtype=torch.float64)
    )
    general_scales[::2].neg_()
    general_std, general_orientation, _ = suffix_math.canonical_derangement(
        mean * general_scales, variance * general_scales.square(),
        down / general_scales, DERANGEMENT_SEED,
    )
    general_canonical_down = (
        down / general_scales * general_std[None, :] * general_orientation[None, :]
    )
    generator = torch.Generator().manual_seed(2026090210)
    order = tuple(torch.randperm(hidden, generator=generator).tolist())
    index = torch.tensor(order)
    _, _, permuted_derangement = suffix_math.canonical_derangement(
        mean[index], variance[index], down[:, index], DERANGEMENT_SEED,
    )
    permuted_random = suffix_math.canonical_hash_random_support(
        mean[index], variance[index], down[:, index], 512, HASH_RANDOM_SEED,
    )
    permuted_suffix = _select_top(suffix_score[index], 512)
    return {
        "dyadic_reciprocal": {
            "derangement_exact": permutation2 == permutation,
            "hash_random_exact": random2 == random_support,
            "canonical_down_max_abs_error": float(
                (canonical_down2 - canonical_down).abs().max()
            ),
        },
        "general_reciprocal_functional": {
            "canonical_down_max_relative_error": float((
                (general_canonical_down - canonical_down).abs()
                / canonical_down.abs().clamp_min(1e-30)
            ).max()),
            "hash_byte_replay_required": False,
        },
        "channel_permutation": {
            "derangement_equivariant": permuted_derangement
                == suffix_math.mapped_permutation(permutation, order),
            "hash_random_equivariant": {
                order[i] for i in permuted_random
            } == set(random_support),
            "suffix_support_equivariant": {
                order[i] for i in permuted_suffix
            } == set(int(value) for value in suffix_support.tolist()),
        },
        "derangement_sha256": _canonical_sha256(list(permutation)),
        "hash_random_support_sha256": _canonical_sha256(list(random_support)),
    }


def physical_gauge_permutation_replay(
    programs: Mapping[str, PhysicalRetainedBilinearMLP],
) -> dict[str, Any]:
    """Replay product/Down gauge and channel permutation on owned programs.

    This uses CPU float64 copies of the actually materialized buffers. It therefore
    checks the compiled function itself, independently of the selector audit and of
    the native MLP implementation.
    """

    if set(programs) != set(PHYSICAL_ARMS):
        raise ValueError("physical gauge replay program family changed")
    first = programs[PHYSICAL_ARMS[0]]
    if any(
        program.retained_products != 512
        or program.input_width != first.input_width
        or program.output_width != first.output_width
        or program.native_products != first.native_products
        for program in programs.values()
    ):
        raise ValueError("physical gauge replay topology changed")
    generator = torch.Generator().manual_seed(2026082917)
    state = torch.randn(
        2, 3, first.input_width, generator=generator, dtype=torch.float64,
    )
    permutation = torch.randperm(512, generator=generator)
    dyadic_pattern = torch.tensor(
        [2.0, -4.0, 0.5, -0.25, 8.0, -2.0, 0.125], dtype=torch.float64,
    )
    dyadic = dyadic_pattern.repeat(74)[:512]
    general = torch.exp(torch.randn(512, generator=generator, dtype=torch.float64))
    general[::2].neg_()
    rows: dict[str, dict[str, float | bool]] = {}
    for arm in PHYSICAL_ARMS:
        program = programs[arm]
        left = program.left.detach().cpu().double()
        right = program.right.detach().cpu().double()
        down = program.down.detach().cpu().double()
        bias = program.folded_bias.detach().cpu().double()
        support = program.support.detach().cpu()
        original = PhysicalRetainedBilinearMLP(
            left, right, down, bias, support,
            native_products=program.native_products,
        )
        permuted = PhysicalRetainedBilinearMLP(
            left[permutation], right[permutation], down[:, permutation], bias,
            support[permutation], native_products=program.native_products,
        )
        dyadic_program = PhysicalRetainedBilinearMLP(
            left * dyadic[:, None], right, down / dyadic[None, :], bias,
            support, native_products=program.native_products,
        )
        general_program = PhysicalRetainedBilinearMLP(
            left * general[:, None], right, down / general[None, :], bias,
            support, native_products=program.native_products,
        )
        with torch.inference_mode():
            reference = original(state)
            permuted_write = permuted(state)
            dyadic_write = dyadic_program(state)
            general_write = general_program(state)
        denominator = reference.abs().clamp_min(1e-30)
        rows[arm] = {
            "permutation_max_absolute_error": float(
                (permuted_write - reference).abs().max()
            ),
            "dyadic_max_relative_error": float(
                ((dyadic_write - reference).abs() / denominator).max()
            ),
            "general_max_relative_error": float(
                ((general_write - reference).abs() / denominator).max()
            ),
        }
        rows[arm]["passed"] = (
            rows[arm]["permutation_max_absolute_error"] == 0.0
            and rows[arm]["dyadic_max_relative_error"] <= 5e-12
            and rows[arm]["general_max_relative_error"] <= 5e-12
        )
    return {
        "currency": "CPU float64 copies of materialized owned buffers",
        "tolerance": 5e-12,
        "per_arm": rows,
        "passed": all(row["passed"] is True for row in rows.values()),
    }


def new_call_ledger() -> dict[str, dict[str, Any]]:
    return {
        arm: {
            "forward_calls": 0,
            "forward_returns": 0,
            "attention_calls_by_site": [0] * 18,
            "native_mlp_calls_by_site": [0] * 18,
            "physical_mlp2_calls": 0,
            "zero_mlp2_calls": 0,
            "diagnostic_full_product_evaluations": 0,
        } for arm in ALL_ARMS
    }


def call_ledger_passes(ledger: Mapping[str, Mapping[str, Any]]) -> bool:
    if set(ledger) != set(ALL_ARMS):
        return False
    for arm in ALL_ARMS:
        row = ledger[arm]
        expected_native = [CALLS] * 18
        if arm == "ZERO" or arm in PHYSICAL_ARMS:
            expected_native[SITE] = 0
        expected_physical = CALLS if arm in PHYSICAL_ARMS or arm in SIGNED_T else 0
        expected_zero = CALLS if arm == "ZERO" else 0
        expected_diagnostic = CALLS if arm == "NATIVE" else 0
        if row != {
            "forward_calls": CALLS,
            "forward_returns": CALLS,
            "attention_calls_by_site": [CALLS] * 18,
            "native_mlp_calls_by_site": expected_native,
            "physical_mlp2_calls": expected_physical,
            "zero_mlp2_calls": expected_zero,
            "diagnostic_full_product_evaluations": expected_diagnostic,
        }:
            return False
    return True


@torch.no_grad()
def additivity_batch(
    product: torch.Tensor, down: torch.Tensor, mean: torch.Tensor,
    retained_support: torch.Tensor, eligible: torch.Tensor,
) -> torch.Tensor:
    hidden = product.shape[-1] if product.ndim == 3 else -1
    if hidden != 4608 or down.shape != (1152, 4608) or mean.shape != (
        4608,
    ) or retained_support.shape != (512,) or eligible.shape != product.shape[:2] or (
        eligible.device.type != "cpu" or eligible.dtype != torch.bool
    ):
        raise ValueError("additivity diagnostic inputs are malformed")
    omitted_mask = torch.ones(hidden, dtype=torch.bool, device=product.device)
    omitted_mask[retained_support.to(product.device)] = False
    omitted = torch.nonzero(omitted_mask, as_tuple=False).flatten()
    centered = product.float()[..., omitted] - mean.to(
        device=product.device, dtype=torch.float32,
    )[omitted]
    outgoing = down.to(device=product.device, dtype=torch.float32)[:, omitted]
    joint = F.linear(centered, outgoing).square().sum(-1)
    additive = (
        centered.square() * outgoing.square().sum(0)[None, None, :]
    ).sum(-1)
    output = torch.zeros(product.shape[0], 3, dtype=torch.float64)
    for document in range(product.shape[0]):
        mask = eligible[document].to(product.device)
        output[document, 0] = int(eligible[document].sum())
        output[document, 1] = joint[document, mask].detach().cpu().double().sum()
        output[document, 2] = additive[document, mask].detach().cpu().double().sum()
    if not bool(torch.isfinite(output).all()) or bool((output < 0).any()):
        raise RuntimeError("additivity sufficient statistics are nonfinite")
    return output


def forward_arm(
    model: torch.nn.Module, tokens: torch.Tensor, arm: str,
    programs: Mapping[str, PhysicalRetainedBilinearMLP],
    ledger: dict[str, dict[str, Any]], *,
    native_mlp2_observer: Callable[[torch.Tensor, torch.nn.Module], None] | None = None,
) -> torch.Tensor:
    if arm not in ALL_ARMS or set(programs) != set(PHYSICAL_ARMS) or set(ledger) != (
        set(ALL_ARMS)
    ):
        raise ValueError("validation arm dispatcher inputs changed")
    row = ledger[arm]

    def attention(event: facade.AttentionEvent):
        row["attention_calls_by_site"][event.site] += 1
        return event.block.attn(event.state, event.first_value)

    def mlp(event: facade.EarlyMLPEvent):
        if event.site != SITE:
            row["native_mlp_calls_by_site"][event.site] += 1
            return event.block.mlp(event.state)
        if arm == "ZERO":
            row["zero_mlp2_calls"] += 1
            return zero_mlp_write(event.state)
        if arm in PHYSICAL_ARMS:
            row["physical_mlp2_calls"] += 1
            return programs[arm](event.state)
        if arm in SIGNED_T:
            row["native_mlp_calls_by_site"][SITE] += 1
            native_write = event.block.mlp(event.state)
            row["physical_mlp2_calls"] += 1
            suffix_write = programs["SUFFIX"](event.state)
            return native_write + SIGNED_T[arm] * (suffix_write - native_write)
        row["native_mlp_calls_by_site"][SITE] += 1
        if native_mlp2_observer is not None:
            row["diagnostic_full_product_evaluations"] += 1
            product = event.block.mlp.Left(event.state) * event.block.mlp.Right(event.state)
            native_mlp2_observer(product, event.block.mlp)
        return event.block.mlp(event.state)

    row["forward_calls"] += 1
    logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
    row["forward_returns"] += 1
    return logits


def pack_ledgers(
    ledgers: Mapping[str, Mapping[int, Mapping[str, statistics.CellSums]]],
) -> dict[str, Any]:
    if set(ledgers) != set(ALL_ARMS):
        raise ValueError("arm ledger family changed")
    arms, cells = tuple(ALL_ARMS), tuple(statistics.CELL_NAMES)
    counts = torch.zeros(len(arms), DOCUMENTS, len(cells), dtype=torch.long)
    floats = torch.zeros(
        len(arms), DOCUMENTS, len(cells), len(FLOAT_FIELDS), dtype=torch.float64,
    )
    integers = torch.zeros(
        len(arms), DOCUMENTS, len(cells), len(INTEGER_FIELDS), dtype=torch.long,
    )
    support_hashes: list[list[str]] | None = None
    for arm_index, arm in enumerate(arms):
        if tuple(sorted(ledgers[arm])) != tuple(range(DOCUMENTS)):
            raise ValueError("arm ledger document support changed")
        arm_support = []
        for document in range(DOCUMENTS):
            if set(ledgers[arm][document]) != set(cells):
                raise ValueError("arm ledger cell support changed")
            row_support = []
            for cell_index, cell in enumerate(cells):
                value = ledgers[arm][document][cell]
                statistics._validate_cell_sum(value)
                counts[arm_index, document, cell_index] = value.count
                floats[arm_index, document, cell_index] = torch.tensor([
                    getattr(value, field) for field in FLOAT_FIELDS
                ], dtype=torch.float64)
                integers[arm_index, document, cell_index] = torch.tensor([
                    getattr(value, field) for field in INTEGER_FIELDS
                ], dtype=torch.long)
                row_support.append(value.support_sha256)
            arm_support.append(row_support)
        if support_hashes is None:
            support_hashes = arm_support
        elif arm_support != support_hashes:
            raise ValueError("arms do not share exact cell support hashes")
    return {
        "arm_names": arms,
        "cell_names": cells,
        "float_fields": FLOAT_FIELDS,
        "integer_fields": INTEGER_FIELDS,
        "counts": counts,
        "float_sums": floats,
        "integer_sums": integers,
        "support_hashes": support_hashes,
    }


def unpack_ledgers(bundle: Mapping[str, Any]) -> dict[str, dict[int, dict[str, statistics.CellSums]]]:
    expected = {
        "arm_names", "cell_names", "float_fields", "integer_fields", "counts",
        "float_sums", "integer_sums", "support_hashes",
    }
    if set(bundle) != expected or tuple(bundle["arm_names"]) != ALL_ARMS or tuple(
        bundle["cell_names"]
    ) != statistics.CELL_NAMES or tuple(bundle["float_fields"]) != FLOAT_FIELDS or tuple(
        bundle["integer_fields"]
    ) != INTEGER_FIELDS:
        raise RuntimeError("packed arm ledger schema changed")
    counts, floats, integers = (
        bundle["counts"], bundle["float_sums"], bundle["integer_sums"],
    )
    expected_counts = (len(ALL_ARMS), DOCUMENTS, len(statistics.CELL_NAMES))
    if not torch.is_tensor(counts) or counts.shape != expected_counts or counts.dtype != (
        torch.long
    ) or not torch.is_tensor(floats) or floats.shape != (*expected_counts, len(FLOAT_FIELDS)) or (
        floats.dtype != torch.float64 or not bool(torch.isfinite(floats).all())
    ) or not torch.is_tensor(integers) or integers.shape != (
        *expected_counts, len(INTEGER_FIELDS)
    ) or integers.dtype != torch.long:
        raise RuntimeError("packed arm ledger tensors changed")
    supports = bundle["support_hashes"]
    if not isinstance(supports, list) or len(supports) != DOCUMENTS or any(
        not isinstance(row, list) or len(row) != len(statistics.CELL_NAMES)
        or any(not isinstance(value, str) or len(value) != 64 for value in row)
        for row in supports
    ):
        raise RuntimeError("packed arm support hashes changed")
    output = {}
    for arm_index, arm in enumerate(ALL_ARMS):
        output[arm] = {}
        for document in range(DOCUMENTS):
            output[arm][document] = {}
            for cell_index, cell in enumerate(statistics.CELL_NAMES):
                values = {
                    field: float(floats[arm_index, document, cell_index, index])
                    for index, field in enumerate(FLOAT_FIELDS)
                }
                values.update({
                    field: int(integers[arm_index, document, cell_index, index])
                    for index, field in enumerate(INTEGER_FIELDS)
                })
                value = statistics.CellSums(
                    count=int(counts[arm_index, document, cell_index]),
                    support_sha256=supports[document][cell_index], **values,
                )
                statistics._validate_cell_sum(value)
                output[arm][document][cell] = value
    return output


def pack_geometry(
    batches: Sequence[Mapping[str, Mapping[str, statistics.PairSums]]],
) -> torch.Tensor:
    tensor = torch.empty(
        len(batches), len(statistics.CELL_NAMES), len(statistics.GEOMETRY_PAIRS), 3,
        dtype=torch.float64,
    )
    for batch_index, batch in enumerate(batches):
        if set(batch) != set(statistics.CELL_NAMES):
            raise ValueError("geometry cell schema changed")
        for cell_index, cell in enumerate(statistics.CELL_NAMES):
            if set(batch[cell]) != set(statistics.GEOMETRY_PAIRS):
                raise ValueError("geometry pair schema changed")
            for pair_index, pair in enumerate(statistics.GEOMETRY_PAIRS):
                value = batch[cell][pair]
                tensor[batch_index, cell_index, pair_index] = torch.tensor([
                    value.dot, value.left_norm2, value.right_norm2,
                ], dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()) or bool((tensor[..., 1:] < 0).any()):
        raise RuntimeError("packed geometry is malformed")
    return tensor


def unpack_geometry(value: torch.Tensor) -> list[dict[str, dict[str, statistics.PairSums]]]:
    expected = (CALLS, len(statistics.CELL_NAMES), len(statistics.GEOMETRY_PAIRS), 3)
    if not torch.is_tensor(value) or value.shape != expected or value.dtype != (
        torch.float64
    ) or not bool(torch.isfinite(value).all()) or bool((value[..., 1:] < 0).any()):
        raise RuntimeError("packed geometry tensor changed")
    batches = []
    for batch_index in range(CALLS):
        batches.append({
            cell: {
                pair: statistics.PairSums(*(
                    float(item) for item in value[batch_index, cell_index, pair_index]
                ))
                for pair_index, pair in enumerate(statistics.GEOMETRY_PAIRS)
            } for cell_index, cell in enumerate(statistics.CELL_NAMES)
        })
    return batches


def score_validation_bundle(
    bundle: Mapping[str, Any], *, protocol_audits: Mapping[str, bool],
) -> dict[str, Any]:
    expected = {
        "schema", "ledgers", "margin_counts", "margin_support_counts",
        "epsilon_grid", "geometry", "additivity",
    }
    if set(bundle) != expected or bundle.get("schema") != "mlp2_cmr_v1_validation_ledger":
        raise RuntimeError("validation bundle schema changed")
    ledgers = unpack_ledgers(bundle["ledgers"])
    margin_counts = bundle["margin_counts"]
    margin_support = bundle["margin_support_counts"]
    epsilon = bundle["epsilon_grid"]
    additivity = bundle["additivity"]
    if not torch.is_tensor(additivity) or additivity.shape != (DOCUMENTS, 3) or (
        additivity.dtype != torch.float64 or not bool(torch.isfinite(additivity).all())
        or bool((additivity < 0).any())
    ):
        raise RuntimeError("additivity ledger changed")
    summaries = {
        arm: {
            str(prefix): statistics.summarize_arm(
                ledgers[arm], prefix_documents=prefix,
                include_raw_sufficient_statistics=False,
            ) for prefix in PREFIXES
        } for arm in ALL_ARMS
    }
    margins = {
        arm: {
            str(prefix): statistics.margin_certificate_curve(
                ledgers[arm], margin_counts, margin_support, epsilon,
                prefix_documents=prefix,
            ) for prefix in PREFIXES
        } for arm in ALL_ARMS
    }
    bootstrap = statistics.simultaneous_relative_kl_bootstrap({
        arm: ledgers[arm] for arm in ("SUFFIX", *statistics.EQUAL_PRICE_CONTROLS)
    })
    geometry = statistics.summarize_signed_geometry(unpack_geometry(bundle["geometry"]))
    total_count = float(additivity[:, 0].sum())
    joint, additive = float(additivity[:, 1].sum()), float(additivity[:, 2].sum())
    if total_count <= 0 or additive <= 0:
        raise RuntimeError("additivity diagnostic has zero support")
    additivity_summary = {
        "count": int(total_count),
        "J": joint / total_count,
        "A": additive / total_count,
        "J_over_A": joint / additive,
        "material_disagreement": not 0.90 <= joint / additive <= 1.10,
    }
    suffix = summaries["SUFFIX"]["192"]["cells"]["all_scored"]
    cell_gate = all(
        row.get("empty") is False and row["candidate_minus_native_ce"] <= 0.02
        for row in summaries["SUFFIX"]["192"]["cells"].values()
    )
    geometry_gate = all(
        geometry["all_scored"][pair]["nonzero"] is True
        and geometry["all_scored"][pair]["cosine"] >= 0.90
        for pair in statistics.GEOMETRY_PAIRS
    )
    gates = {
        "simultaneous_relative_kl_lcb_at_least_0p05": (
            bootstrap["simultaneous_lower_bound"] >= 0.05
        ),
        "absolute_delta_ce_at_most_0p02": abs(suffix["candidate_minus_native_ce"]) <= 0.02,
        "teacher_kl_at_most_0p02": suffix["teacher_kl"] <= 0.02,
        "centered_logit_nrmse_at_most_0p10": suffix["centered_logit_nrmse"] <= 0.10,
        "top1_agreement_at_least_0p90": suffix["native_top1_agreement"] >= 0.90,
        "margin_certificate_at_least_0p90": margins["SUFFIX"]["192"][
            "maximum_bound"
        ] >= 0.90,
        "every_registered_cell_nonempty_and_ce_harm_at_most_0p02": cell_gate,
        "signed_direction_cosines_at_least_0p90": geometry_gate,
        **dict(protocol_audits),
    }
    if set(protocol_audits) != {
        "exact_price_and_support_replay", "gauge_and_permutation_replay",
        "physical_materialization_replay", "physical_call_ledger_replay",
        "float32_cpu_float64_precision_audit",
    }:
        raise RuntimeError("validation protocol-audit gate family changed")
    return {
        "prefix_summaries": summaries,
        "margin_certificates": margins,
        "shared_document_bootstrap": bootstrap,
        "signed_geometry": geometry,
        "singleton_additivity": additivity_summary,
        "gates": gates,
        "validation_passed": all(gates.values()),
        "replication_authorized": all(gates.values()),
    }
