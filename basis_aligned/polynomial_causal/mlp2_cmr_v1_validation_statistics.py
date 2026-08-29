"""Streaming CPU sufficient statistics for the frozen MLP2 CMR validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


FREQUENCY_CELL_NAMES = tuple(f"frequency_bin_{index}" for index in range(9))
COPY_CELL_NAMES = ("copy_positive", "repeat_negative", "nonrepeat")
CELL_NAMES = ("all_scored", *FREQUENCY_CELL_NAMES, *COPY_CELL_NAMES)
EQUAL_PRICE_CONTROLS = ("LOCAL", "RMS", "MASS", "DERANGED", "HASH_RANDOM")
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = "mlp2-cmr-v1-validation-document-bootstrap:0"
DOCUMENTS = 192
NLL_KL_PRECISION_TOLERANCE = 1e-4
SQUARED_ERROR_RELATIVE_TOLERANCE = 1e-4
SIGNED_KEYS = ("minus_0p25", "minus_0p10", "plus_0p10", "plus_0p25")
GEOMETRY_PAIRS = (
    "g0p10_vs_g0p25", "g0p10_vs_full", "plus0p10_vs_negminus0p10",
    "plus0p25_vs_negminus0p25",
)


@dataclass(frozen=True)
class CellSums:
    count: int
    native_nll_sum: float
    candidate_nll_sum: float
    teacher_kl_sum: float
    centered_logit_sse: float
    native_centered_logit_energy: float
    raw_logit_sse: float
    native_correct_count: int
    candidate_correct_count: int
    native_top1_agreement_count: int
    support_sha256: str


@dataclass(frozen=True)
class PairSums:
    dot: float
    left_norm2: float
    right_norm2: float


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _support_sha256(row: torch.Tensor, mask: torch.Tensor) -> str:
    selected = torch.nonzero(mask, as_tuple=False).flatten().long()
    digest = hashlib.sha256()
    digest.update(_tensor_sha256(row).encode())
    digest.update(_tensor_sha256(selected).encode())
    return digest.hexdigest()


def nearest_repeat_cells(
    rows: torch.Tensor, eligible: torch.Tensor, *, window: int = 128,
) -> dict[str, torch.Tensor]:
    if rows.ndim != 2 or rows.dtype != torch.long or eligible.shape != (
        rows.shape[0], rows.shape[1] - 1
    ) or eligible.dtype != torch.bool or not 0 < window <= eligible.shape[1]:
        raise ValueError("repeat-cell inputs are malformed")
    sequence = eligible.shape[1]
    inputs, targets = rows[:, :-1], rows[:, 1:]
    positions = torch.arange(sequence)
    source = torch.full_like(inputs, -1)
    expanded = positions.expand(rows.shape[0], -1)
    for distance in range(1, window + 1):
        candidate = (positions - distance).clamp_min(0)
        token = inputs.gather(1, candidate.expand(rows.shape[0], -1))
        choose = (
            (source < 0) & (positions >= distance).unsqueeze(0) & (inputs == token)
        )
        source[choose] = expanded[choose] - distance
    repeated = source >= 0
    successor = (source.clamp_min(0) + 1).clamp_max(sequence - 1)
    copied = inputs.gather(1, successor)
    positive = eligible & repeated & (targets == copied)
    negative = eligible & repeated & ~positive
    nonrepeat = eligible & ~repeated
    if not torch.equal(positive | negative | nonrepeat, eligible) or bool(
        (positive & negative).any() or (positive & nonrepeat).any()
        or (negative & nonrepeat).any()
    ):
        raise RuntimeError("repeat cells do not exactly partition eligible positions")
    return {
        "copy_positive": positive,
        "repeat_negative": negative,
        "nonrepeat": nonrepeat,
    }


def validation_cells(
    rows: torch.Tensor, eligible: torch.Tensor, fit_token_counts: torch.Tensor,
    frequency_boundaries: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if rows.ndim != 2 or rows.dtype != torch.long or eligible.shape != (
        rows.shape[0], rows.shape[1] - 1
    ) or eligible.dtype != torch.bool or fit_token_counts.ndim != 1 or (
        fit_token_counts.dtype != torch.long or frequency_boundaries.shape != (8,)
        or frequency_boundaries.dtype != torch.long
        or not torch.equal(
            frequency_boundaries, torch.tensor([1, 2, 4, 8, 16, 32, 64, 128]),
        )
    ):
        raise ValueError("validation cell inputs are malformed")
    targets = rows[:, 1:]
    if int(targets.min()) < 0 or int(targets.max()) >= fit_token_counts.numel():
        raise ValueError("validation target is outside the frozen frequency support")
    frequencies = fit_token_counts.index_select(0, targets.flatten()).reshape_as(targets)
    bins = torch.bucketize(frequencies, frequency_boundaries, right=True)
    cells = {"all_scored": eligible.clone()}
    cells.update({
        name: eligible & (bins == index)
        for index, name in enumerate(FREQUENCY_CELL_NAMES)
    })
    cells.update(nearest_repeat_cells(rows, eligible))
    if not torch.equal(
        torch.stack([cells[name] for name in FREQUENCY_CELL_NAMES]).sum(0),
        eligible.long(),
    ) or not torch.equal(
        torch.stack([cells[name] for name in COPY_CELL_NAMES]).sum(0),
        eligible.long(),
    ):
        raise RuntimeError("registered validation cells do not partition eligible support")
    return cells


def _validate_batch(
    native_logits: torch.Tensor, candidate_logits: torch.Tensor,
    rows: torch.Tensor, cells: Mapping[str, torch.Tensor],
    document_ordinals: Sequence[int],
) -> None:
    batch = rows.shape[0] if torch.is_tensor(rows) and rows.ndim == 2 else -1
    if rows.device.type != "cpu" or rows.dtype != torch.long or rows.shape[1:] != (
        257,
    ) or len(document_ordinals) != batch or len(set(document_ordinals)) != batch or (
        any(type(value) is not int or value < 0 for value in document_ordinals)
    ) or set(cells) != set(CELL_NAMES) or any(
        mask.device.type != "cpu" or mask.dtype != torch.bool
        or mask.shape != (batch, 256) for mask in cells.values()
    ):
        raise ValueError("streaming rows, cells, or document ordinals are malformed")
    for logits in (native_logits, candidate_logits):
        if not torch.is_tensor(logits) or logits.ndim != 3 or logits.shape[:2] != (
            batch, 256
        ) or not logits.is_floating_point() or not bool(torch.isfinite(logits).all()):
            raise ValueError("streaming logits are malformed")
    if native_logits.shape != candidate_logits.shape or native_logits.device != (
        candidate_logits.device
    ) or native_logits.shape[-1] <= int(rows.max()):
        raise ValueError("native and candidate logits are not aligned")


def reduce_arm_batch(
    native_logits: torch.Tensor, candidate_logits: torch.Tensor,
    rows: torch.Tensor, cells: Mapping[str, torch.Tensor],
    document_ordinals: Sequence[int],
) -> dict[int, dict[str, CellSums]]:
    """Reduce one arm immediately; no raw logits or per-token values are returned."""
    _validate_batch(native_logits, candidate_logits, rows, cells, document_ordinals)
    device = native_logits.device
    targets = rows[:, 1:].to(device)
    native = native_logits.float()
    candidate = candidate_logits.float()
    native_logp = F.log_softmax(native, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    native_nll = -native_logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    candidate_nll = -candidate_logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    teacher_kl = (
        native_logp.exp() * (native_logp - candidate_logp)
    ).sum(-1)
    if float(teacher_kl.min()) < -2e-5:
        raise RuntimeError("teacher KL is numerically negative")
    teacher_kl = teacher_kl.clamp_min(0)
    delta = candidate - native
    centered_delta = delta - delta.mean(-1, keepdim=True)
    native_centered = native - native.mean(-1, keepdim=True)
    centered_sse = centered_delta.square().sum(-1)
    native_energy = native_centered.square().sum(-1)
    raw_sse = delta.square().sum(-1)
    native_top1 = native.argmax(-1)
    candidate_top1 = candidate.argmax(-1)
    native_correct = native_top1 == targets
    candidate_correct = candidate_top1 == targets
    agreement = native_top1 == candidate_top1

    output: dict[int, dict[str, CellSums]] = {}
    for batch_index, ordinal in enumerate(document_ordinals):
        output[ordinal] = {}
        for cell in CELL_NAMES:
            mask_cpu = cells[cell][batch_index]
            mask = mask_cpu.to(device)
            count = int(mask_cpu.sum())

            def total(value: torch.Tensor) -> float:
                return float(value[batch_index, mask].detach().cpu().double().sum())

            output[ordinal][cell] = CellSums(
                count=count,
                native_nll_sum=total(native_nll),
                candidate_nll_sum=total(candidate_nll),
                teacher_kl_sum=total(teacher_kl),
                centered_logit_sse=total(centered_sse),
                native_centered_logit_energy=total(native_energy),
                raw_logit_sse=total(raw_sse),
                native_correct_count=int(native_correct[batch_index, mask].sum().cpu()),
                candidate_correct_count=int(candidate_correct[batch_index, mask].sum().cpu()),
                native_top1_agreement_count=int(agreement[batch_index, mask].sum().cpu()),
                support_sha256=_support_sha256(rows[batch_index], mask_cpu),
            )
    return output


def _validate_cell_sum(value: CellSums) -> None:
    if type(value) is not CellSums or type(value.count) is not int or value.count < 0:
        raise ValueError("cell sufficient statistics are malformed")
    floats = (
        value.native_nll_sum, value.candidate_nll_sum, value.teacher_kl_sum,
        value.centered_logit_sse, value.native_centered_logit_energy,
        value.raw_logit_sse,
    )
    if not all(math.isfinite(item) and item >= -1e-10 for item in floats) or not (
        0 <= value.native_correct_count <= value.count
        and 0 <= value.candidate_correct_count <= value.count
        and 0 <= value.native_top1_agreement_count <= value.count
        and len(value.support_sha256) == 64
    ):
        raise ValueError("cell sufficient statistics are malformed")


def summarize_arm(
    ledger: Mapping[int, Mapping[str, CellSums]], *, prefix_documents: int,
    include_raw_sufficient_statistics: bool = True,
) -> dict[str, object]:
    selected = tuple(sorted(key for key in ledger if key < prefix_documents))
    if not selected or selected != tuple(range(prefix_documents)):
        raise ValueError("arm ledger does not contain the exact document prefix")
    cells: dict[str, dict[str, float | int | None]] = {}
    per_document_harm = []
    for cell in CELL_NAMES:
        values = [ledger[document][cell] for document in selected]
        if any(set(ledger[document]) != set(CELL_NAMES) for document in selected):
            raise ValueError("arm ledger cell schema changed")
        for value in values:
            _validate_cell_sum(value)
        count = sum(value.count for value in values)
        if count == 0:
            cells[cell] = {"count": 0, "empty": True}
            continue
        totals = {
            field: sum(getattr(value, field) for value in values)
            for field in (
                "native_nll_sum", "candidate_nll_sum", "teacher_kl_sum",
                "centered_logit_sse", "native_centered_logit_energy", "raw_logit_sse",
                "native_correct_count", "candidate_correct_count",
                "native_top1_agreement_count",
            )
        }
        energy = float(totals["native_centered_logit_energy"])
        if energy <= 0:
            raise RuntimeError("native centered-logit energy is zero")
        cells[cell] = {
            "count": count,
            "empty": False,
            "native_ce": float(totals["native_nll_sum"]) / count,
            "candidate_ce": float(totals["candidate_nll_sum"]) / count,
            "candidate_minus_native_ce": (
                float(totals["candidate_nll_sum"] - totals["native_nll_sum"]) / count
            ),
            "teacher_kl": float(totals["teacher_kl_sum"]) / count,
            "centered_logit_nrmse": math.sqrt(
                float(totals["centered_logit_sse"]) / energy
            ),
            "raw_logit_D2": float(totals["raw_logit_sse"]) / count,
            "native_accuracy": int(totals["native_correct_count"]) / count,
            "candidate_accuracy": int(totals["candidate_correct_count"]) / count,
            "native_top1_agreement": int(
                totals["native_top1_agreement_count"]
            ) / count,
        }
        for value in values:
            if cell == "all_scored" and value.count:
                per_document_harm.append(
                    (value.candidate_nll_sum - value.native_nll_sum) / value.count
                )
    result: dict[str, object] = {
        "prefix_documents": prefix_documents,
        "cells": cells,
        "maximum_document_ce_harm": max(per_document_harm),
    }
    if include_raw_sufficient_statistics:
        result["raw_sufficient_statistics"] = {
            str(document): {
                cell: asdict(ledger[document][cell]) for cell in CELL_NAMES
            } for document in selected
        }
    return result


def native_margin_counts(
    native_logits: torch.Tensor, eligible: torch.Tensor, epsilon_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if native_logits.ndim != 3 or not native_logits.is_floating_point() or not bool(
        torch.isfinite(native_logits).all()
    ) or eligible.shape != native_logits.shape[:2] or (
        eligible.device.type != "cpu" or eligible.dtype != torch.bool
        or epsilon_grid.device.type != "cpu" or epsilon_grid.dtype != torch.float64
        or epsilon_grid.ndim != 1 or not bool(torch.isfinite(epsilon_grid).all())
        or not bool((epsilon_grid > 0).all())
        or not bool((epsilon_grid[1:] > epsilon_grid[:-1]).all())
    ):
        raise ValueError("native margin inputs are malformed")
    margins = torch.topk(native_logits, 2, dim=-1).values
    margins = (margins[..., 0] - margins[..., 1]).cpu().double()
    counts = torch.zeros(
        native_logits.shape[0], epsilon_grid.numel(), dtype=torch.long,
    )
    support = eligible.sum(1).long()
    for document in range(native_logits.shape[0]):
        selected = margins[document, eligible[document]]
        counts[document] = (selected[:, None] <= 2 * epsilon_grid[None, :]).sum(0)
    return counts, support


def margin_certificate_curve(
    ledger: Mapping[int, Mapping[str, CellSums]], margin_counts: torch.Tensor,
    support_counts: torch.Tensor, epsilon_grid: torch.Tensor, *, prefix_documents: int,
) -> dict[str, object]:
    documents = tuple(sorted(ledger))
    if documents != tuple(range(len(documents))) or margin_counts.shape != (
        len(ledger), epsilon_grid.numel()
    ) or support_counts.shape != (len(ledger),) or margin_counts.dtype != torch.long or (
        support_counts.dtype != torch.long or bool((margin_counts < 0).any())
        or bool((support_counts < 0).any())
        or bool((margin_counts > support_counts[:, None]).any())
        or bool((margin_counts[:, 1:] < margin_counts[:, :-1]).any())
        or epsilon_grid.dtype != torch.float64 or epsilon_grid.ndim != 1
        or not bool(torch.isfinite(epsilon_grid).all())
        or not bool((epsilon_grid > 0).all())
        or not bool((epsilon_grid[1:] > epsilon_grid[:-1]).all())
    ):
        raise ValueError("margin-count ledger is malformed")
    values = [ledger[index]["all_scored"] for index in range(prefix_documents)]
    for index, value in enumerate(values):
        _validate_cell_sum(value)
        if value.count != int(support_counts[index]):
            raise RuntimeError("per-document margin support changed")
    n = sum(value.count for value in values)
    if n <= 0 or n != int(support_counts[:prefix_documents].sum()):
        raise RuntimeError("margin certificate support changed")
    d2 = sum(value.raw_logit_sse for value in values) / n
    small = margin_counts[:prefix_documents].sum(0).double() / n
    bound = (1 - small - d2 / epsilon_grid.square()).clamp(0, 1)
    best = int(bound.argmax())
    return {
        "epsilon_grid": epsilon_grid.tolist(),
        "small_margin_probability": small.tolist(),
        "raw_logit_D2": d2,
        "bounds": bound.tolist(),
        "maximizing_epsilon": float(epsilon_grid[best]),
        "maximum_bound": float(bound[best]),
    }


def _relative_kl_bootstrap(
    ledgers: Mapping[str, Mapping[int, Mapping[str, CellSums]]], *,
    primary: str = "SUFFIX", controls: Sequence[str] = EQUAL_PRICE_CONTROLS,
    repetitions: int = BOOTSTRAP_REPETITIONS, seed: str = BOOTSTRAP_SEED,
) -> dict[str, object]:
    expected = {primary, *controls}
    if set(ledgers) != expected or len(set(controls)) != len(controls) or (
        type(repetitions) is not int or repetitions <= 0 or not seed
    ):
        raise ValueError("relative-KL bootstrap family or protocol changed")
    documents = tuple(sorted(ledgers[primary]))
    if documents != tuple(range(len(documents))) or len(documents) < 2:
        raise ValueError("relative-KL bootstrap documents changed")
    reference_support = [
        ledgers[primary][document]["all_scored"].support_sha256
        for document in documents
    ]
    counts = torch.tensor([
        ledgers[primary][document]["all_scored"].count for document in documents
    ], dtype=torch.float64)
    if bool((counts < 0).any()) or float(counts.sum()) <= 0:
        raise ValueError("relative-KL bootstrap support is empty")
    sums = {}
    for arm in expected:
        if tuple(sorted(ledgers[arm])) != documents or any(
            ledgers[arm][document]["all_scored"].support_sha256
            != reference_support[document] or (
                ledgers[arm][document]["all_scored"].count != int(counts[document])
            ) for document in documents
        ):
            raise ValueError("relative-KL arms do not share exact document support")
        sums[arm] = torch.tensor([
            ledgers[arm][document]["all_scored"].teacher_kl_sum
            for document in documents
        ], dtype=torch.float64)
        for document in documents:
            _validate_cell_sum(ledgers[arm][document]["all_scored"])
    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little")
    )
    draws = torch.randint(
        len(documents), (repetitions, len(documents)), generator=generator,
    )
    weights = torch.zeros(repetitions, len(documents), dtype=torch.float64)
    weights.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    denominator = weights @ counts
    if bool((denominator <= 0).any()):
        raise ZeroDivisionError("relative-KL bootstrap draw has zero denominator")
    pooled = {arm: (weights @ value) / denominator for arm, value in sums.items()}
    if any(bool((pooled[control] <= 0).any()) for control in controls):
        raise ZeroDivisionError("equal-price control KL is zero")
    relative = {
        control: (pooled[control] - pooled[primary]) / pooled[control]
        for control in controls
    }
    minimum = torch.stack([relative[control] for control in controls], 1).min(1).values
    point = {
        control: float((sums[control].sum() - sums[primary].sum()) / sums[control].sum())
        for control in controls
    }
    return {
        "repetitions": repetitions,
        "seed": seed,
        "per_control_point_relative_kl_improvement": point,
        "minimum_point_relative_kl_improvement": min(point.values()),
        "simultaneous_lower_bound": float(
            torch.quantile(minimum, 0.025, interpolation="lower")
        ),
    }


def simultaneous_relative_kl_bootstrap(
    ledgers: Mapping[str, Mapping[int, Mapping[str, CellSums]]], *,
    primary: str = "SUFFIX", controls: Sequence[str] = EQUAL_PRICE_CONTROLS,
    repetitions: int = BOOTSTRAP_REPETITIONS, seed: str = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Canonical frozen validation bootstrap; protocol constants are not tunable."""
    primary_documents = tuple(sorted(ledgers.get(primary, {})))
    if primary != "SUFFIX" or tuple(controls) != EQUAL_PRICE_CONTROLS or (
        repetitions != BOOTSTRAP_REPETITIONS or seed != BOOTSTRAP_SEED
    ) or (
        primary_documents != tuple(range(DOCUMENTS))
    ):
        raise ValueError("canonical relative-KL bootstrap protocol changed")
    return _relative_kl_bootstrap(
        ledgers, primary=primary, controls=controls,
        repetitions=repetitions, seed=seed,
    )


def _centered_delta(native: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    delta = candidate.detach().cpu().double() - native.detach().cpu().double()
    return delta - delta.mean(-1, keepdim=True)


def reduce_signed_geometry_batch(
    native_logits: torch.Tensor, suffix_logits: torch.Tensor,
    signed_logits: Mapping[str, torch.Tensor], cells: Mapping[str, torch.Tensor],
) -> dict[str, dict[str, PairSums]]:
    tensors = (native_logits, suffix_logits, *signed_logits.values())
    if set(signed_logits) != set(SIGNED_KEYS) or set(cells) != set(CELL_NAMES) or (
        native_logits.ndim != 3
        or any(value.shape != native_logits.shape for value in tensors)
        or any(value.device != native_logits.device for value in tensors)
        or any(not value.is_floating_point() or not bool(torch.isfinite(value).all())
               for value in tensors)
        or any(mask.device.type != "cpu" or mask.dtype != torch.bool
               or mask.shape != native_logits.shape[:2] for mask in cells.values())
    ):
        raise ValueError("signed-geometry arms are malformed")
    deltas = {
        key: _centered_delta(native_logits, value).cpu().double()
        for key, value in signed_logits.items()
    }
    deltas["full"] = _centered_delta(native_logits, suffix_logits).cpu().double()
    output = {}
    for cell in CELL_NAMES:
        mask = cells[cell]
        g10 = (
            deltas["plus_0p10"][mask] - deltas["minus_0p10"][mask]
        ) / 0.20
        g25 = (
            deltas["plus_0p25"][mask] - deltas["minus_0p25"][mask]
        ) / 0.50
        pairs = {
            "g0p10_vs_g0p25": (g10, g25),
            "g0p10_vs_full": (g10, deltas["full"][mask]),
            "plus0p10_vs_negminus0p10": (
                deltas["plus_0p10"][mask], -deltas["minus_0p10"][mask],
            ),
            "plus0p25_vs_negminus0p25": (
                deltas["plus_0p25"][mask], -deltas["minus_0p25"][mask],
            ),
        }
        output[cell] = {
            name: PairSums(
                dot=float((left * right).sum()),
                left_norm2=float(left.square().sum()),
                right_norm2=float(right.square().sum()),
            ) for name, (left, right) in pairs.items()
        }
    return output


def float32_precision_audit(
    native_logits: torch.Tensor, candidate_logits: torch.Tensor,
    rows: torch.Tensor, eligible: torch.Tensor,
) -> dict[str, float | bool]:
    """Compare the production float32 reductions with a CPU-float64 recomputation."""
    if native_logits.shape != candidate_logits.shape or native_logits.ndim != 3 or (
        native_logits.device != candidate_logits.device
        or not native_logits.is_floating_point() or not candidate_logits.is_floating_point()
        or not bool(torch.isfinite(native_logits).all())
        or not bool(torch.isfinite(candidate_logits).all())
        or rows.device.type != "cpu" or rows.dtype != torch.long
    ):
        raise ValueError("precision-audit logits or rows are malformed")
    if rows.shape != (native_logits.shape[0], native_logits.shape[1] + 1) or (
        eligible.shape != native_logits.shape[:2] or eligible.device.type != "cpu"
        or eligible.dtype != torch.bool
    ):
        raise ValueError("precision-audit support is malformed")
    targets32 = rows[:, 1:].to(native_logits.device)
    native32, candidate32 = native_logits.float(), candidate_logits.float()
    native_logp32 = F.log_softmax(native32, -1)
    candidate_logp32 = F.log_softmax(candidate32, -1)
    native_nll32 = -native_logp32.gather(2, targets32.unsqueeze(-1)).squeeze(-1)
    candidate_nll32 = -candidate_logp32.gather(2, targets32.unsqueeze(-1)).squeeze(-1)
    kl32 = (native_logp32.exp() * (native_logp32 - candidate_logp32)).sum(-1)
    delta32 = candidate32 - native32
    raw32 = delta32.square().sum(-1)
    centered32 = (delta32 - delta32.mean(-1, keepdim=True)).square().sum(-1)
    native_energy32 = (
        native32 - native32.mean(-1, keepdim=True)
    ).square().sum(-1)

    native64 = native_logits.detach().cpu().double()
    candidate64 = candidate_logits.detach().cpu().double()
    targets64 = rows[:, 1:]
    native_logp64 = F.log_softmax(native64, -1)
    candidate_logp64 = F.log_softmax(candidate64, -1)
    native_nll64 = -native_logp64.gather(2, targets64.unsqueeze(-1)).squeeze(-1)
    candidate_nll64 = -candidate_logp64.gather(2, targets64.unsqueeze(-1)).squeeze(-1)
    kl64 = (native_logp64.exp() * (native_logp64 - candidate_logp64)).sum(-1)
    delta64 = candidate64 - native64
    raw64 = delta64.square().sum(-1)
    centered64 = (delta64 - delta64.mean(-1, keepdim=True)).square().sum(-1)
    native_energy64 = (
        native64 - native64.mean(-1, keepdim=True)
    ).square().sum(-1)
    selected = eligible

    def maximum_absolute(single: torch.Tensor, double: torch.Tensor) -> float:
        return float((single.detach().cpu().double()[selected] - double[selected]).abs().max())

    def maximum_relative(single: torch.Tensor, double: torch.Tensor) -> float:
        difference = (single.detach().cpu().double()[selected] - double[selected]).abs()
        return float((difference / double[selected].abs().clamp_min(1e-12)).max())

    native_nll_error = maximum_absolute(native_nll32, native_nll64)
    candidate_nll_error = maximum_absolute(candidate_nll32, candidate_nll64)
    kl_error = maximum_absolute(kl32, kl64)
    raw_error = maximum_relative(raw32, raw64)
    centered_error = maximum_relative(centered32, centered64)
    native_energy_error = maximum_relative(native_energy32, native_energy64)
    return {
        "maximum_native_nll_absolute_error": native_nll_error,
        "maximum_candidate_nll_absolute_error": candidate_nll_error,
        "maximum_teacher_kl_absolute_error": kl_error,
        "maximum_raw_sse_relative_error": raw_error,
        "maximum_centered_sse_relative_error": centered_error,
        "maximum_native_centered_energy_relative_error": native_energy_error,
        "passed": (
            native_nll_error <= NLL_KL_PRECISION_TOLERANCE
            and candidate_nll_error <= NLL_KL_PRECISION_TOLERANCE
            and kl_error <= NLL_KL_PRECISION_TOLERANCE
            and raw_error <= SQUARED_ERROR_RELATIVE_TOLERANCE
            and centered_error <= SQUARED_ERROR_RELATIVE_TOLERANCE
            and native_energy_error <= SQUARED_ERROR_RELATIVE_TOLERANCE
        ),
    }


def enforce_float32_precision_audit(
    native_logits: torch.Tensor, candidate_logits: torch.Tensor,
    rows: torch.Tensor, eligible: torch.Tensor,
) -> dict[str, float | bool]:
    audit = float32_precision_audit(
        native_logits, candidate_logits, rows, eligible,
    )
    if audit["passed"] is not True:
        raise RuntimeError("float32 validation metrics failed the frozen CPU-float64 audit")
    return audit


def summarize_signed_geometry(
    batches: Sequence[Mapping[str, Mapping[str, PairSums]]],
) -> dict[str, dict[str, dict[str, float | bool]]]:
    if not batches:
        raise ValueError("signed-geometry batch ledger is empty")
    output = {}
    for cell in CELL_NAMES:
        output[cell] = {}
        for pair in GEOMETRY_PAIRS:
            values = [batch[cell][pair] for batch in batches]
            dot = sum(value.dot for value in values)
            left = sum(value.left_norm2 for value in values)
            right = sum(value.right_norm2 for value in values)
            cosine = None if left <= 0 or right <= 0 else dot / math.sqrt(left * right)
            output[cell][pair] = {
                "dot": dot,
                "left_norm2": left,
                "right_norm2": right,
                "cosine": cosine,
                "nonzero": cosine is not None,
            }
    return output
