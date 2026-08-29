"""Pure CPU contract for the preregistered terminal copy/induction v1 screen.

No function in this module loads rows, a tokenizer, a checkpoint, or a model.  It
freezes token-ID labels, deterministic matched controls, semantic reductions, and the
launch-NO-GO boundary used by a future source-closed collector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


ROW_WIDTH = 257
MODEL_WIDTH = 256
SCORE_START = 64
SCORE_STOP = 256
MATCH_SEED = "terminal_copy_induction_v1:matched-natural:0"

NAMED_SIX_HEAD_FAMILY = (
    "L5H5", "L7H3", "L8H3", "L8H4", "L13H0", "L14H7",
)
REGISTERED_FOUR_HEAD_SET = ("L5H5", "L7H3", "L8H3", "L8H4")
REGISTERED_LATE_PAIR = ("L13H0", "L14H7")
CAPITALIZATION_SPECIFICITY_NEGATIVE = {
    "name": "boundary_capitalization_generic_booster",
    "prior_boundary_over_proper_noun_ratio": 1.0,
    "eligible_for_selection": False,
}

REQUIRED_LAUNCH_BINDINGS = (
    "fresh_row_authority",
    "checkpoint_authority",
    "per_head_attention_adapter",
    "late_product_gate_adapter_or_omission_authority",
    "scorer_bootstrap_authority",
    "empty_create_only_result_namespace",
)


def _validate_rows(rows: torch.Tensor) -> None:
    if (
        not torch.is_tensor(rows) or rows.device.type != "cpu"
        or rows.dtype != torch.long or rows.ndim != 2
        or rows.shape[1] != ROW_WIDTH or rows.shape[0] == 0
        or int(rows.min()) < 0
    ):
        raise ValueError("rows must be nonempty CPU long[n,257] token IDs")


def _validate_counts(counts: torch.Tensor, maximum_token: int) -> None:
    if (
        not torch.is_tensor(counts) or counts.device.type != "cpu"
        or counts.dtype != torch.long or counts.ndim != 1
        or len(counts) <= maximum_token or bool((counts < 0).any())
    ):
        raise ValueError("fit token counts do not cover the row token support")


def _log2_bin(value: int) -> int:
    return int(math.floor(math.log2(max(value, 1))))


def _order_digest(document_id: str, position: int, label: str) -> bytes:
    return hashlib.sha256(
        f"{MATCH_SEED}\0{label}\0{document_id}\0{position}".encode()
    ).digest()


@dataclass(frozen=True)
class CopyCells:
    """Disjoint confirmatory cells and matching diagnostics for one row role."""

    all_positive: torch.Tensor
    positive: torch.Tensor
    matched_negative: torch.Tensor
    off_target: torch.Tensor
    pair_indices: tuple[tuple[int, int, int, int], ...]
    unmatched_positive_count: int
    negative_candidate_count: int

    def __post_init__(self) -> None:
        shape = self.all_positive.shape
        masks = (self.all_positive, self.positive, self.matched_negative, self.off_target)
        if any(
            not torch.is_tensor(value) or value.device.type != "cpu"
            or value.dtype != torch.bool or value.shape != shape
            for value in masks
        ) or len(shape) != 2 or shape[1] != MODEL_WIDTH:
            raise ValueError("copy cells must be aligned CPU bool[n,256] masks")
        valid = torch.zeros(shape, dtype=torch.bool)
        valid[:, SCORE_START:SCORE_STOP] = True
        if bool((self.positive & ~self.all_positive).any()):
            raise ValueError("matched positives must be genuine copy positives")
        if bool((self.positive & self.matched_negative).any()) or bool(
            (self.all_positive & self.off_target).any()
        ) or bool((self.matched_negative & self.off_target).any()):
            raise ValueError("confirmatory copy cells overlap")
        if bool(((self.all_positive | self.matched_negative | self.off_target) ^ valid).any()):
            raise ValueError("copy cells do not close the scored support")
        if int(self.positive.sum()) != int(self.matched_negative.sum()) or int(
            self.positive.sum()
        ) != len(self.pair_indices):
            raise ValueError("positive/negative matching is not one-to-one")


def build_copy_cells(
    rows: torch.Tensor,
    fit_token_counts: torch.Tensor,
    ordered_document_ids: Sequence[str],
) -> CopyCells:
    """Derive copy labels and deterministic, without-replacement matched controls."""

    _validate_rows(rows)
    _validate_counts(fit_token_counts, int(rows.max()))
    if len(ordered_document_ids) != len(rows) or any(
        not isinstance(value, str) or not value for value in ordered_document_ids
    ) or len(set(ordered_document_ids)) != len(ordered_document_ids):
        raise ValueError("ordered document IDs must be unique nonempty strings")

    shape = (len(rows), MODEL_WIDTH)
    all_positive = torch.zeros(shape, dtype=torch.bool)
    positive_records: dict[tuple[int, int, int, int], list[tuple[bytes, int, int]]] = {}
    negative_records: dict[tuple[int, int, int, int], list[tuple[bytes, int, int]]] = {}
    for row_index, row in enumerate(rows):
        document_id = ordered_document_ids[row_index]
        for position in range(SCORE_START, SCORE_STOP):
            query, target = int(row[position]), int(row[position + 1])
            predecessors = [
                earlier for earlier in range(position)
                if int(row[earlier]) == query
            ]
            if not predecessors:
                continue
            matching = [
                earlier for earlier in predecessors
                if int(row[earlier + 1]) == target
            ]
            chosen = max(matching if matching else predecessors)
            distance = position - chosen
            stratum = (
                position // 16,
                _log2_bin(distance),
                _log2_bin(int(fit_token_counts[query])),
                _log2_bin(int(fit_token_counts[target])),
            )
            record = (
                _order_digest(document_id, position, "positive" if matching else "negative"),
                row_index,
                position,
            )
            if matching:
                all_positive[row_index, position] = True
                positive_records.setdefault(stratum, []).append(record)
            else:
                negative_records.setdefault(stratum, []).append(record)

    positive = torch.zeros(shape, dtype=torch.bool)
    matched_negative = torch.zeros(shape, dtype=torch.bool)
    pairs: list[tuple[int, int, int, int]] = []
    for stratum in sorted(set(positive_records) & set(negative_records)):
        positives = sorted(positive_records[stratum])
        negatives = sorted(negative_records[stratum])
        for (_, positive_row, positive_position), (_, negative_row, negative_position) in zip(
            positives, negatives, strict=False,
        ):
            positive[positive_row, positive_position] = True
            matched_negative[negative_row, negative_position] = True
            pairs.append((positive_row, positive_position, negative_row, negative_position))

    valid = torch.zeros(shape, dtype=torch.bool)
    valid[:, SCORE_START:SCORE_STOP] = True
    off_target = valid & ~all_positive & ~matched_negative
    return CopyCells(
        all_positive=all_positive,
        positive=positive,
        matched_negative=matched_negative,
        off_target=off_target,
        pair_indices=tuple(pairs),
        unmatched_positive_count=int(all_positive.sum() - positive.sum()),
        negative_candidate_count=sum(map(len, negative_records.values())),
    )


def build_synthetic_copy_pair(
    stem: Sequence[int], sequence: Sequence[int], cut: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct a same-length, same-multiset positive/control row pair."""

    stem_values, values = tuple(stem), tuple(sequence)
    if (
        len(values) < 4 or len(set(values)) != len(values)
        or type(cut) is not int or not 1 <= cut < len(values)
        or any(type(value) is not int or value < 0 for value in stem_values + values)
        or not set(stem_values).isdisjoint(values)
    ):
        raise ValueError("synthetic copy inputs are malformed")
    alternate = next(index for index in range(len(values)) if index not in (cut, cut - 1))
    control_first = list(values)
    control_first[cut], control_first[alternate] = (
        control_first[alternate], control_first[cut],
    )
    positive = stem_values + values + values[:cut] + (values[cut],)
    control = stem_values + tuple(control_first) + values[:cut] + (values[cut],)
    if len(positive) != ROW_WIDTH or len(control) != ROW_WIDTH:
        raise ValueError("synthetic stem does not produce an exact 257-token row")
    return torch.tensor(positive), torch.tensor(control)


@dataclass(frozen=True)
class CellReduction:
    count: int
    ce: float
    target_logprob: float
    top1_accuracy: float
    native_to_candidate_kl: float | None


def _reduce_cell(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
    native_logits: torch.Tensor | None,
) -> CellReduction:
    count = int(mask.sum())
    if count <= 0:
        raise ValueError("every reported confirmatory cell must have positive support")
    selected = logits[mask].double()
    selected_targets = targets[mask]
    logprob = F.log_softmax(selected, dim=-1)
    target_logprob = logprob.gather(1, selected_targets[:, None]).squeeze(1)
    kl = None
    if native_logits is not None:
        native_logprob = F.log_softmax(native_logits[mask].double(), dim=-1)
        native_prob = native_logprob.exp()
        kl = float((native_prob * (native_logprob - logprob)).sum(1).mean())
    return CellReduction(
        count=count,
        ce=float(-target_logprob.mean()),
        target_logprob=float(target_logprob.mean()),
        top1_accuracy=float((selected.argmax(1) == selected_targets).double().mean()),
        native_to_candidate_kl=kl,
    )


def reduce_behavior(
    logits: torch.Tensor, rows: torch.Tensor, cells: CopyCells,
    *, native_logits: torch.Tensor | None = None,
) -> dict[str, CellReduction]:
    """Reduce CE/log-prob/top-1/KL without pooling behavioral currencies."""

    _validate_rows(rows)
    expected = (len(rows), MODEL_WIDTH)
    if (
        not torch.is_tensor(logits) or logits.device.type != "cpu"
        or logits.ndim != 3 or tuple(logits.shape[:2]) != expected
        or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
        or cells.all_positive.shape != expected
    ):
        raise ValueError("behavior logits/cells are malformed or misaligned")
    if native_logits is not None and (
        not torch.is_tensor(native_logits) or native_logits.shape != logits.shape
        or native_logits.device.type != "cpu" or not native_logits.is_floating_point()
        or not bool(torch.isfinite(native_logits).all())
    ):
        raise ValueError("native logits are malformed or misaligned")
    targets = rows[:, 1:]
    return {
        "positive": _reduce_cell(logits, targets, cells.positive, native_logits),
        "matched_negative": _reduce_cell(
            logits, targets, cells.matched_negative, native_logits,
        ),
        "off_target": _reduce_cell(logits, targets, cells.off_target, native_logits),
    }


def extraction_recovery(
    *, native_positive_ce: float, ablated_positive_ce: float,
    extracted_positive_ce: float,
) -> float:
    """Return the preregistered positive-CE extraction currency."""

    values = (native_positive_ce, ablated_positive_ce, extracted_positive_ce)
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
        raise ValueError("extraction CE inputs must be finite scalars")
    denominator = ablated_positive_ce - native_positive_ce
    if denominator <= 0:
        raise ValueError("native-versus-ablation positive-CE stake is not positive")
    return (ablated_positive_ce - extracted_positive_ce) / denominator


def assert_launch_ready(bindings: Mapping[str, bool]) -> None:
    """Fail closed until every separately reviewed source/lifecycle owner exists."""

    if set(bindings) != set(REQUIRED_LAUNCH_BINDINGS) or any(
        type(value) is not bool for value in bindings.values()
    ):
        raise ValueError("launch bindings must be the exact frozen boolean schema")
    missing = [name for name in REQUIRED_LAUNCH_BINDINGS if not bindings[name]]
    if missing:
        raise RuntimeError("terminal copy/induction v1 launch NO-GO: " + ", ".join(missing))
