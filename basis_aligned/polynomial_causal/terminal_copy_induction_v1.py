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
MIN_DOCUMENTS_PER_POLARITY_STRATUM = 2

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


def _frequency_bin(value: int) -> int:
    """Keep unseen tokens distinct from tokens observed exactly once."""
    if value < 0:
        raise ValueError("token frequency cannot be negative")
    return -1 if value == 0 else _log2_bin(value)


@dataclass(frozen=True)
class FitTokenFrequencies:
    """Fit-only frequencies for the distinct query and target column domains."""

    query: torch.Tensor
    target: torch.Tensor

    def validate(self, maximum_token: int) -> None:
        _validate_counts(self.query, maximum_token)
        _validate_counts(self.target, maximum_token)

    @classmethod
    def from_rows(cls, rows: torch.Tensor, vocab_size: int = 50_257) -> "FitTokenFrequencies":
        _validate_rows(rows)
        if type(vocab_size) is not int or vocab_size <= int(rows.max()):
            raise ValueError("fit frequency vocabulary does not cover rows")
        return cls(
            query=torch.bincount(rows[:, :MODEL_WIDTH].reshape(-1), minlength=vocab_size).long(),
            target=torch.bincount(rows[:, 1:].reshape(-1), minlength=vocab_size).long(),
        )


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
    eligible_stratum_count: int
    excluded_low_document_stratum_count: int

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
    fit_token_frequencies: FitTokenFrequencies,
    ordered_document_ids: Sequence[str],
    *, minimum_documents_per_polarity_stratum: int = MIN_DOCUMENTS_PER_POLARITY_STRATUM,
) -> CopyCells:
    """Derive nearest-successor labels and deterministic matched specificity cells.

    Matching is not the causal estimator.  It balances registered observables for
    comparing within-input intervention effects on positive versus negative cells.
    """

    _validate_rows(rows)
    if not isinstance(fit_token_frequencies, FitTokenFrequencies):
        raise ValueError("fit query/target frequencies must be separately bound")
    fit_token_frequencies.validate(int(rows.max()))
    if type(minimum_documents_per_polarity_stratum) is not int or (
        minimum_documents_per_polarity_stratum < 2
    ):
        raise ValueError("each polarity needs at least two documents per stratum")
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
            # Freeze the strong induction label: the nearest previous occurrence of
            # the current query must itself have the current successor.  An older
            # q->y witness cannot override a nearer contradictory q->z association.
            chosen = max(predecessors)
            matching = int(row[chosen + 1]) == target
            distance = position - chosen
            stratum = (
                position // 16,
                _log2_bin(distance),
                _frequency_bin(int(fit_token_frequencies.query[query])),
                _frequency_bin(int(fit_token_frequencies.target[target])),
                _log2_bin(len(predecessors)),
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
    eligible_strata = 0
    excluded_strata = 0
    for stratum in sorted(set(positive_records) & set(negative_records)):
        positives = sorted(positive_records[stratum])
        negatives = sorted(negative_records[stratum])
        positive_documents = {row for _, row, _ in positives}
        negative_documents = {row for _, row, _ in negatives}
        if min(len(positive_documents), len(negative_documents)) < (
            minimum_documents_per_polarity_stratum
        ):
            excluded_strata += 1
            continue
        eligible_strata += 1
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
        eligible_stratum_count=eligible_strata,
        excluded_low_document_stratum_count=excluded_strata,
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
class SyntheticAssociationCrossover:
    """Two histories for a reciprocal q/r by y/z association difference-in-differences."""

    query_to_y: torch.Tensor
    query_to_z: torch.Tensor
    query_position: int
    query_token: int
    successor_y: int
    successor_z: int
    reciprocal_query: int


def build_synthetic_association_crossover(
    base_row: Sequence[int], *, first_query_position: int,
    reciprocal_position: int, query_position: int,
    query_token: int, reciprocal_query: int, successor_y: int, successor_z: int,
) -> SyntheticAssociationCrossover:
    """Build a reciprocal association crossover without treating a swap as one edge.

    Both rows have the same length and token multiset.  The scorer must evaluate both
    y and z log-probabilities at ``query_position`` and take
    ``[(log p(y)-log p(z))_q->y - (log p(y)-log p(z))_q->z]``.
    """

    values = tuple(base_row)
    tokens = (query_token, reciprocal_query, successor_y, successor_z)
    positions = (first_query_position, reciprocal_position, query_position)
    overwritten = {
        first_query_position, first_query_position + 1,
        reciprocal_position, reciprocal_position + 1,
        query_position, query_position + 1,
    }
    if len(values) != ROW_WIDTH or any(type(value) is not int or not 0 <= value < 50_257 for value in values):
        raise ValueError("synthetic crossover base must be 257 valid token IDs")
    if len(set(tokens)) != 4 or any(type(value) is not int or not 0 <= value < 50_256 for value in tokens):
        raise ValueError("synthetic crossover tokens must be four distinct nonspecial IDs")
    if any(type(position) is not int for position in positions) or not (
        0 <= first_query_position < first_query_position + 1 < query_position
        and 0 <= reciprocal_position < reciprocal_position + 1 < query_position
        and SCORE_START <= query_position < SCORE_STOP
        and len(set(overwritten)) == 6
    ):
        raise ValueError("synthetic crossover positions are malformed")
    if any(value in tokens for index, value in enumerate(values) if index not in overwritten):
        raise ValueError("synthetic crossover bank token already occurs outside intervention slots")

    query_to_y = list(values)
    query_to_z = list(values)
    query_to_y[first_query_position:first_query_position + 2] = [query_token, successor_y]
    query_to_y[reciprocal_position:reciprocal_position + 2] = [reciprocal_query, successor_z]
    query_to_z[first_query_position:first_query_position + 2] = [query_token, successor_z]
    query_to_z[reciprocal_position:reciprocal_position + 2] = [reciprocal_query, successor_y]
    # The observed next token is held fixed.  Both alternative successors are scored
    # explicitly, so this column is not used to manufacture a second input prefix.
    query_to_y[query_position:query_position + 2] = [query_token, successor_y]
    query_to_z[query_position:query_position + 2] = [query_token, successor_y]
    first = torch.tensor(query_to_y, dtype=torch.long)
    second = torch.tensor(query_to_z, dtype=torch.long)
    if not torch.equal(torch.sort(first).values, torch.sort(second).values):
        raise RuntimeError("synthetic crossover failed to preserve the token multiset")
    return SyntheticAssociationCrossover(
        query_to_y=first, query_to_z=second, query_position=query_position,
        query_token=query_token, successor_y=successor_y, successor_z=successor_z,
        reciprocal_query=reciprocal_query,
    )


@dataclass(frozen=True)
class CellReduction:
    count: int
    ce: float
    target_logprob: float
    top1_accuracy: float
    native_to_candidate_kl: float | None


@dataclass(frozen=True)
class CausalCopyContrast:
    """Within-input ablation effects and their matched-cell specificity contrast."""

    positive_ce_effect: float
    matched_negative_ce_effect: float
    specificity_ce_effect: float


def causal_copy_contrast(
    native: Mapping[str, CellReduction],
    ablated: Mapping[str, CellReduction],
) -> CausalCopyContrast:
    """Compute causal CE effects; matching is used only for effect specificity.

    Positive values mean that ablation worsened prediction.  The primary causal
    estimand is the native-to-ablation change on the same positive inputs.  The
    matched-negative subtraction asks whether that harm is specific to copy events.
    """

    required = {"positive", "matched_negative", "off_target"}
    if set(native) != required or set(ablated) != required:
        raise ValueError("causal reductions must contain the exact cell schema")
    for name in required:
        if native[name].count != ablated[name].count:
            raise ValueError("native and ablated reductions have unequal cell support")
    positive = ablated["positive"].ce - native["positive"].ce
    negative = ablated["matched_negative"].ce - native["matched_negative"].ce
    if not math.isfinite(positive) or not math.isfinite(negative):
        raise ValueError("causal CE effects must be finite")
    return CausalCopyContrast(
        positive_ce_effect=positive,
        matched_negative_ce_effect=negative,
        specificity_ce_effect=positive - negative,
    )


def synthetic_association_did(
    logits: torch.Tensor,
    crossover: SyntheticAssociationCrossover,
) -> float:
    """Score the reciprocal association difference-in-differences.

    ``logits[0]`` is the q->y/r->z history and ``logits[1]`` is q->z/r->y.
    Both alternatives y and z are scored at the shared current-query position.
    A positive value means the earlier q->y association selectively shifts the
    current q prediction toward y rather than z.
    """

    if (
        not torch.is_tensor(logits) or logits.device.type != "cpu"
        or logits.ndim != 3 or tuple(logits.shape[:2]) != (2, MODEL_WIDTH)
        or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
        or logits.shape[2] <= max(crossover.successor_y, crossover.successor_z)
    ):
        raise ValueError("synthetic crossover logits are malformed")
    logprob = F.log_softmax(
        logits[:, crossover.query_position, :].double(), dim=-1,
    )
    preference = (
        logprob[:, crossover.successor_y] - logprob[:, crossover.successor_z]
    )
    return float(preference[0] - preference[1])


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
