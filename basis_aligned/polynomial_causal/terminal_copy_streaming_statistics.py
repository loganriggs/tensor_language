"""Pure CPU sufficient statistics and simultaneous bootstrap for E4 copy v1."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


CELL_NAMES = ("positive", "matched_negative", "off_target")
COLLATERAL_LIMIT = 0.01
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = "terminal-copy-v1-document-bootstrap:0"
FROZEN_CANDIDATES = (
    "L5H5", "L7H3", "L8H3", "L8H4", "L13H0", "L14H7",
    "registered_four_head_set", "registered_late_pair",
)
REPLICATION_ROLES = ("final_natural", "ood_code")


@dataclass(frozen=True)
class DocumentCellSums:
    n: int
    native_nll_sum: float
    ablated_nll_sum: float
    native_correct_count: int
    ablated_correct_count: int
    native_to_ablated_kl_sum: float
    support_sha256: str


@dataclass(frozen=True)
class CandidateEffects:
    tau_positive: float
    tau_matched_negative: float
    specificity: float
    tau_off_target: float
    collateral_margin: float


@dataclass(frozen=True)
class SelectionResult:
    candidates: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    point_estimates: torch.Tensor
    simultaneous_lower_bounds: torch.Tensor
    critical_value: float
    selected_candidate: str | None


@dataclass(frozen=True)
class ReplicationResult:
    roles: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    point_estimates: torch.Tensor
    simultaneous_lower_bounds: torch.Tensor
    critical_value: float
    passed: bool


def _support_digest(row: torch.Tensor, mask: torch.Tensor) -> str:
    selected = torch.nonzero(mask, as_tuple=False).to(torch.int64).contiguous()
    digest = hashlib.sha256()
    digest.update(str(row.dtype).encode())
    digest.update(str(tuple(row.shape)).encode())
    digest.update(row.contiguous().numpy().tobytes(order="C"))
    digest.update(selected.numpy().tobytes(order="C"))
    return digest.hexdigest()


def reduce_document_batch(
    native_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
    rows: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    document_ids: Sequence[str],
) -> dict[str, dict[str, DocumentCellSums]]:
    """Reduce a batch without retaining logits; one row belongs to one source document."""

    expected = (len(rows), 256)
    if (
        not torch.is_tensor(rows) or rows.device.type != "cpu" or rows.dtype != torch.long
        or rows.ndim != 2 or tuple(rows.shape[1:]) != (257,)
        or len(document_ids) != len(rows) or len(set(document_ids)) != len(document_ids)
        or any(not isinstance(value, str) or not value for value in document_ids)
        or set(masks) != set(CELL_NAMES)
        or any(
            not torch.is_tensor(mask) or mask.device.type != "cpu"
            or mask.dtype != torch.bool or tuple(mask.shape) != expected
            for mask in masks.values()
        )
    ):
        raise ValueError("streaming rows, document IDs, or masks are malformed")
    for logits in (native_logits, ablated_logits):
        if (
            not torch.is_tensor(logits) or logits.device.type != "cpu"
            or logits.ndim != 3 or tuple(logits.shape[:2]) != expected
            or not logits.is_floating_point() or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("streaming logits are malformed")
    if native_logits.shape != ablated_logits.shape or native_logits.shape[2] <= int(rows.max()):
        raise ValueError("streaming arms or vocabulary are misaligned")

    native_logprob = F.log_softmax(native_logits.double(), dim=-1)
    ablated_logprob = F.log_softmax(ablated_logits.double(), dim=-1)
    targets = rows[:, 1:]
    native_target = native_logprob.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    ablated_target = ablated_logprob.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    native_correct = native_logits.argmax(-1) == targets
    ablated_correct = ablated_logits.argmax(-1) == targets
    point_kl = (
        native_logprob.exp() * (native_logprob - ablated_logprob)
    ).sum(-1)
    if float(point_kl.min()) < -1e-12:
        raise RuntimeError("native-to-ablated KL is numerically negative")
    point_kl = point_kl.clamp_min(0)

    output: dict[str, dict[str, DocumentCellSums]] = {}
    for row_index, document_id in enumerate(document_ids):
        output[document_id] = {}
        for cell in CELL_NAMES:
            mask = masks[cell][row_index]
            n = int(mask.sum())
            output[document_id][cell] = DocumentCellSums(
                n=n,
                native_nll_sum=float(-native_target[row_index, mask].sum()),
                ablated_nll_sum=float(-ablated_target[row_index, mask].sum()),
                native_correct_count=int(native_correct[row_index, mask].sum()),
                ablated_correct_count=int(ablated_correct[row_index, mask].sum()),
                native_to_ablated_kl_sum=float(point_kl[row_index, mask].sum()),
                support_sha256=_support_digest(rows[row_index], mask),
            )
    return output


def _validate_ledger(
    ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    candidates = tuple(sorted(ledgers))
    if not candidates:
        raise ValueError("candidate ledger is empty")
    documents = tuple(sorted(ledgers[candidates[0]]))
    if not documents:
        raise ValueError("document ledger is empty")
    reference_support: dict[tuple[str, str], str] = {}
    for candidate in candidates:
        if tuple(sorted(ledgers[candidate])) != documents:
            raise ValueError("candidate ledgers do not share exact documents")
        for document in documents:
            cells = ledgers[candidate][document]
            if set(cells) != set(CELL_NAMES):
                raise ValueError("document ledger has wrong cell schema")
            for cell in CELL_NAMES:
                value = cells[cell]
                scalars = (
                    value.native_nll_sum, value.ablated_nll_sum,
                    value.native_to_ablated_kl_sum,
                )
                if (
                    type(value.n) is not int or value.n < 0
                    or not all(math.isfinite(item) for item in scalars)
                    or not 0 <= value.native_correct_count <= value.n
                    or not 0 <= value.ablated_correct_count <= value.n
                    or value.native_to_ablated_kl_sum < -1e-12
                    or not isinstance(value.support_sha256, str)
                    or len(value.support_sha256) != 64
                ):
                    raise ValueError("document sufficient statistics are malformed")
                key = (document, cell)
                if key in reference_support and reference_support[key] != value.support_sha256:
                    raise ValueError("candidates do not share exact input support")
                reference_support[key] = value.support_sha256
    return candidates, documents


def _effect_from_totals(delta: Mapping[str, float], counts: Mapping[str, float]) -> CandidateEffects:
    if any(counts[cell] <= 0 for cell in CELL_NAMES):
        raise ZeroDivisionError("bootstrap or point estimate has zero cell denominator")
    tau = {cell: delta[cell] / counts[cell] for cell in CELL_NAMES}
    return CandidateEffects(
        tau_positive=tau["positive"],
        tau_matched_negative=tau["matched_negative"],
        specificity=tau["positive"] - tau["matched_negative"],
        tau_off_target=tau["off_target"],
        collateral_margin=COLLATERAL_LIMIT - tau["off_target"],
    )


def pooled_effects(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]],
    multiplicities: Mapping[str, int] | None = None,
) -> CandidateEffects:
    documents = tuple(sorted(ledger))
    weights = {document: 1 for document in documents} if multiplicities is None else dict(multiplicities)
    if set(weights) != set(documents) or any(type(value) is not int or value < 0 for value in weights.values()):
        raise ValueError("document multiplicities are malformed")
    delta = {cell: 0.0 for cell in CELL_NAMES}
    counts = {cell: 0.0 for cell in CELL_NAMES}
    for document in documents:
        for cell in CELL_NAMES:
            value = ledger[document][cell]
            weight = weights[document]
            delta[cell] += weight * (value.ablated_nll_sum - value.native_nll_sum)
            counts[cell] += weight * value.n
    return _effect_from_totals(delta, counts)


def simultaneous_selection_bootstrap(
    ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
    *, repetitions: int = BOOTSTRAP_REPETITIONS, seed: str = BOOTSTRAP_SEED,
    expected_candidates: Sequence[str] = FROZEN_CANDIDATES,
) -> SelectionResult:
    """Apply the frozen 24-coordinate one-sided document-cluster selection band."""

    candidates, documents = _validate_ledger(ledgers)
    if candidates != tuple(sorted(expected_candidates)) or len(set(expected_candidates)) != len(
        expected_candidates
    ):
        raise ValueError("selection candidate family differs from the frozen bank")
    if type(repetitions) is not int or repetitions <= 0 or not isinstance(seed, str) or not seed:
        raise ValueError("bootstrap repetitions or seed are malformed")
    point_values = []
    for candidate in candidates:
        effect = pooled_effects(ledgers[candidate])
        point_values.extend((effect.tau_positive, effect.specificity, effect.collateral_margin))
    point = torch.tensor(point_values, dtype=torch.float64)

    generator = torch.Generator().manual_seed(
        int.from_bytes(hashlib.sha256(seed.encode()).digest()[:8], "little")
    )
    draws = torch.randint(len(documents), (repetitions, len(documents)), generator=generator)
    multiplicities = torch.zeros(repetitions, len(documents), dtype=torch.float64)
    multiplicities.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
    replicate_columns = []
    for candidate in candidates:
        candidate_cells = []
        for cell in CELL_NAMES:
            delta = torch.tensor([
                ledgers[candidate][document][cell].ablated_nll_sum
                - ledgers[candidate][document][cell].native_nll_sum
                for document in documents
            ], dtype=torch.float64)
            count = torch.tensor([
                ledgers[candidate][document][cell].n for document in documents
            ], dtype=torch.float64)
            denominator = multiplicities @ count
            if bool((denominator <= 0).any()):
                raise ZeroDivisionError("bootstrap replicate has zero cell denominator")
            candidate_cells.append((multiplicities @ delta) / denominator)
        tau_positive, tau_negative, tau_off = candidate_cells
        replicate_columns.extend((
            tau_positive,
            tau_positive - tau_negative,
            COLLATERAL_LIMIT - tau_off,
        ))
    replicates = torch.stack(replicate_columns, dim=1)
    maxima = (replicates - point).max(dim=1).values.sort().values
    critical_index = math.ceil(0.95 * repetitions) - 1
    critical = float(maxima[critical_index])
    lower = point - critical
    passers = []
    for index, candidate in enumerate(candidates):
        if bool((lower[3 * index:3 * index + 3] > 0).all()):
            passers.append((float(lower[3 * index + 1]), candidate))
    selected = sorted(passers, key=lambda item: (-item[0], item[1]))[0][1] if passers else None
    coordinate_names = tuple(
        f"{candidate}:{name}"
        for candidate in candidates
        for name in ("tau_positive", "specificity", "collateral_margin")
    )
    return SelectionResult(
        candidates=candidates,
        coordinate_names=coordinate_names,
        point_estimates=point,
        simultaneous_lower_bounds=lower,
        critical_value=critical,
        selected_candidate=selected,
    )


def _bootstrap_effect_columns(
    ledger: Mapping[str, Mapping[str, DocumentCellSums]],
    documents: tuple[str, ...], multiplicities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cells = []
    for cell in CELL_NAMES:
        delta = torch.tensor([
            ledger[document][cell].ablated_nll_sum
            - ledger[document][cell].native_nll_sum
            for document in documents
        ], dtype=torch.float64)
        count = torch.tensor([
            ledger[document][cell].n for document in documents
        ], dtype=torch.float64)
        denominator = multiplicities @ count
        if bool((denominator <= 0).any()):
            raise ZeroDivisionError("bootstrap replicate has zero cell denominator")
        cells.append((multiplicities @ delta) / denominator)
    tau_positive, tau_negative, tau_off = cells
    return tau_positive, tau_positive - tau_negative, COLLATERAL_LIMIT - tau_off


def simultaneous_final_ood_bootstrap(
    role_ledgers: Mapping[str, Mapping[str, Mapping[str, DocumentCellSums]]],
    *, repetitions: int = BOOTSTRAP_REPETITIONS, seed: str = BOOTSTRAP_SEED,
) -> ReplicationResult:
    """Apply the frozen six-coordinate joint final/OOD replication gate."""

    if tuple(sorted(role_ledgers)) != tuple(sorted(REPLICATION_ROLES)):
        raise ValueError("replication role family differs from the frozen bank")
    if type(repetitions) is not int or repetitions <= 0 or not isinstance(seed, str) or not seed:
        raise ValueError("bootstrap repetitions or seed are malformed")
    point_values = []
    replicate_columns = []
    for role in REPLICATION_ROLES:
        candidates, documents = _validate_ledger({"selected": role_ledgers[role]})
        assert candidates == ("selected",)
        effect = pooled_effects(role_ledgers[role])
        point_values.extend((effect.tau_positive, effect.specificity, effect.collateral_margin))
        generator = torch.Generator().manual_seed(
            int.from_bytes(hashlib.sha256(f"{seed}\0{role}".encode()).digest()[:8], "little")
        )
        draws = torch.randint(len(documents), (repetitions, len(documents)), generator=generator)
        multiplicities = torch.zeros(repetitions, len(documents), dtype=torch.float64)
        multiplicities.scatter_add_(1, draws, torch.ones_like(draws, dtype=torch.float64))
        replicate_columns.extend(
            _bootstrap_effect_columns(role_ledgers[role], documents, multiplicities)
        )
    point = torch.tensor(point_values, dtype=torch.float64)
    replicates = torch.stack(replicate_columns, dim=1)
    maxima = (replicates - point).max(dim=1).values.sort().values
    critical = float(maxima[math.ceil(0.95 * repetitions) - 1])
    lower = point - critical
    names = tuple(
        f"{role}:{name}"
        for role in REPLICATION_ROLES
        for name in ("tau_positive", "specificity", "collateral_margin")
    )
    return ReplicationResult(
        roles=REPLICATION_ROLES,
        coordinate_names=names,
        point_estimates=point,
        simultaneous_lower_bounds=lower,
        critical_value=critical,
        passed=bool((lower > 0).all()),
    )
