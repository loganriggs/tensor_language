"""Gauge-fixed mathematical primitives for Block-3 consequence fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

import native_gate_subset as subset


SCORE_PROJECTION_ITERATIONS = 64
REPLAY_RELATIVE_LIMIT = 2e-5


def project_capped_simplex(
    value: torch.Tensor, budget: int, *, iterations: int = SCORE_PROJECTION_ITERATIONS,
) -> torch.Tensor:
    """Euclidean projection onto ``[0,1]^n`` with an exact target sum.

    The KKT solution has ``projected = clamp(value - tau, 0, 1)``.  Fixed-count
    bisection makes the implementation deterministic on CPU and CUDA.
    """

    if value.ndim != 1 or not value.is_floating_point() or not bool(
        torch.isfinite(value).all()
    ) or type(budget) is not int or not (0 <= budget <= value.numel()) or (
        type(iterations) is not int or iterations < 32
    ):
        raise ValueError("capped-simplex value, budget, or iteration count is malformed")
    work = value.double()
    if budget == 0:
        return torch.zeros_like(work)
    if budget == value.numel():
        return torch.ones_like(work)
    lower = work.min() - 1.0
    upper = work.max()
    for _ in range(iterations):
        threshold = (lower + upper) / 2
        total = torch.clamp(work - threshold, 0.0, 1.0).sum()
        if float(total) > budget:
            lower = threshold
        else:
            upper = threshold
    threshold = (lower + upper) / 2
    projected = torch.clamp(work - threshold, 0.0, 1.0)
    if not bool(torch.isfinite(projected).all()) or abs(
        float(projected.double().sum()) - budget
    ) > 1e-10 or float(projected.min()) < 0 or float(projected.max()) > 1 or not (
        torch.equal(projected, torch.clamp(work - threshold, 0.0, 1.0))
    ):
        raise RuntimeError("capped-simplex projection failed its numerical contract")
    return projected


def stable_nested_supports(
    scores: torch.Tensor, global_indices: torch.Tensor, budgets: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Rank scores with global native-gate index as the deterministic tie break."""

    if scores.ndim != 1 or global_indices.shape != scores.shape or (
        global_indices.dtype != torch.long
    ) or not bool(torch.isfinite(scores).all()) or len(torch.unique(global_indices)) != len(
        global_indices
    ) or not budgets or any(type(k) is not int or not 1 <= k <= len(scores) for k in budgets):
        raise ValueError("scores, global indices, or budgets are malformed")
    order = sorted(
        range(len(scores)),
        key=lambda local: (-float(scores[local]), int(global_indices[local])),
    )
    local_order = torch.tensor(order, dtype=torch.long, device=global_indices.device)
    return {
        budget: global_indices[local_order[:budget]].clone()
        for budget in sorted(set(budgets))
    }


def source_document_row_weights(row_to_document: torch.Tensor) -> torch.Tensor:
    """Give every source document total weight one, independent of cached row count."""

    if row_to_document.ndim != 1 or row_to_document.dtype != torch.long or len(
        row_to_document
    ) == 0 or int(row_to_document.min()) != 0:
        raise ValueError("row-to-document mapping is malformed")
    counts = torch.bincount(row_to_document)
    if len(counts) != int(row_to_document.max()) + 1 or bool((counts == 0).any()):
        raise ValueError("row-to-document mapping is not contiguous")
    return counts[row_to_document].reciprocal().to(torch.float64)


def document_deranged_row_map(row_to_document: torch.Tensor) -> torch.Tensor:
    """Pair every target row with a row from a deterministically shifted document.

    Documents retain first-occurrence order.  Document ``d`` receives teacher labels
    from ``(d + floor(D/2)) mod D``; unequal row counts are handled by cycling the
    target row's within-document occurrence through the donor's rows.
    """

    source_document_row_weights(row_to_document)  # validates contiguous identities
    documents = int(row_to_document.max()) + 1
    if documents < 2:
        raise ValueError("a document derangement requires at least two documents")
    rows_by_document = [
        torch.nonzero(row_to_document == document, as_tuple=False).flatten()
        for document in range(documents)
    ]
    shift = documents // 2
    output = torch.empty_like(row_to_document)
    occurrence = torch.zeros(documents, dtype=torch.long)
    for target_row, document in enumerate(row_to_document.tolist()):
        donor_document = (document + shift) % documents
        donor_rows = rows_by_document[donor_document]
        output[target_row] = donor_rows[int(occurrence[document] % len(donor_rows))]
        occurrence[document] += 1
    if bool((row_to_document[output] == row_to_document).any()) or len(output) != len(
        row_to_document
    ):
        raise RuntimeError("document derangement retained a source-document identity")
    return output


def teacher_kl_by_row(
    teacher_logits: torch.Tensor, student_logits: torch.Tensor,
) -> torch.Tensor:
    """Native-to-student KL, averaged over token positions separately for each row."""

    if teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3 or not (
        teacher_logits.is_floating_point() and student_logits.is_floating_point()
    ) or not bool(torch.isfinite(teacher_logits).all() and torch.isfinite(student_logits).all()):
        raise ValueError("teacher/student logits are malformed")
    teacher_logp = F.log_softmax(teacher_logits, -1)
    student_logp = F.log_softmax(student_logits, -1)
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1).mean(-1)


def document_balanced_batch_loss(
    row_losses: torch.Tensor, row_weights: torch.Tensor, *, document_count: int,
) -> torch.Tensor:
    """Return this batch's additive contribution to the whole-document mean."""

    if row_losses.ndim != 1 or row_weights.shape != row_losses.shape or not (
        row_losses.is_floating_point() and row_weights.is_floating_point()
    ) or not bool(torch.isfinite(row_losses).all() and torch.isfinite(row_weights).all()) or (
        type(document_count) is not int or document_count <= 0
    ):
        raise ValueError("document-balanced loss inputs are malformed")
    return (row_losses * row_weights.to(row_losses)).sum() / document_count


def logical_batch_adam_step(
    optimizer: torch.optim.Optimizer, parameters: Iterable[torch.Tensor],
    additive_microbatch_losses: Iterable[torch.Tensor], *, max_grad_norm: float = 1.0,
) -> float:
    """Accumulate every microbatch, then clip and update exactly once."""

    parameter_list = list(parameters)
    losses = list(additive_microbatch_losses)
    optimized = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if not isinstance(optimizer, torch.optim.Adam) or not parameter_list or len(losses) != 4 or (
        len({id(parameter) for parameter in parameter_list}) != len(parameter_list)
    ) or len(optimized) != len(parameter_list) or {
        id(parameter) for parameter in optimized
    } != {
        id(parameter) for parameter in parameter_list
    } or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("logical Adam step inputs are malformed")
    optimizer.zero_grad(set_to_none=True)
    for loss in losses:
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise RuntimeError("logical batch contains a nonfinite microbatch loss")
        loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(parameter_list, max_grad_norm)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("logical batch gradient norm is nonfinite")
    optimizer.step()
    return float(norm)


def logical_batch_adam_closure_step(
    optimizer: torch.optim.Optimizer, parameters: Iterable[torch.Tensor],
    additive_microbatch_loss_closures: Sequence[Callable[[], torch.Tensor]],
    *, max_grad_norm: float = 1.0,
) -> float:
    """Backpropagate four lazily built microbatch graphs, then clip/update once.

    This has the same gradient and Adam trajectory as :func:`logical_batch_adam_step`
    but releases each suffix graph before constructing the next one.
    """

    parameter_list = list(parameters)
    closures = tuple(additive_microbatch_loss_closures)
    optimized = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if not isinstance(optimizer, torch.optim.Adam) or not parameter_list or len(
        closures
    ) != 4 or not all(callable(closure) for closure in closures) or (
        len({id(parameter) for parameter in parameter_list}) != len(parameter_list)
    ) or len(optimized) != len(parameter_list) or {
        id(parameter) for parameter in optimized
    } != {
        id(parameter) for parameter in parameter_list
    } or not math.isfinite(max_grad_norm) or max_grad_norm <= 0:
        raise ValueError("logical Adam closure step inputs are malformed")
    optimizer.zero_grad(set_to_none=True)
    for closure in closures:
        loss = closure()
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise RuntimeError("logical batch contains a nonfinite microbatch loss")
        loss.backward()
        del loss
    norm = torch.nn.utils.clip_grad_norm_(parameter_list, max_grad_norm)
    if not bool(torch.isfinite(norm)):
        raise RuntimeError("logical batch gradient norm is nonfinite")
    optimizer.step()
    return float(norm)


def softcap_scored_raw_logits(
    raw_logits: torch.Tensor, *, start: int = 64, stop: int = 256,
) -> torch.Tensor:
    """Slice raw LM-head logits and apply bilin18's softcap exactly once."""

    logits = raw_logits
    if logits.ndim != 3 or not 0 <= start < stop <= logits.shape[1]:
        raise ValueError("logit scoring slice is malformed")
    return (30.0 * torch.tanh(logits[:, start:stop] / 30.0)).float()


def consequence_score_write(
    value: torch.Tensor, balanced_left: torch.Tensor, balanced_right: torch.Tensor,
    native_down: torch.Tensor, native_bias: torch.Tensor,
    prefilter_indices: torch.Tensor, scores: torch.Tensor,
) -> torch.Tensor:
    """The identifiable F1 write: fixed native decoder, continuous gate scores."""

    if prefilter_indices.ndim != 1 or prefilter_indices.dtype != torch.long or (
        scores.shape != prefilter_indices.shape
    ) or value.shape[-1] != balanced_left.shape[1] or balanced_right.shape != (
        balanced_left.shape
    ) or native_down.shape != (value.shape[-1], balanced_left.shape[0]) or (
        native_bias.shape != (value.shape[-1],)
    ):
        raise ValueError("consequence-score write tensors are incompatible")
    left = F.linear(value, balanced_left[prefilter_indices])
    right = F.linear(value, balanced_right[prefilter_indices])
    features = left * right * scores.to(value).reshape(
        *((1,) * (value.ndim - 1)), -1,
    )
    return F.linear(features, native_down[:, prefilter_indices], native_bias)


def refit_joint_program(
    *, left: torch.Tensor, right: torch.Tensor, bias: torch.Tensor,
    prefilter_indices: torch.Tensor, gram: torch.Tensor, cross: torch.Tensor,
    global_support: torch.Tensor, relative_ridge: float = 1e-6,
) -> subset.NativeGateSubsetProgram:
    """Analytically refit the common four-term decoder on a selected global support."""

    lookup = {int(gate): local for local, gate in enumerate(prefilter_indices.tolist())}
    try:
        local = torch.tensor(
            [lookup[int(gate)] for gate in global_support.tolist()],
            dtype=torch.long, device=gram.device,
        )
    except KeyError as error:
        raise ValueError("support contains a gate outside the sealed prefilter") from error
    decoder = subset.fit_joint_decoder(
        gram[local][:, local], cross[local], relative_ridge=relative_ridge,
    ).to(dtype=left.dtype, device=left.device)
    return subset.build_program(
        left, right, bias, global_support.to(left.device), decoder,
    )


def fold_affine_calibration(
    program: subset.NativeGateSubsetProgram, scale: torch.Tensor | float,
    correction: torch.Tensor,
) -> subset.NativeGateSubsetProgram:
    """Fold ``b+c+a*(w-b)`` without adding an array or deployed operation."""

    scalar = torch.as_tensor(scale, dtype=program.decoder.dtype, device=program.decoder.device)
    if scalar.numel() != 1 or correction.shape != program.bias.shape or not bool(
        torch.isfinite(scalar).all() and torch.isfinite(correction).all()
    ):
        raise ValueError("affine calibration is malformed")
    return subset.NativeGateSubsetProgram(
        indices=program.indices.clone(), left=program.left.clone(), right=program.right.clone(),
        decoder=(program.decoder * scalar).contiguous(),
        bias=(program.bias + correction.to(program.bias)).contiguous(),
    )


def program_price(program: subset.NativeGateSubsetProgram) -> dict[str, int]:
    return {
        "float_values": program.float_parameter_count,
        "float_bytes": program.float_parameter_count * program.left.element_size(),
        "index_bytes": program.indices.numel() * program.indices.element_size(),
        "total_bytes": (
            program.float_parameter_count * program.left.element_size()
            + program.indices.numel() * program.indices.element_size()
        ),
        "products_per_token": program.product_count_per_token,
        "linear_multiplies_per_token": 3 * program.width * program.gates,
    }


@dataclass(frozen=True, slots=True)
class ScoreTrace:
    epoch: int
    document_kl: float
    row_kl: float
    score_min: float
    score_max: float
    score_sum: float
    saturated_zero: float
    saturated_one: float
    gradient_norm_max: float

    def __post_init__(self) -> None:
        values = (
            self.document_kl, self.row_kl, self.score_min, self.score_max,
            self.score_sum, self.saturated_zero, self.saturated_one,
            self.gradient_norm_max,
        )
        if self.epoch < 0 or not all(math.isfinite(value) for value in values):
            raise ValueError("nonfinite or malformed consequence-fit trace")
