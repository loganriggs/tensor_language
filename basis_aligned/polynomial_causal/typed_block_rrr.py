"""Reduced-rank shared factor maps for exact typed bilinear-block variables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from grouped_block_coefficient_screen import balance_product_gauge


TERM_NAMES = ("uu", "uv", "vu", "vv")
MASKS: Mapping[str, tuple[str, ...]] = {
    "all": TERM_NAMES,
    "no_vv": ("uu", "uv", "vu"),
    "no_cross": ("uu", "vv"),
    "cross_only": ("uv", "vu"),
    "uu_only": ("uu",),
}


def validate_second_moment(value: torch.Tensor, *, width: int | None = None) -> None:
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != value.shape[1] or (
        width is not None and tuple(value.shape) != (width, width)
    ) or not value.is_floating_point() or not bool(torch.isfinite(value).all()) or not (
        torch.allclose(value, value.T, atol=1e-9, rtol=1e-9)
    ):
        raise ValueError("second moment must be a finite symmetric square matrix")
    tolerance = torch.finfo(value.dtype).eps * max(value.shape) * max(
        float(torch.linalg.matrix_norm(value, ord=2)), 1.0,
    )
    if float(torch.linalg.eigvalsh(value.double()).min()) < -10 * tolerance:
        raise ValueError("second moment is not positive semidefinite")


def empirical_second_moment(*values: torch.Tensor) -> torch.Tensor:
    if not values or any(
        not torch.is_tensor(value) or value.ndim != 2 or not value.is_floating_point()
        for value in values
    ) or len({value.shape[1] for value in values}) != 1 or len(
        {value.dtype for value in values}
    ) != 1 or len({value.device for value in values}) != 1:
        raise ValueError("moment inputs must share a finite floating matrix interface")
    count = sum(value.shape[0] for value in values)
    if count <= 0 or any(not bool(torch.isfinite(value).all()) for value in values):
        raise ValueError("moment inputs are empty or nonfinite")
    moment = sum(value.T @ value for value in values) / count
    return (moment + moment.T) / 2


@dataclass(frozen=True, slots=True)
class ReducedRankFactorMap:
    encoder: torch.Tensor
    left_decoder: torch.Tensor
    right_decoder: torch.Tensor
    gate_weight: torch.Tensor
    rank: int
    covariance_support_rank: int
    objective_energy_fraction: float

    def __post_init__(self) -> None:
        tensors = (
            self.encoder, self.left_decoder, self.right_decoder, self.gate_weight,
        )
        if any(
            not torch.is_tensor(value) or not value.is_floating_point() or not bool(
                torch.isfinite(value).all()
            ) for value in tensors
        ) or self.encoder.ndim != 2 or self.left_decoder.ndim != 2 or (
            self.right_decoder.shape != self.left_decoder.shape
        ) or self.left_decoder.shape[1] != self.encoder.shape[0] or (
            self.gate_weight.shape != (self.left_decoder.shape[0],)
        ) or self.rank != self.encoder.shape[0] or not (
            1 <= self.rank <= self.encoder.shape[1]
        ) or not (self.rank <= self.covariance_support_rank <= self.encoder.shape[1]) or not (
            0.0 <= self.objective_energy_fraction <= 1.0 + 1e-12
        ):
            raise ValueError("reduced-rank factor map is malformed")

    @property
    def width(self) -> int:
        return self.encoder.shape[1]

    @property
    def gates(self) -> int:
        return self.left_decoder.shape[0]

    @property
    def factor_parameter_count(self) -> int:
        return self.rank * (self.width + 2 * self.gates)

    def factors(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if value.shape[-1] != self.width or value.dtype != self.encoder.dtype or (
            value.device != self.encoder.device
        ) or not bool(torch.isfinite(value).all()):
            raise ValueError("factor-map input is incompatible")
        code = torch.nn.functional.linear(value, self.encoder)
        return (
            torch.nn.functional.linear(code, self.left_decoder),
            torch.nn.functional.linear(code, self.right_decoder),
        )


def fit_reduced_rank_factor_map(
    left: torch.Tensor,
    right: torch.Tensor,
    down: torch.Tensor,
    second_moment: torch.Tensor,
    *,
    rank: int,
    relative_support_tolerance: float = 1e-10,
) -> ReducedRankFactorMap:
    """Fit the exact best rank-r map under a fixed empirical second moment.

    The objective is weighted factor-output squared error.  The closed form is the
    truncated SVD of ``A C^(1/2)``, represented as decoder times encoder without
    materializing a dense approximate map.
    """

    if left.ndim != 2 or right.shape != left.shape or down.shape != (
        left.shape[1], left.shape[0]
    ) or any(value.dtype != left.dtype or value.device != left.device for value in (
        right, down, second_moment,
    )) or type(rank) is not int or not (1 <= rank <= left.shape[1]) or not (
        isinstance(relative_support_tolerance, float)
        and 0 < relative_support_tolerance < 1
    ):
        raise ValueError("RRR factors, rank, or tolerance are incompatible")
    validate_second_moment(second_moment, width=left.shape[1])
    balanced_left, balanced_right, _ = balance_product_gauge(left, right)
    gate_weight = torch.linalg.vector_norm(down, dim=0)
    if bool((gate_weight <= torch.finfo(gate_weight.dtype).tiny).any()):
        raise ValueError("zero Down column has no weighted decoder inverse")
    weighted = torch.cat(
        (gate_weight[:, None] * balanced_left,
         gate_weight[:, None] * balanced_right), dim=0,
    )

    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment.double())
    eigenvalues = eigenvalues.clamp_min(0)
    maximum = float(eigenvalues.max())
    keep = eigenvalues > maximum * relative_support_tolerance
    support_rank = int(keep.sum())
    if support_rank < rank:
        raise ValueError("requested rank exceeds empirical covariance support")
    basis = eigenvectors[:, keep]
    root_values = torch.sqrt(eigenvalues[keep])
    root = (basis * root_values) @ basis.T
    inverse_root = (basis / root_values) @ basis.T

    weighted64 = weighted.double()
    gram = root @ (weighted64.T @ weighted64) @ root
    objective_values, objective_vectors = torch.linalg.eigh((gram + gram.T) / 2)
    objective_values = objective_values.clamp_min(0)
    order = torch.argsort(objective_values, descending=True)
    selected = objective_vectors[:, order[:rank]]
    total = objective_values.sum()
    retained = objective_values[order[:rank]].sum()

    encoder = (selected.T @ inverse_root).to(left.dtype)
    weighted_decoder = (weighted64 @ root @ selected).to(left.dtype)
    gates = left.shape[0]
    left_decoder = weighted_decoder[:gates] / gate_weight[:, None]
    right_decoder = weighted_decoder[gates:] / gate_weight[:, None]
    return ReducedRankFactorMap(
        encoder=encoder.contiguous(),
        left_decoder=left_decoder.contiguous(),
        right_decoder=right_decoder.contiguous(),
        gate_weight=gate_weight.contiguous(),
        rank=rank,
        covariance_support_rank=support_rank,
        objective_energy_fraction=float(retained / total),
    )


def native_factors(
    left: torch.Tensor, right: torch.Tensor, value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    balanced_left, balanced_right, _ = balance_product_gauge(left, right)
    return (
        torch.nn.functional.linear(value, balanced_left),
        torch.nn.functional.linear(value, balanced_right),
    )


def terms_from_factors(
    down: torch.Tensor,
    u_factors: tuple[torch.Tensor, torch.Tensor],
    v_factors: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    lu, ru = u_factors
    lv, rv = v_factors
    if lu.shape != ru.shape or lv.shape != rv.shape or lu.shape != lv.shape or (
        down.ndim != 2 or down.shape[1] != lu.shape[-1]
    ):
        raise ValueError("typed factor shapes are incompatible")
    products = {"uu": lu * ru, "uv": lu * rv, "vu": lv * ru, "vv": lv * rv}
    return {
        name: torch.nn.functional.linear(value, down)
        for name, value in products.items()
    }


def typed_terms(
    program: ReducedRankFactorMap,
    down: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if down.shape != (program.width, program.gates) or down.dtype != (
        program.encoder.dtype
    ) or down.device != program.encoder.device or u.shape != v.shape:
        raise ValueError("typed program inputs are incompatible")
    return terms_from_factors(down, program.factors(u), program.factors(v))


def masked_write(
    terms: Mapping[str, torch.Tensor], bias: torch.Tensor, mask: str,
) -> torch.Tensor:
    if set(terms) != set(TERM_NAMES) or mask not in MASKS or bias.ndim != 1:
        raise ValueError("typed terms, bias, or mask are malformed")
    selected = MASKS[mask]
    first = terms[selected[0]]
    if any(value.shape != first.shape for value in terms.values()) or (
        first.shape[-1] != len(bias)
    ) or bias.dtype != first.dtype or bias.device != first.device:
        raise ValueError("typed term outputs do not share their residual port")
    output = sum((terms[name] for name in selected), torch.zeros_like(first))
    shape = (1,) * (output.ndim - 1) + (len(bias),)
    return output + bias.reshape(shape)


def relative_error(observed: torch.Tensor, predicted: torch.Tensor) -> float:
    if observed.shape != predicted.shape or not (
        observed.is_floating_point() and predicted.is_floating_point()
    ):
        raise ValueError("relative-error tensors are incompatible")
    denominator = torch.linalg.vector_norm(observed.double()).clamp_min(
        torch.finfo(torch.float64).tiny,
    )
    return float(torch.linalg.vector_norm((observed - predicted).double()) / denominator)
