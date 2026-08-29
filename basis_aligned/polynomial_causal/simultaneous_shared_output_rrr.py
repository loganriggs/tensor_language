"""Exact CPU core for simultaneous reduced-rank regression.

For sites j=1,...,m, fit

    Y_j ~= X_j A_j V.T,       V.T V = I_q,

with one output basis V shared across sites and site-specific input maps A_j.
Only the sufficient statistics G_j=X_j.T X_j and C_j=X_j.T Y_j are needed.

For fixed V, ridge regression gives

    A_j = (G_j + lambda I)^(-1) C_j V.

Substitution into the penalized squared-error objective shows that the globally
optimal V spans the top-q eigenspace of

    M = sum_j C_j.T (G_j + lambda I)^(-1) C_j.

This is the predictive analogue of a shared output-mode HOSVD.  Unlike an SVD of
the raw stack of coefficient matrices, it weights directions by the observed
input covariance at each site.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Hashable
from typing import Sequence

import torch


@dataclass(frozen=True)
class MapPrice:
    n_sites: int
    input_dim: int
    output_dim: int
    rank: int
    separate_float_count: int
    shared_output_float_count: int
    separate_float_bytes: int
    shared_output_float_bytes: int
    saved_float_count: int
    saved_fraction: float
    multiplies_per_site: int


@dataclass(frozen=True)
class GroupedMapPrice:
    """Literal price for one output dictionary per specified site group."""

    n_sites: int
    n_output_bases: int
    input_dim: int
    output_dim: int
    rank: int
    separate_float_count: int
    grouped_float_count: int
    separate_float_bytes: int
    grouped_float_bytes: int
    saved_float_count: int
    saved_fraction: float
    multiplies_per_site: int


@dataclass(frozen=True)
class EqualStorageIndependentAllocation:
    """Fit-only independent rank allocation at exactly a grouped map's storage.

    Each independently factorized site consumes ``input_dim + output_dim`` floats
    per retained rank-one term.  The allocation takes the globally largest marginal
    predictive eigenvalues, with deterministic site/eigen-index tie breaking.  This
    is stronger than rounding one common independent rank and is exact whenever the
    grouped storage is divisible by the per-rank independent charge.
    """

    n_sites: int
    n_output_bases: int
    input_dim: int
    output_dim: int
    shared_rank: int
    grouped_float_budget: int
    independent_float_count: int
    total_rank_slots: int
    ranks_by_site: tuple[int, ...]
    selected_marginal_merit: float


def map_price(n_sites: int, input_dim: int, output_dim: int, rank: int) -> MapPrice:
    """Literal factor storage and dense multiply count for two map grammars."""
    for name, value in {
        "n_sites": n_sites,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank": rank,
    }.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if rank > min(input_dim, output_dim):
        raise ValueError("rank exceeds a matrix dimension")

    separate = n_sites * (input_dim * rank + rank * output_dim)
    shared = n_sites * input_dim * rank + rank * output_dim
    saved = separate - shared
    return MapPrice(
        n_sites=n_sites,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        separate_float_count=separate,
        shared_output_float_count=shared,
        separate_float_bytes=4 * separate,
        shared_output_float_bytes=4 * shared,
        saved_float_count=saved,
        saved_fraction=saved / separate,
        multiplies_per_site=input_dim * rank + rank * output_dim,
    )


def grouped_map_price(
    n_sites: int,
    n_output_bases: int,
    input_dim: int,
    output_dim: int,
    rank: int,
) -> GroupedMapPrice:
    """Price site-specific input maps plus one output basis for each group.

    ``n_output_bases=1`` is the global shared-dictionary grammar and
    ``n_output_bases=n_sites`` has the same storage as independent factors.
    """
    for name, value in {
        "n_sites": n_sites,
        "n_output_bases": n_output_bases,
        "input_dim": input_dim,
        "output_dim": output_dim,
        "rank": rank,
    }.items():
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_output_bases > n_sites:
        raise ValueError("n_output_bases exceeds n_sites")
    if rank > min(input_dim, output_dim):
        raise ValueError("rank exceeds a matrix dimension")

    separate = n_sites * (input_dim * rank + rank * output_dim)
    grouped = n_sites * input_dim * rank + n_output_bases * rank * output_dim
    saved = separate - grouped
    return GroupedMapPrice(
        n_sites=n_sites,
        n_output_bases=n_output_bases,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=rank,
        separate_float_count=separate,
        grouped_float_count=grouped,
        separate_float_bytes=4 * separate,
        grouped_float_bytes=4 * grouped,
        saved_float_count=saved,
        saved_fraction=saved / separate,
        multiplies_per_site=input_dim * rank + rank * output_dim,
    )


def allocate_equal_storage_independent_ranks(
    marginal_eigenvalues: Sequence[torch.Tensor],
    *,
    n_output_bases: int,
    input_dim: int,
    output_dim: int,
    shared_rank: int,
) -> EqualStorageIndependentAllocation:
    """Allocate exact matched-storage independent ranks from fit-only spectra.

    ``marginal_eigenvalues[j][k]`` is the fit-objective gain of adding the
    ``(k+1)``th reduced-rank direction at site ``j``.  Every spectrum must be a
    finite, nonnegative, nonincreasing float64 CPU vector.  Ranking all marginal
    gains is optimal because independent-site reduced-rank objectives add.
    """
    n_sites = len(marginal_eigenvalues)
    price = grouped_map_price(
        n_sites=n_sites,
        n_output_bases=n_output_bases,
        input_dim=input_dim,
        output_dim=output_dim,
        rank=shared_rank,
    )
    per_slot = input_dim + output_dim
    total_slots, remainder = divmod(price.grouped_float_count, per_slot)
    if remainder:
        raise ValueError("grouped storage cannot be matched by integer independent ranks")

    candidates: list[tuple[float, int, int]] = []
    for site, values in enumerate(marginal_eigenvalues):
        if (
            not torch.is_tensor(values)
            or values.device.type != "cpu"
            or values.dtype != torch.float64
            or values.ndim != 1
            or values.numel() == 0
        ):
            raise ValueError("each marginal spectrum must be a nonempty CPU float64 vector")
        if values.numel() > min(input_dim, output_dim):
            raise ValueError("a marginal spectrum exceeds the maximum site rank")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("marginal spectra must be finite and nonnegative")
        scale = max(float(values.abs().max()), 1.0)
        if values.numel() > 1 and bool((values[1:] - values[:-1] > 1e-12 * scale).any()):
            raise ValueError("marginal spectra must be nonincreasing")
        candidates.extend((float(value), site, index) for index, value in enumerate(values))
    if total_slots > len(candidates):
        raise ValueError("marginal spectra do not contain enough rank slots for the budget")

    # Python's stable tuple order supplies the frozen tie break: larger merit first,
    # then smaller site, then smaller within-site eigen-index.
    selected = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[:total_slots]
    ranks = [0] * n_sites
    selected_indices: list[list[int]] = [[] for _ in range(n_sites)]
    for _, site, index in selected:
        ranks[site] += 1
        selected_indices[site].append(index)
    for indices in selected_indices:
        if indices and indices != list(range(len(indices))):
            raise RuntimeError("optimal marginal allocation violated a spectral prefix")

    independent_count = per_slot * sum(ranks)
    if independent_count != price.grouped_float_count:
        raise RuntimeError("independent allocation failed exact storage matching")
    return EqualStorageIndependentAllocation(
        n_sites=n_sites,
        n_output_bases=n_output_bases,
        input_dim=input_dim,
        output_dim=output_dim,
        shared_rank=shared_rank,
        grouped_float_budget=price.grouped_float_count,
        independent_float_count=independent_count,
        total_rank_slots=total_slots,
        ranks_by_site=tuple(ranks),
        selected_marginal_merit=sum(value for value, _, _ in selected),
    )


def _validate_statistics(
    grams: Sequence[torch.Tensor],
    crosses: Sequence[torch.Tensor],
    rank: int,
    ridge: float,
) -> tuple[int, int, torch.dtype, torch.device]:
    if not grams or len(grams) != len(crosses):
        raise ValueError("grams and crosses must have the same nonzero length")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("rank must be a positive integer")
    if not isinstance(ridge, (int, float)) or not torch.isfinite(torch.tensor(ridge)):
        raise ValueError("ridge must be finite")
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")

    first_g, first_c = grams[0], crosses[0]
    if first_g.ndim != 2 or first_g.shape[0] != first_g.shape[1]:
        raise ValueError("each Gram matrix must be square")
    if first_c.ndim != 2 or first_c.shape[0] != first_g.shape[0]:
        raise ValueError("each cross matrix must have shape [input_dim, output_dim]")
    input_dim, output_dim = first_c.shape
    if rank > output_dim:
        raise ValueError("rank exceeds output dimension")
    if first_g.dtype != torch.float64 or first_c.dtype != torch.float64:
        raise ValueError("statistics must be float64")
    dtype, device = first_g.dtype, first_g.device

    for index, (gram, cross) in enumerate(zip(grams, crosses, strict=True)):
        if gram.shape != (input_dim, input_dim):
            raise ValueError(f"Gram {index} has inconsistent shape")
        if cross.shape != (input_dim, output_dim):
            raise ValueError(f"cross {index} has inconsistent shape")
        if gram.dtype != dtype or cross.dtype != dtype:
            raise ValueError("all statistics must share float64 dtype")
        if gram.device != device or cross.device != device:
            raise ValueError("all statistics must share one device")
        if not torch.isfinite(gram).all() or not torch.isfinite(cross).all():
            raise ValueError("statistics must be finite")
        scale = max(float(gram.abs().max()), 1.0)
        if float((gram - gram.T).abs().max()) > 1e-11 * scale:
            raise ValueError(f"Gram {index} is not symmetric")
        regularized = 0.5 * (gram + gram.T) + ridge * torch.eye(
            input_dim, dtype=dtype, device=device
        )
        _, info = torch.linalg.cholesky_ex(regularized)
        if int(info) != 0:
            raise ValueError(f"regularized Gram {index} is not positive definite")
    return input_dim, output_dim, dtype, device


def fit_shared_output_basis(
    grams: Sequence[torch.Tensor],
    crosses: Sequence[torch.Tensor],
    rank: int,
    ridge: float,
) -> dict[str, object]:
    """Return the exact optimum for a common output basis.

    The returned ``basis`` has shape [output_dim, rank], every ``input_map`` has
    shape [input_dim, rank], and the deployed coefficient matrix at site j is
    ``input_maps[j] @ basis.T``.
    """
    input_dim, output_dim, dtype, device = _validate_statistics(
        grams, crosses, rank, ridge
    )
    identity = torch.eye(input_dim, dtype=dtype, device=device)
    solved_crosses: list[torch.Tensor] = []
    merit = torch.zeros((output_dim, output_dim), dtype=dtype, device=device)
    for gram, cross in zip(grams, crosses, strict=True):
        solved = torch.linalg.solve(0.5 * (gram + gram.T) + ridge * identity, cross)
        solved_crosses.append(solved)
        merit.add_(cross.T @ solved)
    merit = 0.5 * (merit + merit.T)

    eigenvalues, eigenvectors = torch.linalg.eigh(merit)
    order = torch.arange(output_dim - 1, output_dim - rank - 1, -1, device=device)
    selected_values = eigenvalues.index_select(0, order)
    basis = eigenvectors.index_select(1, order).contiguous()
    input_maps = [(solved @ basis).contiguous() for solved in solved_crosses]

    projector = basis @ basis.T
    projection_error = float((projector @ projector - projector).abs().max())
    orthogonality_error = float(
        (basis.T @ basis - torch.eye(rank, dtype=dtype, device=device)).abs().max()
    )
    return {
        "basis": basis,
        "input_maps": input_maps,
        "coefficient_maps": [input_map @ basis.T for input_map in input_maps],
        "selected_eigenvalues": selected_values,
        "explained_penalized_fit": float(selected_values.sum()),
        "merit": merit,
        "projector": projector,
        "projector_idempotence_max_abs": projection_error,
        "basis_orthogonality_max_abs": orthogonality_error,
    }


def fit_grouped_output_bases(
    grams: Sequence[torch.Tensor],
    crosses: Sequence[torch.Tensor],
    groups: Sequence[Hashable],
    rank: int,
    ridge: float,
) -> dict[str, object]:
    """Fit one exact shared output basis per prospectively fixed site group.

    The returned coefficient maps remain in the original site order.  Grouping all
    sites together exactly reproduces :func:`fit_shared_output_basis`; assigning a
    separate group to every site reproduces independent reduced-rank regression.
    """
    _validate_statistics(grams, crosses, rank, ridge)
    if len(groups) != len(grams):
        raise ValueError("groups must contain one label per site")
    if any(not isinstance(label, Hashable) for label in groups):
        raise ValueError("every group label must be hashable")

    ordered_labels = list(dict.fromkeys(groups))
    coefficient_maps: list[torch.Tensor | None] = [None] * len(grams)
    bases: dict[Hashable, torch.Tensor] = {}
    input_maps: dict[Hashable, list[torch.Tensor]] = {}
    selected_eigenvalues: dict[Hashable, torch.Tensor] = {}
    group_indices: dict[Hashable, list[int]] = {}
    explained = 0.0

    for label in ordered_labels:
        indices = [index for index, value in enumerate(groups) if value == label]
        fit = fit_shared_output_basis(
            [grams[index] for index in indices],
            [crosses[index] for index in indices],
            rank=rank,
            ridge=ridge,
        )
        bases[label] = fit["basis"]
        input_maps[label] = fit["input_maps"]
        selected_eigenvalues[label] = fit["selected_eigenvalues"]
        group_indices[label] = indices
        explained += float(fit["explained_penalized_fit"])
        for index, coefficient in zip(indices, fit["coefficient_maps"], strict=True):
            coefficient_maps[index] = coefficient

    if any(value is None for value in coefficient_maps):
        raise RuntimeError("group fit did not populate every site")
    return {
        "group_order": ordered_labels,
        "group_indices": group_indices,
        "bases": bases,
        "input_maps": input_maps,
        "selected_eigenvalues": selected_eigenvalues,
        "coefficient_maps": coefficient_maps,
        "explained_penalized_fit": explained,
    }


def penalized_objective_from_statistics(
    grams: Sequence[torch.Tensor],
    crosses: Sequence[torch.Tensor],
    y_squared_frobenius: Sequence[float],
    basis: torch.Tensor,
    input_maps: Sequence[torch.Tensor],
    ridge: float,
) -> float:
    """Evaluate sum_j ||Y-X A V.T||_F^2 + ridge ||A||_F^2 from stats."""
    if not (len(grams) == len(crosses) == len(y_squared_frobenius) == len(input_maps)):
        raise ValueError("all site sequences must have equal length")
    if basis.dtype != torch.float64 or basis.ndim != 2:
        raise ValueError("basis must be a float64 matrix")
    total = 0.0
    for gram, cross, y2, input_map in zip(
        grams, crosses, y_squared_frobenius, input_maps, strict=True
    ):
        if not torch.isfinite(torch.tensor(y2)) or y2 < 0:
            raise ValueError("Y squared Frobenius values must be finite and nonnegative")
        if input_map.shape != (gram.shape[0], basis.shape[1]):
            raise ValueError("input map has inconsistent shape")
        fit = input_map @ basis.T
        term = (
            float(y2)
            - 2.0 * float((fit * cross).sum())
            + float(torch.trace(fit.T @ gram @ fit))
            + ridge * float(input_map.square().sum())
        )
        total += term
    return total


def canonical_price_receipt() -> dict[str, object]:
    """Exact price implication for the current 36-site, d=1152, rank-512 maps."""
    return asdict(map_price(n_sites=36, input_dim=1152, output_dim=1152, rank=512))


if __name__ == "__main__":
    import json

    print(json.dumps(canonical_price_receipt(), indent=2, sort_keys=True))
