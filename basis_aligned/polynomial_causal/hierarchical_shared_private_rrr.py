"""Exact CPU core for a fixed shared trunk plus site-private residual RRR.

For site merits M_j = C_j.T (G_j + lambda I)^-1 C_j and a prospectively
fixed orthonormal shared basis V0, the site output basis is [V0, U_j], with
U_j.T V0 = 0.  Private directions are the leading eigenvectors of

    Q M_j Q,  Q = I - V0 V0.T,

restricted to range(Q).  At fixed literal storage, private rank slots are
allocated by globally largest residual eigenvalue, with deterministic ties.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class HierarchicalPrice:
    n_sites: int
    dimension: int
    shared_rank: int
    private_ranks: tuple[int, ...]
    shared_input_float_count: int
    shared_output_float_count: int
    private_float_count: int
    map_float_count: int
    map_float_bytes: int
    dense_multiplies_by_site: tuple[int, ...]
    dense_multiplies_per_uncovered_token: int


@dataclass(frozen=True)
class PrivateRankAllocation:
    n_sites: int
    dimension: int
    shared_rank: int
    total_float_budget: int
    shared_float_count: int
    private_float_budget: int
    private_rank_slots: int
    private_ranks: tuple[int, ...]
    selected_residual_merit: float


@dataclass(frozen=True)
class ResidualEigensystem:
    complement_basis: torch.Tensor
    eigenvalues: tuple[torch.Tensor, ...]
    eigenvectors: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class HierarchicalFit:
    shared_basis: torch.Tensor
    shared_input_maps: tuple[torch.Tensor, ...]
    private_bases: tuple[torch.Tensor, ...]
    private_input_maps: tuple[torch.Tensor, ...]
    residual_eigenvalues: tuple[torch.Tensor, ...]
    allocation: PrivateRankAllocation
    price: HierarchicalPrice
    explained_shared_merit: float
    explained_private_merit: float
    combined_orthogonality_max_abs: float


@dataclass(frozen=True)
class DeployedHierarchicalProgram:
    """Exact float32 factor order used by any future intervention adapter."""

    shared_basis: torch.Tensor
    shared_input_maps: tuple[torch.Tensor, ...]
    private_bases: tuple[torch.Tensor, ...]
    private_input_maps: tuple[torch.Tensor, ...]
    private_ranks: tuple[int, ...]


def _positive_integer(name: str, value: Any, *, allow_zero: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be {'a nonnegative' if allow_zero else 'a positive'} integer")
    return value


def hierarchical_price(
    n_sites: int, dimension: int, shared_rank: int,
    private_ranks: Sequence[int],
) -> HierarchicalPrice:
    n_sites = _positive_integer("n_sites", n_sites)
    dimension = _positive_integer("dimension", dimension)
    shared_rank = _positive_integer("shared_rank", shared_rank, allow_zero=True)
    if shared_rank > dimension or len(private_ranks) != n_sites:
        raise ValueError("hierarchical rank dimensions are inconsistent")
    ranks = tuple(_positive_integer("private rank", value, allow_zero=True)
                  for value in private_ranks)
    if any(value > dimension - shared_rank for value in ranks):
        raise ValueError("a private rank exceeds the shared orthogonal complement")
    shared_inputs = n_sites * dimension * shared_rank
    shared_output = dimension * shared_rank
    private = 2 * dimension * sum(ranks)
    total = shared_inputs + shared_output + private
    multiplies = tuple(2 * dimension * (shared_rank + value) for value in ranks)
    return HierarchicalPrice(
        n_sites=n_sites, dimension=dimension, shared_rank=shared_rank,
        private_ranks=ranks, shared_input_float_count=shared_inputs,
        shared_output_float_count=shared_output, private_float_count=private,
        map_float_count=total, map_float_bytes=4 * total,
        dense_multiplies_by_site=multiplies,
        dense_multiplies_per_uncovered_token=sum(multiplies),
    )


def _validate_merits(merits: Sequence[torch.Tensor]) -> tuple[int, torch.dtype]:
    if not merits:
        raise ValueError("merits must be nonempty")
    first = merits[0]
    if not torch.is_tensor(first) or first.ndim != 2 or first.shape[0] != first.shape[1]:
        raise ValueError("each merit must be a square matrix")
    dimension = first.shape[0]
    if dimension == 0:
        raise ValueError("merit dimension must be positive")
    for index, merit in enumerate(merits):
        if not torch.is_tensor(merit) or merit.shape != (dimension, dimension) or (
            merit.dtype != torch.float64
        ) or (
            merit.device.type != "cpu"
        ) or not bool(torch.isfinite(merit).all()):
            raise ValueError("all merits must be finite CPU float64 matrices of one shape")
        scale = max(float(merit.abs().max()), 1.0)
        if float((merit - merit.T).abs().max()) > 1e-11 * scale:
            raise ValueError(f"merit {index} is not symmetric")
        minimum = float(torch.linalg.eigvalsh(0.5 * (merit + merit.T))[0])
        if minimum < -1e-10 * scale:
            raise ValueError(f"merit {index} is materially indefinite")
    return dimension, torch.float64


def _validate_shared_basis(basis: torch.Tensor, dimension: int) -> int:
    if not torch.is_tensor(basis) or basis.dtype != torch.float64 or basis.device.type != "cpu" or (
        basis.ndim != 2 or basis.shape[0] != dimension or not bool(torch.isfinite(basis).all())
    ):
        raise ValueError("shared basis must be a finite CPU float64 [dimension, rank] matrix")
    rank = basis.shape[1]
    if rank > dimension:
        raise ValueError("shared rank exceeds dimension")
    identity = torch.eye(rank, dtype=torch.float64)
    if rank and float((basis.T @ basis - identity).abs().max()) > 1e-10:
        raise ValueError("shared basis is not orthonormal")
    return rank


def global_shared_basis(merits: Sequence[torch.Tensor], rank: int) -> torch.Tensor:
    """Top projector representative of the summed site merit, fixed from fit only."""
    dimension, _ = _validate_merits(merits)
    rank = _positive_integer("rank", rank, allow_zero=True)
    if rank > dimension:
        raise ValueError("global shared rank exceeds dimension")
    if rank == 0:
        return torch.empty((dimension, 0), dtype=torch.float64)
    total = torch.stack(tuple(merits)).sum(0)
    values, vectors = torch.linalg.eigh(0.5 * (total + total.T))
    del values
    return vectors[:, -rank:].flip(1).contiguous()


def orthogonal_complement(shared_basis: torch.Tensor) -> torch.Tensor:
    dimension = shared_basis.shape[0] if torch.is_tensor(shared_basis) and shared_basis.ndim == 2 else -1
    shared_rank = _validate_shared_basis(shared_basis, dimension)
    if shared_rank == 0:
        return torch.eye(dimension, dtype=torch.float64)
    if shared_rank == dimension:
        return torch.empty((dimension, 0), dtype=torch.float64)
    complete, _ = torch.linalg.qr(shared_basis, mode="complete")
    complement = complete[:, shared_rank:].contiguous()
    if float((shared_basis.T @ complement).abs().max()) > 1e-10:
        raise RuntimeError("constructed complement is not orthogonal to shared basis")
    return complement


def residual_eigensystems(
    merits: Sequence[torch.Tensor], shared_basis: torch.Tensor,
) -> ResidualEigensystem:
    dimension, _ = _validate_merits(merits)
    shared_rank = _validate_shared_basis(shared_basis, dimension)
    complement = orthogonal_complement(shared_basis)
    residual_dimension = dimension - shared_rank
    values_out: list[torch.Tensor] = []
    vectors_out: list[torch.Tensor] = []
    for merit in merits:
        if residual_dimension == 0:
            values = torch.empty(0, dtype=torch.float64)
            vectors = torch.empty((dimension, 0), dtype=torch.float64)
        else:
            reduced = complement.T @ merit @ complement
            values, reduced_vectors = torch.linalg.eigh(0.5 * (reduced + reduced.T))
            values = values.flip(0).contiguous()
            reduced_vectors = reduced_vectors.flip(1).contiguous()
            tolerance = 1e-10 * max(float(values.abs().max()), 1.0)
            if float(values.min()) < -tolerance:
                raise ValueError("a residual merit is materially indefinite")
            values = values.clamp_min(0)
            vectors = (complement @ reduced_vectors).contiguous()
        values_out.append(values)
        vectors_out.append(vectors)
    return ResidualEigensystem(
        complement_basis=complement, eigenvalues=tuple(values_out),
        eigenvectors=tuple(vectors_out),
    )


def allocate_private_ranks(
    residual_eigenvalues: Sequence[torch.Tensor], *, dimension: int,
    shared_rank: int, total_float_budget: int,
) -> PrivateRankAllocation:
    n_sites = len(residual_eigenvalues)
    _positive_integer("n_sites", n_sites)
    dimension = _positive_integer("dimension", dimension)
    shared_rank = _positive_integer("shared_rank", shared_rank, allow_zero=True)
    total_float_budget = _positive_integer("total_float_budget", total_float_budget,
                                           allow_zero=True)
    if shared_rank > dimension:
        raise ValueError("shared rank exceeds dimension")
    residual_dimension = dimension - shared_rank
    candidates: list[tuple[float, int, int]] = []
    for site, values in enumerate(residual_eigenvalues):
        if not torch.is_tensor(values) or values.dtype != torch.float64 or values.device.type != (
            "cpu"
        ) or values.ndim != 1 or values.numel() != residual_dimension or not bool(
            torch.isfinite(values).all()
        ) or bool((values < 0).any()):
            raise ValueError("residual spectra have invalid schema")
        scale = max(float(values.abs().max()), 1.0) if values.numel() else 1.0
        if values.numel() > 1 and bool((values[1:] - values[:-1] > 1e-12 * scale).any()):
            raise ValueError("residual spectra must be nonincreasing")
        candidates.extend((float(value), site, index) for index, value in enumerate(values))
    shared_floats = dimension * (n_sites + 1) * shared_rank
    private_budget = total_float_budget - shared_floats
    if private_budget < 0:
        raise ValueError("total budget is smaller than the shared trunk")
    slot_cost = 2 * dimension
    slots, remainder = divmod(private_budget, slot_cost)
    if remainder:
        raise ValueError("total budget cannot be expressed as literal private rank slots")
    if slots > len(candidates):
        raise ValueError("total budget exceeds all available private ranks")
    selected = sorted(candidates, key=lambda item: (-item[0], item[1], item[2]))[:slots]
    ranks = [0] * n_sites
    indices: list[list[int]] = [[] for _ in range(n_sites)]
    for _, site, index in selected:
        ranks[site] += 1
        indices[site].append(index)
    if any(value and value != list(range(len(value))) for value in indices):
        raise RuntimeError("private allocation violated a spectral prefix")
    price = hierarchical_price(n_sites, dimension, shared_rank, ranks)
    if price.map_float_count != total_float_budget:
        raise RuntimeError("private allocation missed its literal storage budget")
    return PrivateRankAllocation(
        n_sites=n_sites, dimension=dimension, shared_rank=shared_rank,
        total_float_budget=total_float_budget, shared_float_count=shared_floats,
        private_float_budget=private_budget, private_rank_slots=slots,
        private_ranks=tuple(ranks), selected_residual_merit=sum(item[0] for item in selected),
    )


def fit_hierarchical_shared_private(
    solved_crosses: Sequence[torch.Tensor], merits: Sequence[torch.Tensor],
    shared_basis: torch.Tensor, *, total_float_budget: int,
) -> HierarchicalFit:
    dimension, _ = _validate_merits(merits)
    if len(solved_crosses) != len(merits):
        raise ValueError("solved crosses and merits must have equal site count")
    for value in solved_crosses:
        if not torch.is_tensor(value) or value.shape != (dimension, dimension) or (
            value.dtype != torch.float64 or value.device.type != "cpu" or
            not bool(torch.isfinite(value).all())
        ):
            raise ValueError("solved crosses must be finite CPU float64 square matrices")
    shared_rank = _validate_shared_basis(shared_basis, dimension)
    residual = residual_eigensystems(merits, shared_basis)
    allocation = allocate_private_ranks(
        residual.eigenvalues, dimension=dimension, shared_rank=shared_rank,
        total_float_budget=total_float_budget,
    )
    shared_inputs = tuple((value @ shared_basis).contiguous() for value in solved_crosses)
    private_bases: list[torch.Tensor] = []
    private_inputs: list[torch.Tensor] = []
    orthogonality = 0.0
    for site, rank in enumerate(allocation.private_ranks):
        basis = residual.eigenvectors[site][:, :rank].contiguous()
        private_bases.append(basis)
        private_inputs.append((solved_crosses[site] @ basis).contiguous())
        combined = torch.cat((shared_basis, basis), dim=1)
        identity = torch.eye(combined.shape[1], dtype=torch.float64)
        if combined.numel():
            orthogonality = max(orthogonality, float((combined.T @ combined - identity).abs().max()))
    explained_shared = sum(float(torch.trace(shared_basis.T @ merit @ shared_basis))
                           for merit in merits)
    price = hierarchical_price(len(merits), dimension, shared_rank, allocation.private_ranks)
    return HierarchicalFit(
        shared_basis=shared_basis.contiguous(), shared_input_maps=shared_inputs,
        private_bases=tuple(private_bases), private_input_maps=tuple(private_inputs),
        residual_eigenvalues=residual.eigenvalues, allocation=allocation, price=price,
        explained_shared_merit=explained_shared,
        explained_private_merit=allocation.selected_residual_merit,
        combined_orthogonality_max_abs=orthogonality,
    )


def coefficient_maps(fit: HierarchicalFit) -> tuple[torch.Tensor, ...]:
    """Materialize only for known-answer tests; production should deploy factors."""
    return tuple(
        shared @ fit.shared_basis.T + private @ basis.T
        for shared, private, basis in zip(
            fit.shared_input_maps, fit.private_input_maps, fit.private_bases, strict=True,
        )
    )


def materialize_float32_program(fit: HierarchicalFit) -> DeployedHierarchicalProgram:
    """Freeze the only licensed cast and factor ordering for future execution.

    Fit is float64.  Every factor is independently converted by ``.float()`` on
    CPU and made contiguous.  Runtime must compute the shared term first, the
    private term second, and add them once; it may not pre-materialize a dense map.
    """
    program = DeployedHierarchicalProgram(
        shared_basis=fit.shared_basis.float().contiguous(),
        shared_input_maps=tuple(value.float().contiguous() for value in fit.shared_input_maps),
        private_bases=tuple(value.float().contiguous() for value in fit.private_bases),
        private_input_maps=tuple(value.float().contiguous() for value in fit.private_input_maps),
        private_ranks=fit.allocation.private_ranks,
    )
    tensors = (
        program.shared_basis, *program.shared_input_maps,
        *program.private_bases, *program.private_input_maps,
    )
    if any(value.dtype != torch.float32 or value.device.type != "cpu" or not (
        value.is_contiguous() and bool(torch.isfinite(value).all())
    ) for value in tensors):
        raise RuntimeError("deployed hierarchical factors have invalid schema")
    return program


def deployed_coefficient_maps(
    program: DeployedHierarchicalProgram,
) -> tuple[torch.Tensor, ...]:
    """Apply the frozen float32 shared-then-private factor materialization order."""
    output = []
    for shared, private, basis in zip(
        program.shared_input_maps, program.private_input_maps,
        program.private_bases, strict=True,
    ):
        value = shared @ program.shared_basis.T
        if private.shape[1]:
            value = value + private @ basis.T
        output.append(value.contiguous())
    return tuple(output)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def factor_hash_receipt(fit: HierarchicalFit) -> dict[str, Any]:
    """Hash identified deployed maps/projectors, never gauge-sensitive columns."""
    deployed = materialize_float32_program(fit)
    shared_projector = deployed.shared_basis @ deployed.shared_basis.T
    site_projectors = tuple(
        shared_projector + basis @ basis.T for basis in deployed.private_bases
    )
    body = {
        "hash_currency": "float32_deployed_projectors_and_coefficient_maps",
        "shared_projector_sha256": _tensor_sha256(shared_projector),
        "site_projector_sha256s": [_tensor_sha256(value) for value in site_projectors],
        "coefficient_map_sha256s": [
            _tensor_sha256(value) for value in deployed_coefficient_maps(deployed)
        ],
        "private_ranks": list(fit.allocation.private_ranks),
        "price": asdict(fit.price),
        "raw_factor_hashes_reported": False,
        "serialized_program_authority": False,
    }
    return {**body, "sha256": hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()}


def canonical_bilin18_storage_points() -> dict[str, Any]:
    dimension, n_sites, rank = 1152, 36, 512
    budgets = {
        "global_q512": dimension * (n_sites + 1) * rank,
        "typed_q512": dimension * (n_sites + 2) * rank,
        "independent_q512": 2 * dimension * n_sites * rank,
    }
    shared_ranks = (0, 64, 128, 256)
    cells = {
        budget: {
            str(q0): allocate_private_ranks(
                tuple(torch.zeros(dimension - q0, dtype=torch.float64) for _ in range(n_sites)),
                dimension=dimension, shared_rank=q0, total_float_budget=value,
            ).private_rank_slots
            for q0 in shared_ranks
        } for budget, value in budgets.items()
    }
    cells["global_q512"]["512"] = 0
    common_table = n_sites * 5419 * dimension
    return {
        "dimension": dimension,
        "n_sites": n_sites,
        "budgets": budgets,
        "private_rank_slots": cells,
        "common_table_float_count": common_table,
        "full_program_float_counts": {
            name: common_table + budget for name, budget in budgets.items()
        },
    }


if __name__ == "__main__":
    print(json.dumps(canonical_bilin18_storage_points(), indent=2, sort_keys=True))
