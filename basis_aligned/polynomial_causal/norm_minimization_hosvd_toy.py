"""Small exact diagnostics for "minimum norm, then HOSVD".

The two gauges that are easy to conflate are tested separately:

1. A bilinear MLP has one reciprocal *scalar* gauge per product term.  Balancing
   that gauge minimizes the displayed factors' Frobenius norm but leaves the
   folded third-order tensor exactly fixed.  Consequently it cannot alter that
   tensor's HOSVD spectrum.
2. A two-node tensor network has a full ``GL(r)`` gauge on its contracted edge.
   Replacing the factors by square-root SVD factors gives the minimum-norm
   representative (or an orbit-closure representative when the bond contains
   dormant rank).  It fixes conditioning and deletes dormant bond directions,
   while again leaving the physical contraction and its singular spectrum fixed.

This file is CPU-only and has no model/checkpoint dependency.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from mlp1_implicit_folded_tensor_v1 import balance_factors


@dataclass(frozen=True)
class CpGaugeToy:
    parameter_norm_before: float
    parameter_norm_after: float
    folded_tensor_relative_drift: float
    hosvd_spectrum_relative_drift: float
    balanced_log_defect: float


@dataclass(frozen=True)
class EdgeGaugeToy:
    parameter_norm_before: float
    parameter_norm_after: float
    exact_minimum_norm: float
    contraction_relative_drift: float
    contraction_spectrum_relative_drift: float
    balanced_gram_relative_defect: float
    bond_rank_before: int
    bond_rank_after: int


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = torch.linalg.vector_norm(expected)
    numerator = torch.linalg.vector_norm(actual - expected)
    if float(denominator) == 0.0:
        return float(numerator)
    return float(numerator / denominator)


def _mode_spectra(tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Return singular values of every mode unfolding of a third-order tensor."""
    if tensor.ndim != 3:
        raise ValueError("tensor must be third order")
    spectra = []
    for mode in range(3):
        unfolding = tensor.movedim(mode, 0).reshape(tensor.shape[mode], -1)
        spectra.append(torch.linalg.svdvals(unfolding))
    return tuple(spectra)


def _fold_cp(down: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("oh,hi,hj->oij", down, left, right)


def cp_scalar_gauge_toy(seed: int = 1827) -> CpGaugeToy:
    """Show that CP balancing helps conditioning, not folded-tensor HOSVD rank."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    output, hidden, width = 5, 9, 6
    down = torch.randn(output, hidden, generator=generator, dtype=torch.float64)
    left = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    right = torch.randn(hidden, width, generator=generator, dtype=torch.float64)
    bias = torch.randn(output, generator=generator, dtype=torch.float64)

    # Deliberately make the displayed decomposition very ill-conditioned without
    # changing any rank-one term.
    log_alpha = 5.0 * torch.randn(hidden, generator=generator, dtype=torch.float64)
    log_beta = 5.0 * torch.randn(hidden, generator=generator, dtype=torch.float64)
    alpha, beta = log_alpha.exp(), log_beta.exp()
    gauged_down = down / (alpha * beta).unsqueeze(0)
    gauged_left = left * alpha.unsqueeze(1)
    gauged_right = right * beta.unsqueeze(1)

    folded_before = _fold_cp(gauged_down, gauged_left, gauged_right)
    spectra_before = _mode_spectra(folded_before)
    balanced = balance_factors(gauged_down, gauged_left, gauged_right, bias)
    folded_after = _fold_cp(balanced.down, balanced.left, balanced.right)
    spectra_after = _mode_spectra(folded_after)

    spectrum_drift = max(
        _relative_error(after, before)
        for before, after in zip(spectra_before, spectra_after)
    )
    norm_before = float(
        gauged_down.square().sum()
        + gauged_left.square().sum()
        + gauged_right.square().sum()
    )
    norm_after = float(
        balanced.down.square().sum()
        + balanced.left.square().sum()
        + balanced.right.square().sum()
    )
    return CpGaugeToy(
        parameter_norm_before=norm_before,
        parameter_norm_after=norm_after,
        folded_tensor_relative_drift=_relative_error(folded_after, folded_before),
        hosvd_spectrum_relative_drift=spectrum_drift,
        balanced_log_defect=balanced.max_log_defect_after,
    )


def minimum_norm_edge_factors(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    relative_tolerance: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return square-root SVD factors of ``left @ right``.

    If ``P = U diag(s) V^T``, the returned factors are
    ``U diag(sqrt(s))`` and ``diag(sqrt(s)) V^T``.  They attain

        min_{AB=P} ||A||_F^2 + ||B||_F^2 = 2 ||P||_*,

    with the minimum taken over factorizations having sufficient bond width.
    Zero singular directions are removed, so a rank-deficient input bond is mapped
    to its orbit closure rather than reached by an invertible gauge.
    """
    if left.ndim != 2 or right.ndim != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("left and right must be compatible matrices")
    if left.device.type != "cpu" or right.device.type != "cpu":
        raise ValueError("toy accepts CPU tensors only")
    product = left.to(torch.float64) @ right.to(torch.float64)
    u, singular, vh = torch.linalg.svd(product, full_matrices=False)
    cutoff = 0.0 if singular.numel() == 0 else relative_tolerance * float(singular.max())
    live = singular > cutoff
    root = singular[live].sqrt()
    balanced_left = u[:, live] * root.unsqueeze(0)
    balanced_right = root.unsqueeze(1) * vh[live, :]
    return balanced_left, balanced_right, singular[live]


def gl_edge_gauge_toy(seed: int = 1828, *, dormant: bool = False) -> EdgeGaugeToy:
    """Show when full-edge balancing helps and what it provably cannot change."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows, columns = 8, 7
    physical_rank = 3 if dormant else 4
    u, _ = torch.linalg.qr(
        torch.randn(rows, physical_rank, generator=generator, dtype=torch.float64),
        mode="reduced",
    )
    v, _ = torch.linalg.qr(
        torch.randn(columns, physical_rank, generator=generator, dtype=torch.float64),
        mode="reduced",
    )
    singular = torch.tensor([9.0, 3.0, 0.7, 0.15], dtype=torch.float64)[:physical_rank]
    canonical_left = u * singular.sqrt().unsqueeze(0)
    canonical_right = singular.sqrt().unsqueeze(1) * v.T

    if dormant:
        # Two nonzero producer columns are killed by zero consumer rows.  They cost
        # parameters and destroy factor-level canonicality but do no physical work.
        junk = torch.randn(rows, 2, generator=generator, dtype=torch.float64)
        left = torch.cat((canonical_left, junk), dim=1)
        right = torch.cat(
            (canonical_right, torch.zeros(2, columns, dtype=torch.float64)), dim=0
        )
    else:
        # An arbitrary invertible gauge creates six orders of conditioning while
        # preserving the contraction exactly.
        q1, _ = torch.linalg.qr(
            torch.randn(physical_rank, physical_rank, generator=generator, dtype=torch.float64)
        )
        q2, _ = torch.linalg.qr(
            torch.randn(physical_rank, physical_rank, generator=generator, dtype=torch.float64)
        )
        scales = torch.logspace(-3, 3, physical_rank, dtype=torch.float64)
        gauge = q1 @ torch.diag(scales) @ q2.T
        left = canonical_left @ gauge
        right = torch.linalg.solve(gauge, canonical_right)

    product_before = left @ right
    spectrum_before = torch.linalg.svdvals(product_before)
    balanced_left, balanced_right, live_singular = minimum_norm_edge_factors(left, right)
    product_after = balanced_left @ balanced_right
    spectrum_after = torch.linalg.svdvals(product_after)
    producer_gram = balanced_left.T @ balanced_left
    consumer_gram = balanced_right @ balanced_right.T

    norm_before = float(left.square().sum() + right.square().sum())
    norm_after = float(balanced_left.square().sum() + balanced_right.square().sum())
    exact_minimum = float(2.0 * live_singular.sum())
    return EdgeGaugeToy(
        parameter_norm_before=norm_before,
        parameter_norm_after=norm_after,
        exact_minimum_norm=exact_minimum,
        contraction_relative_drift=_relative_error(product_after, product_before),
        contraction_spectrum_relative_drift=_relative_error(spectrum_after, spectrum_before),
        balanced_gram_relative_defect=_relative_error(producer_gram, consumer_gram),
        bond_rank_before=left.shape[1],
        bond_rank_after=balanced_left.shape[1],
    )


def run_toys() -> dict[str, dict[str, float | int]]:
    return {
        "cp_scalar_gauge": asdict(cp_scalar_gauge_toy()),
        "gl_edge_ill_conditioned": asdict(gl_edge_gauge_toy()),
        "gl_edge_dormant_rank": asdict(gl_edge_gauge_toy(dormant=True)),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_toys(), indent=2, sort_keys=True))
