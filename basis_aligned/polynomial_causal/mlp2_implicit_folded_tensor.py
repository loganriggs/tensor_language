"""Implicit folded-tensor spectra and executable prices for bilin18 MLP2.

The quadratic map is represented by factors rather than a materialized third-order
tensor::

    T[o, i, j] = sum_n Down[o, n] * sym(Left[n] outer Right[n])[i, j].

Only the two small mode Gram matrices are returned.  The implementation balances each
product term over its exact scale gauge before any spectrum is computed.  Consequently
the authoritative spectra are invariant to per-gate rescaling, sign redistribution,
Left/Right exchange, and gate permutation.

This module is deliberately data- and checkpoint-agnostic.  It does not score model
outcomes or construct an executable replacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable

import numpy as np


BILIN18_INPUT_DIM = 1152
BILIN18_OUTPUT_DIM = 1152
BILIN18_MLP_PRODUCTS = 4608
BILIN18_MLP_SITE = 2
DEFAULT_ENERGY_LEVELS = (0.90, 0.95, 0.99, 0.999)


@dataclass(frozen=True)
class BalancedFactors:
    """A minimum-sum-of-squared-norm representative of each product term."""

    output: np.ndarray
    left: np.ndarray
    right: np.ndarray
    active_mask: np.ndarray
    common_norms: np.ndarray


@dataclass(frozen=True)
class Spectrum:
    """Singular spectrum represented through squared singular values."""

    singular_values: np.ndarray
    eigenvalues: np.ndarray
    total_energy: float
    numerical_rank: int
    energy_ranks: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        return {
            "singular_values": self.singular_values.tolist(),
            "eigenvalues": self.eigenvalues.tolist(),
            "total_energy": self.total_energy,
            "numerical_rank": self.numerical_rank,
            "energy_ranks": dict(self.energy_ranks),
        }


@dataclass(frozen=True)
class FoldedModeGrams:
    """Exact mode Grams of the partially symmetric folded tensor."""

    output: np.ndarray
    input: np.ndarray
    input_modes_shared: bool = True


@dataclass(frozen=True)
class ExecutablePrice:
    """Fixed-grammar floating-value and multiply price for one MLP site."""

    family: str
    stored_values: int
    bias_values: int
    bilinear_products_per_token: int
    linear_weight_multiplies_per_token: int
    metadata_included: bool = False
    status: str = "upper_bound_for_declared_grammar"

    def as_dict(self) -> dict[str, int | str | bool]:
        return asdict(self)


@dataclass(frozen=True)
class HOSVDPricePoint:
    energy_level: float
    output_rank: int
    input_rank: int
    relative_frobenius_error_upper_bound: float
    price: ExecutablePrice
    fewer_products_than_native: bool
    fewer_values_than_native: bool


@dataclass(frozen=True)
class FoldedTensorDiagnostic:
    """Outcome-free weight diagnostic for one vector-valued quadratic map."""

    site: int
    input_dim: int
    output_dim: int
    declared_products: int
    active_products: int
    zero_products: int
    bias_preserved: bool
    balanced_down: Spectrum
    folded_output: Spectrum
    folded_input: Spectrum
    native_price: ExecutablePrice
    down_price_points: tuple[tuple[float, ExecutablePrice], ...]
    hosvd_price_points: tuple[HOSVDPricePoint, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "site": self.site,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "declared_products": self.declared_products,
            "active_products": self.active_products,
            "zero_products": self.zero_products,
            "bias_preserved": self.bias_preserved,
            "balanced_down": self.balanced_down.as_dict(),
            "folded_output": self.folded_output.as_dict(),
            "folded_input": self.folded_input.as_dict(),
            "native_price": self.native_price.as_dict(),
            "down_price_points": [
                {"energy_level": level, "price": price.as_dict()}
                for level, price in self.down_price_points
            ],
            "hosvd_price_points": [
                {
                    "energy_level": point.energy_level,
                    "output_rank": point.output_rank,
                    "input_rank": point.input_rank,
                    "relative_frobenius_error_upper_bound": (
                        point.relative_frobenius_error_upper_bound
                    ),
                    "price": point.price.as_dict(),
                    "fewer_products_than_native": point.fewer_products_than_native,
                    "fewer_values_than_native": point.fewer_values_than_native,
                }
                for point in self.hosvd_price_points
            ],
            "claim_boundary": (
                "spectral_and_fixed_grammar_price_diagnostic_only; no CP-rank, CE, "
                "causal, semantic, removal, OOD, or cube credit"
            ),
        }


def _float64_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return np.array(array, dtype=np.float64, copy=True, order="C")


def _validate_factors(
    output: np.ndarray, left: np.ndarray, right: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = _float64_matrix("output", output)
    left = _float64_matrix("left", left)
    right = _float64_matrix("right", right)
    if left.shape != right.shape:
        raise ValueError("left and right must have identical (products, input) shapes")
    if output.shape[1] != left.shape[0]:
        raise ValueError("output columns must equal the number of product gates")
    if output.shape[0] == 0 or left.shape[0] == 0 or left.shape[1] == 0:
        raise ValueError("factor dimensions must be nonzero")
    return output, left, right


def _stable_row_norms(matrix: np.ndarray) -> np.ndarray:
    scale = np.max(np.abs(matrix), axis=1)
    norms = np.zeros_like(scale)
    active = scale > 0.0
    normalized = matrix[active] / scale[active, None]
    norms[active] = scale[active] * np.sqrt(np.sum(normalized * normalized, axis=1))
    return norms


def balance_product_factors(
    output: np.ndarray, left: np.ndarray, right: np.ndarray
) -> BalancedFactors:
    """Balance every term without changing its symmetric quadratic tensor.

    For a nonzero term, all three factor norms are set to their geometric mean.
    Any term with a zero factor contributes exactly zero and is canonicalized to three
    zero factors.  Positive balancing scales preserve the original sign allocation.
    """

    output, left, right = _validate_factors(output, left, right)
    output_rows = output.T
    output_norm = _stable_row_norms(output_rows)
    left_norm = _stable_row_norms(left)
    right_norm = _stable_row_norms(right)
    active = (output_norm > 0.0) & (left_norm > 0.0) & (right_norm > 0.0)

    common = np.zeros_like(output_norm)
    common[active] = np.exp(
        (np.log(output_norm[active]) + np.log(left_norm[active]) + np.log(right_norm[active]))
        / 3.0
    )

    balanced_output_rows = np.zeros_like(output_rows)
    balanced_left = np.zeros_like(left)
    balanced_right = np.zeros_like(right)
    balanced_output_rows[active] = output_rows[active] * (
        common[active] / output_norm[active]
    )[:, None]
    balanced_left[active] = left[active] * (common[active] / left_norm[active])[:, None]
    balanced_right[active] = right[active] * (common[active] / right_norm[active])[:, None]

    if not (
        np.all(np.isfinite(balanced_output_rows))
        and np.all(np.isfinite(balanced_left))
        and np.all(np.isfinite(balanced_right))
    ):
        raise ValueError("factor balancing overflowed")
    return BalancedFactors(
        output=balanced_output_rows.T,
        left=balanced_left,
        right=balanced_right,
        active_mask=active,
        common_norms=common,
    )


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _weighted_sandwich(
    row_left: np.ndarray,
    coefficient_gram: np.ndarray,
    inner_gram: np.ndarray,
    row_right: np.ndarray,
) -> np.ndarray:
    weighted = np.multiply(coefficient_gram, inner_gram)
    return (row_left.T @ weighted) @ row_right


def implicit_folded_mode_grams(
    output: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    already_balanced: bool = False,
) -> FoldedModeGrams:
    """Compute exact folded mode Grams without materializing ``T[o,i,j]``.

    Workspace is quadratic in the number of product gates, rather than cubic in the
    residual width.  Input-mode symmetry is imposed analytically, yielding one shared
    input Gram for both tensor input modes.
    """

    if already_balanced:
        output, left, right = _validate_factors(output, left, right)
    else:
        balanced = balance_product_factors(output, left, right)
        output, left, right = balanced.output, balanced.left, balanced.right

    ll = left @ left.T
    rr = right @ right.T
    product_form_gram = np.multiply(ll, rr)
    lr = left @ right.T
    product_form_gram += np.multiply(lr, lr.T)
    product_form_gram *= 0.5
    output_mode = _symmetric((output @ product_form_gram) @ output.T)

    output_coeff_gram = output.T @ output
    term_ll = _weighted_sandwich(left, output_coeff_gram, rr, left)
    term_rr = _weighted_sandwich(right, output_coeff_gram, ll, right)
    # Coefficient (n,m) is <r_n,l_m>; the transpose supplies the other cross term.
    rl = right @ left.T
    cross = _weighted_sandwich(left, output_coeff_gram, rl, right)
    input_mode = _symmetric(0.25 * (term_ll + term_rr + cross + cross.T))

    if not np.all(np.isfinite(output_mode)) or not np.all(np.isfinite(input_mode)):
        raise ValueError("implicit mode Gram accumulation produced non-finite values")
    return FoldedModeGrams(output=output_mode, input=input_mode)


def _energy_key(level: float) -> str:
    return f"r{100.0 * level:g}"


def spectrum_from_psd_gram(
    gram: np.ndarray,
    *,
    energy_levels: Iterable[float] = DEFAULT_ENERGY_LEVELS,
    psd_rtol: float = 1e-10,
    psd_atol: float = 1e-12,
    rank_rtol: float | None = None,
) -> Spectrum:
    """Return a fail-closed spectrum from a nominally PSD Gram matrix."""

    gram = _float64_matrix("gram", gram)
    if gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")
    symmetry_scale = max(float(np.max(np.abs(gram), initial=0.0)), 1.0)
    symmetry_error = float(np.max(np.abs(gram - gram.T), initial=0.0))
    if symmetry_error > psd_atol + psd_rtol * symmetry_scale:
        raise ValueError("gram is not symmetric within tolerance")
    eigenvalues = np.linalg.eigvalsh(_symmetric(gram))
    max_eigenvalue = max(float(eigenvalues[-1]), 0.0)
    tolerance = psd_atol + psd_rtol * max(max_eigenvalue, 1.0)
    if float(eigenvalues[0]) < -tolerance:
        raise ValueError("gram is not positive semidefinite within tolerance")
    eigenvalues = np.maximum(eigenvalues, 0.0)[::-1]
    singular_values = np.sqrt(eigenvalues)
    total = float(np.sum(eigenvalues))

    levels = tuple(float(level) for level in energy_levels)
    if any(level <= 0.0 or level > 1.0 for level in levels):
        raise ValueError("energy levels must lie in (0, 1]")
    if len(set(levels)) != len(levels):
        raise ValueError("energy levels must be unique")
    if total == 0.0:
        energy_ranks = {_energy_key(level): 0 for level in levels}
    else:
        cumulative = np.cumsum(eigenvalues) / total
        energy_ranks = {
            _energy_key(level): int(np.searchsorted(cumulative, level, side="left") + 1)
            for level in levels
        }

    if max_eigenvalue == 0.0:
        numerical_rank = 0
    else:
        if rank_rtol is None:
            rank_rtol = gram.shape[0] * np.finfo(np.float64).eps
        numerical_rank = int(np.count_nonzero(eigenvalues > max_eigenvalue * rank_rtol))
    return Spectrum(
        singular_values=singular_values,
        eigenvalues=eigenvalues,
        total_energy=total,
        numerical_rank=numerical_rank,
        energy_ranks=energy_ranks,
    )


def spectrum_from_matrix(
    matrix: np.ndarray,
    *,
    energy_levels: Iterable[float] = DEFAULT_ENERGY_LEVELS,
) -> Spectrum:
    """Return a matrix SVD spectrum in the same schema as a mode Gram spectrum."""

    matrix = _float64_matrix("matrix", matrix)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    eigenvalues = singular_values * singular_values
    total = float(np.sum(eigenvalues))
    levels = tuple(float(level) for level in energy_levels)
    if any(level <= 0.0 or level > 1.0 for level in levels):
        raise ValueError("energy levels must lie in (0, 1]")
    if len(set(levels)) != len(levels):
        raise ValueError("energy levels must be unique")
    if total == 0.0:
        energy_ranks = {_energy_key(level): 0 for level in levels}
        numerical_rank = 0
    else:
        cumulative = np.cumsum(eigenvalues) / total
        energy_ranks = {
            _energy_key(level): int(np.searchsorted(cumulative, level, side="left") + 1)
            for level in levels
        }
        tolerance = max(matrix.shape) * np.finfo(np.float64).eps * singular_values[0]
        numerical_rank = int(np.count_nonzero(singular_values > tolerance))
    return Spectrum(singular_values, eigenvalues, total, numerical_rank, energy_ranks)


def native_mlp_price(*, products: int, input_dim: int, output_dim: int) -> ExecutablePrice:
    linear = products * (2 * input_dim + output_dim)
    return ExecutablePrice(
        family="native_product_factors",
        stored_values=linear + output_dim,
        bias_values=output_dim,
        bilinear_products_per_token=products,
        linear_weight_multiplies_per_token=linear,
        status="exact_native_program",
    )


def down_svd_price(
    *, products: int, input_dim: int, output_dim: int, rank: int
) -> ExecutablePrice:
    if rank < 0 or rank > min(products, output_dim):
        raise ValueError("invalid Down SVD rank")
    linear = 2 * products * input_dim + rank * (products + output_dim)
    return ExecutablePrice(
        family="balanced_down_svd_with_native_products",
        stored_values=linear + output_dim,
        bias_values=output_dim,
        bilinear_products_per_token=products,
        linear_weight_multiplies_per_token=linear,
    )


def symmetric_tucker_price(
    *, input_dim: int, output_dim: int, input_rank: int, output_rank: int
) -> ExecutablePrice:
    if input_rank < 0 or input_rank > input_dim:
        raise ValueError("invalid Tucker input rank")
    if output_rank < 0 or output_rank > output_dim:
        raise ValueError("invalid Tucker output rank")
    monomials = input_rank * (input_rank + 1) // 2
    linear = input_dim * input_rank + output_rank * monomials + output_dim * output_rank
    return ExecutablePrice(
        family="symmetric_tucker_dense_triangular_core",
        stored_values=linear + output_dim,
        bias_values=output_dim,
        bilinear_products_per_token=monomials,
        linear_weight_multiplies_per_token=linear,
    )


def hosvd_relative_error_upper_bound(
    output_spectrum: Spectrum,
    input_spectrum: Spectrum,
    *,
    output_rank: int,
    input_rank: int,
) -> float:
    """Bound relative Frobenius error for ranks ``(r_o, r_i, r_i)``.

    The standard truncated-HOSVD squared-error bound is the sum of discarded mode
    energies.  Both input modes use the same folded spectrum.  Orthogonal projection
    cannot be worse than the zero tensor, so the returned relative bound is capped at
    one.
    """

    if output_rank < 0 or output_rank > output_spectrum.eigenvalues.size:
        raise ValueError("invalid output rank")
    if input_rank < 0 or input_rank > input_spectrum.eigenvalues.size:
        raise ValueError("invalid input rank")
    if output_spectrum.total_energy == 0.0 and input_spectrum.total_energy == 0.0:
        return 0.0
    if output_spectrum.total_energy <= 0.0 or input_spectrum.total_energy <= 0.0:
        raise ValueError("mode spectra disagree about zero tensor energy")
    output_tail = float(np.sum(output_spectrum.eigenvalues[output_rank:])) / (
        output_spectrum.total_energy
    )
    input_tail = float(np.sum(input_spectrum.eigenvalues[input_rank:])) / (
        input_spectrum.total_energy
    )
    return sqrt(min(1.0, max(0.0, output_tail + 2.0 * input_tail)))


def analyze_folded_factors(
    output: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bias: np.ndarray,
    *,
    site: int = BILIN18_MLP_SITE,
    energy_levels: Iterable[float] = DEFAULT_ENERGY_LEVELS,
) -> FoldedTensorDiagnostic:
    """Analyze factors without assuming bilin18 dimensions (used by known-answer tests)."""

    output, left, right = _validate_factors(output, left, right)
    bias = np.asarray(bias, dtype=np.float64)
    if bias.shape != (output.shape[0],) or not np.all(np.isfinite(bias)):
        raise ValueError("bias must be a finite vector with one value per output")
    levels = tuple(float(level) for level in energy_levels)
    balanced = balance_product_factors(output, left, right)
    active = int(np.count_nonzero(balanced.active_mask))
    if active == 0:
        down_matrix = np.zeros((output.shape[0], 0), dtype=np.float64)
    else:
        down_matrix = balanced.output[:, balanced.active_mask]
    down_spectrum = spectrum_from_matrix(down_matrix, energy_levels=levels)
    grams = implicit_folded_mode_grams(
        balanced.output, balanced.left, balanced.right, already_balanced=True
    )
    output_spectrum = spectrum_from_psd_gram(grams.output, energy_levels=levels)
    input_spectrum = spectrum_from_psd_gram(grams.input, energy_levels=levels)

    input_dim = left.shape[1]
    output_dim = output.shape[0]
    declared_products = left.shape[0]
    native = native_mlp_price(
        products=declared_products, input_dim=input_dim, output_dim=output_dim
    )
    # Zero-contribution gates may be dropped exactly from diagnostic candidates.
    down_points: list[tuple[float, ExecutablePrice]] = []
    hosvd_points: list[HOSVDPricePoint] = []
    for level in levels:
        key = _energy_key(level)
        down_rank = down_spectrum.energy_ranks[key]
        down_points.append(
            (
                level,
                down_svd_price(
                    products=active,
                    input_dim=input_dim,
                    output_dim=output_dim,
                    rank=down_rank,
                ),
            )
        )
        output_rank = output_spectrum.energy_ranks[key]
        input_rank = input_spectrum.energy_ranks[key]
        price = symmetric_tucker_price(
            input_dim=input_dim,
            output_dim=output_dim,
            input_rank=input_rank,
            output_rank=output_rank,
        )
        hosvd_points.append(
            HOSVDPricePoint(
                energy_level=level,
                output_rank=output_rank,
                input_rank=input_rank,
                relative_frobenius_error_upper_bound=hosvd_relative_error_upper_bound(
                    output_spectrum,
                    input_spectrum,
                    output_rank=output_rank,
                    input_rank=input_rank,
                ),
                price=price,
                fewer_products_than_native=(
                    price.bilinear_products_per_token < declared_products
                ),
                fewer_values_than_native=(price.stored_values < native.stored_values),
            )
        )
    return FoldedTensorDiagnostic(
        site=int(site),
        input_dim=input_dim,
        output_dim=output_dim,
        declared_products=declared_products,
        active_products=active,
        zero_products=declared_products - active,
        bias_preserved=True,
        balanced_down=down_spectrum,
        folded_output=output_spectrum,
        folded_input=input_spectrum,
        native_price=native,
        down_price_points=tuple(down_points),
        hosvd_price_points=tuple(hosvd_points),
    )


def analyze_bilin18_mlp2_factors(
    output: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    bias: np.ndarray,
    *,
    energy_levels: Iterable[float] = DEFAULT_ENERGY_LEVELS,
) -> FoldedTensorDiagnostic:
    """Strict shape-checked entry point for a future CPU-only bilin18 MLP2 runner."""

    output_array = np.asarray(output)
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    expected_output = (BILIN18_OUTPUT_DIM, BILIN18_MLP_PRODUCTS)
    expected_inputs = (BILIN18_MLP_PRODUCTS, BILIN18_INPUT_DIM)
    if output_array.shape != expected_output:
        raise ValueError(f"MLP2 Down must have shape {expected_output}")
    if left_array.shape != expected_inputs or right_array.shape != expected_inputs:
        raise ValueError(f"MLP2 Left and Right must have shape {expected_inputs}")
    return analyze_folded_factors(
        output_array,
        left_array,
        right_array,
        bias,
        site=BILIN18_MLP_SITE,
        energy_levels=energy_levels,
    )


def dense_symmetric_core_product_threshold(native_products: int) -> int:
    """Largest input rank whose triangular monomials are fewer than native products."""

    if native_products <= 0:
        raise ValueError("native_products must be positive")
    # Solve r(r+1)/2 < native_products, then correct for integer roundoff.
    rank = max(0, int((sqrt(1 + 8 * native_products) - 1) // 2))
    while rank * (rank + 1) // 2 >= native_products:
        rank -= 1
    while (rank + 1) * (rank + 2) // 2 < native_products:
        rank += 1
    return rank
