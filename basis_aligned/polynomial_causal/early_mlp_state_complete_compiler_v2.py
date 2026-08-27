"""Pure algebra and pricing contract for state-complete compiler v2.

This module intentionally contains no model or dataset loading.  It fixes the
typed interface that the authoritative runner must use before label capture.
Rows are token-major: ``z`` and ``mo`` end in d_model, while coefficient arrays
end in the admitted basis dimension.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


D_MODEL = 1152
COEFFICIENT_DIM = 64
NATIVE_PRODUCTS = 4608
NATIVE_K_GRID = (8, 16, 32, 64, 128, 256)
AFFINE_RANK_GRID = (8, 16, 32, 64)
CAUSAL_ISOTROPIC_FLOOR = 0.05


def _finite_matrix(name: str, value: torch.Tensor) -> None:
    if value.ndim != 2 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be a finite matrix")


def canonicalize_native_terms(
    left: torch.Tensor,
    right: torch.Tensor,
    projected_decoder: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove reciprocal scale/sign gauge without changing the tensor program.

    Each L/R row is unit norm and the removed positive norm product is absorbed
    into Q.  Simultaneously flipping L and R leaves their product unchanged; we
    fix that residual sign by making the largest-absolute L entry nonnegative.
    """

    left = left.double().clone()
    right = right.double().clone()
    projected_decoder = projected_decoder.double().clone()
    for name, value in (("left", left), ("right", right),
                        ("projected_decoder", projected_decoder)):
        _finite_matrix(name, value)
    if left.shape != right.shape or projected_decoder.shape[0] != left.shape[0]:
        raise ValueError("native term dimensions do not align")
    left_norm = left.norm(dim=1)
    right_norm = right.norm(dim=1)
    if bool((left_norm <= 0.0).any()) or bool((right_norm <= 0.0).any()):
        raise ValueError("native L/R rows must be nonzero")
    projected_decoder *= (left_norm * right_norm)[:, None]
    left /= left_norm[:, None]
    right /= right_norm[:, None]
    for row in range(left.shape[0]):
        pivot = int(left[row].abs().argmax())
        if float(left[row, pivot]) < 0.0:
            left[row].neg_()
            right[row].neg_()
    return left, right, projected_decoder


def project_native_weights(
    left_weight: torch.Tensor,
    right_weight: torch.Tensor,
    down_weight: torch.Tensor,
    down_bias: torch.Tensor,
    basis: torch.Tensor,
    *,
    indices: Sequence[int] | torch.Tensor | None = None,
    canonicalize: bool = True,
) -> dict[str, torch.Tensor]:
    """Serialize selected native bilinear terms into an independent program.

    PyTorch orientations are L,R=[products,d_model], D=[d_model,products],
    bias=[d_model], and B=[d_model,coefficients].  No checkpoint pointer is
    returned.  ``indices=None`` selects the full adequacy ceiling.
    """

    values = {
        "left_weight": left_weight.double(),
        "right_weight": right_weight.double(),
        "down_weight": down_weight.double(),
        "basis": basis.double(),
    }
    for name, value in values.items():
        _finite_matrix(name, value)
    left = values["left_weight"]
    right = values["right_weight"]
    down = values["down_weight"]
    basis64 = values["basis"]
    bias = down_bias.double()
    if bias.ndim != 1 or not torch.isfinite(bias).all():
        raise ValueError("down_bias must be a finite vector")
    if left.shape != right.shape:
        raise ValueError("left/right native weights must have identical shape")
    products, d_model = left.shape
    if down.shape != (d_model, products) or bias.shape != (d_model,):
        raise ValueError("down projection dimensions do not align")
    if basis64.shape[0] != d_model:
        raise ValueError("basis d_model dimension does not align")

    if indices is None:
        selected = torch.arange(products, dtype=torch.long)
    else:
        selected = torch.as_tensor(indices, dtype=torch.long).flatten()
        if selected.numel() == 0 or bool((selected < 0).any()) or bool(
            (selected >= products).any()
        ):
            raise ValueError("native indices are empty or outside the product range")
        if selected.unique().numel() != selected.numel():
            raise ValueError("native indices must be unique")
    selected_left = left.index_select(0, selected).clone()
    selected_right = right.index_select(0, selected).clone()
    projected_decoder = (down.T @ basis64).index_select(0, selected).clone()
    if canonicalize:
        selected_left, selected_right, projected_decoder = canonicalize_native_terms(
            selected_left, selected_right, projected_decoder
        )
    return {
        "left": selected_left,
        "right": selected_right,
        "projected_decoder": projected_decoder,
        "beta": bias @ basis64,
        "indices": selected.clone(),
    }


def native_projected_output(z: torch.Tensor, state: Mapping[str, Any]) -> torch.Tensor:
    """Evaluate p_hat(z)=sum_i q_i(l_i z)(r_i z)+beta."""

    required = ("left", "right", "projected_decoder", "beta")
    if any(key not in state for key in required):
        raise ValueError("native state is incomplete")
    z64 = z.double()
    left = state["left"].double()
    right = state["right"].double()
    decoder = state["projected_decoder"].double()
    beta = state["beta"].double()
    _finite_matrix("z", z64)
    if left.ndim != 2 or right.shape != left.shape:
        raise ValueError("native L/R state dimensions do not align")
    if z64.shape[1] != left.shape[1] or decoder.shape[0] != left.shape[0]:
        raise ValueError("native program input/product dimensions do not align")
    if decoder.ndim != 2 or beta.shape != (decoder.shape[1],):
        raise ValueError("native decoder/bias dimensions do not align")
    products = (z64 @ left.T) * (z64 @ right.T)
    return products @ decoder + beta


def state_complete_coefficients(
    z: torch.Tensor,
    mo: torch.Tensor,
    basis: torch.Tensor,
    state: Mapping[str, Any],
) -> torch.Tensor:
    """Return legal live correction c_hat(z,mo)=p_hat(z)-mo B."""

    z64 = z.double()
    mo64 = mo.double()
    basis64 = basis.double()
    if z64.ndim != 2 or mo64.ndim != 2 or z64.shape[0] != mo64.shape[0]:
        raise ValueError("z and mo must be row-aligned matrices")
    if basis64.ndim != 2 or mo64.shape[1] != basis64.shape[0]:
        raise ValueError("mo and basis dimensions do not align")
    projected = native_projected_output(z64, state)
    if projected.shape[1] != basis64.shape[1]:
        raise ValueError("program and basis coefficient dimensions do not align")
    return projected - mo64 @ basis64


def physical_correction(coefficients: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    coefficients = coefficients.double()
    basis = basis.double()
    if coefficients.ndim != 2 or basis.ndim != 2:
        raise ValueError("coefficients and basis must be matrices")
    if coefficients.shape[1] != basis.shape[1]:
        raise ValueError("coefficient and basis dimensions do not align")
    return coefficients @ basis.T


def native_program_price(k: int, *, include_basis: bool) -> dict[str, Any]:
    """Registered standalone/amortized storage and runtime price for one site."""

    if k not in (*NATIVE_K_GRID, NATIVE_PRODUCTS):
        raise ValueError("K is outside the registered native ladder")
    basis_reals = D_MODEL * COEFFICIENT_DIM if include_basis else 0
    program_reals = COEFFICIENT_DIM + k * (2 * D_MODEL + COEFFICIENT_DIM)
    total = basis_reals + program_reals
    multiplies = 2 * D_MODEL * COEFFICIENT_DIM + k * (
        2 * D_MODEL + COEFFICIENT_DIM
    )
    original = 3 * NATIVE_PRODUCTS * D_MODEL + D_MODEL
    return {
        "k": int(k),
        "include_basis": bool(include_basis),
        "basis_reals": int(basis_reals),
        "program_reals": int(program_reals),
        "total_reals": int(total),
        "float32_bits": int(32 * total),
        "inference_multiplies_per_token": int(multiplies),
        "hadamard_products_per_token": int(k),
        "original_mlp_reals": int(original),
        "fraction_of_original_reals": total / original,
    }


def corrected_affine_price(rank: int, *, include_basis: bool) -> dict[str, Any]:
    """Price affine p_hat(z)-mo B including the formerly omitted live-state map."""

    if rank not in AFFINE_RANK_GRID:
        raise ValueError("rank is outside the registered affine ladder")
    basis_reals = D_MODEL * COEFFICIENT_DIM if include_basis else 0
    predictor_reals = (
        2 * D_MODEL + COEFFICIENT_DIM + D_MODEL * rank
        + rank * COEFFICIENT_DIM
    )
    multiplies = (
        D_MODEL * rank + rank * COEFFICIENT_DIM
        + 2 * D_MODEL * COEFFICIENT_DIM
    )
    total = basis_reals + predictor_reals
    original = 3 * NATIVE_PRODUCTS * D_MODEL + D_MODEL
    return {
        "rank": int(rank),
        "include_basis": bool(include_basis),
        "basis_reals": int(basis_reals),
        "predictor_reals": int(predictor_reals),
        "total_reals": int(total),
        "float32_bits": int(32 * total),
        "inference_multiplies_per_token": int(multiplies),
        "hadamard_products_per_token": 0,
        "original_mlp_reals": int(original),
        "fraction_of_original_reals": total / original,
    }


def empirical_fisher_loss(
    error: torch.Tensor,
    adjoint: torch.Tensor,
    *,
    isotropic_floor: float = CAUSAL_ISOTROPIC_FLOOR,
    directional_denominator: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Registered causal fit loss with a nonzero Euclidean identifiability floor."""

    error = error.double()
    adjoint = adjoint.double()
    if error.ndim != 2 or error.shape != adjoint.shape:
        raise ValueError("error and adjoint must be aligned matrices")
    if not torch.isfinite(error).all() or not torch.isfinite(adjoint).all():
        raise ValueError("error and adjoint must be finite")
    if not 0.0 < isotropic_floor <= 1.0:
        raise ValueError("isotropic floor must be in (0,1]")
    denominator = (adjoint.square().sum(dim=1).mean()
                   if directional_denominator is None
                   else torch.as_tensor(directional_denominator, dtype=torch.float64,
                                        device=error.device))
    if denominator.ndim != 0 or not torch.isfinite(denominator):
        raise ValueError("causal directional denominator must be a finite scalar")
    if float(denominator) <= 0.0:
        raise ValueError("causal adjoints have zero energy")
    directional = (adjoint * error).sum(dim=1).square().mean() / denominator
    # The tensor mean already equals mean_t ||e_t||^2 / 64.  Dividing by the
    # coefficient count again would make the registered floor 64x too small.
    isotropic = error.square().mean()
    return directional + isotropic_floor * isotropic


def transport_signed_output_gauge(
    state: Mapping[str, Any], basis: torch.Tensor, signs: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor]:
    """Transport a native program through B' = B diag(signs)."""

    basis = basis.double()
    signs = signs.double().flatten()
    if basis.ndim != 2 or signs.shape != (basis.shape[1],):
        raise ValueError("signed gauge dimensions do not align")
    if not bool(torch.all((signs == 1.0) | (signs == -1.0))):
        raise ValueError("signed gauge entries must be exactly +/-1")
    moved = dict(state)
    moved["projected_decoder"] = state["projected_decoder"].double() * signs
    moved["beta"] = state["beta"].double() * signs
    return moved, basis * signs
