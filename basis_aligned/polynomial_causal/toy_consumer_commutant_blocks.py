"""Known-answer toy for consumer-common circuit blocks.

The proposed real-model object is a family of positive-semidefinite pullback forms

    G_c = J_c^T W_c J_c,

on one component's error/write coordinates, one form per downstream consumer c.  A
common reducing subspace is a circuit block that every G_c preserves.  The family is
simultaneously block diagonal in a suitable basis exactly when its generated matrix
*-algebra is reducible.  Its commutant contains the corresponding block projectors.

This CPU toy plants three blocks behind a random orthogonal gauge, verifies that the
commutant recovers them, checks gauge-invariant spectra and additive cross-block edits,
and includes a generic dense family which must have only the trivial scalar commutant.
It validates code and algebra only; it is not evidence that bilin18 has these blocks.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
RECEIPT = HERE / "toy_consumer_commutant_blocks_receipt.json"


def _symmetric(matrix: torch.Tensor) -> torch.Tensor:
    return 0.5 * (matrix + matrix.T)


def whiten_forms(forms: torch.Tensor, ridge: float = 1e-10) -> torch.Tensor:
    """Whiten the sum of PSD forms so consumer magnitudes cannot set the answer."""
    aggregate = _symmetric(forms.sum(0))
    eigenvalues, eigenvectors = torch.linalg.eigh(aggregate)
    floor = ridge * eigenvalues.abs().max().clamp_min(1.0)
    inverse_sqrt = eigenvectors @ torch.diag(
        eigenvalues.clamp_min(floor).rsqrt()
    ) @ eigenvectors.T
    return torch.stack([_symmetric(inverse_sqrt @ form @ inverse_sqrt) for form in forms])


def commutator_operator(forms: torch.Tensor) -> torch.Tensor:
    """Matrix K satisfying ||K vec(X)||^2=sum_c ||G_c X-X G_c||_F^2."""
    _, dimension, _ = forms.shape
    columns = []
    for flat_index in range(dimension * dimension):
        basis = torch.zeros(dimension, dimension, dtype=forms.dtype)
        basis.reshape(-1)[flat_index] = 1.0
        columns.append(torch.cat([
            (form @ basis - basis @ form).reshape(-1) for form in forms
        ]))
    return torch.stack(columns, dim=1)


def commutant_spectrum(forms: torch.Tensor) -> torch.Tensor:
    """Ascending singular values of the commutator operator."""
    return torch.linalg.svdvals(commutator_operator(forms)).flip(0)


def commutant_basis(forms: torch.Tensor, tolerance: float = 1e-9) -> torch.Tensor:
    """Return an orthonormal basis of matrices commuting with every form."""
    operator = commutator_operator(forms)
    _, singular_values, vh = torch.linalg.svd(operator, full_matrices=False)
    cutoff = tolerance * singular_values.max().clamp_min(1.0)
    null_rows = vh[singular_values <= cutoff]
    dimension = forms.shape[-1]
    return null_rows.reshape(-1, dimension, dimension)


def recover_block_basis(
    forms: torch.Tensor, tolerance: float = 1e-9, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Recover blocks from a generic self-adjoint element of the commutant.

    Returns an orthogonal basis and integer labels.  For a generic irreducible family
    within each planted block, every commutant element is scalar on that block, so a
    generic combination has one repeated eigenvalue per block.
    """
    basis = commutant_basis(forms, tolerance)
    generator = torch.Generator().manual_seed(seed)
    coefficients = torch.randn(len(basis), generator=generator, dtype=forms.dtype)
    witness = _symmetric(torch.einsum("a,aij->ij", coefficients, basis))
    values, vectors = torch.linalg.eigh(witness)
    scale = values.abs().max().clamp_min(1.0)
    labels = torch.zeros(len(values), dtype=torch.long)
    label = 0
    for index in range(1, len(values)):
        if abs(values[index] - values[index - 1]) > 100 * tolerance * scale:
            label += 1
        labels[index] = label
    return vectors, labels


def offblock_fraction(forms: torch.Tensor, basis: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of squared form energy lying between recovered blocks."""
    transformed = torch.einsum("ia,cij,jb->cab", basis, forms, basis)
    cross = labels[:, None] != labels[None, :]
    return float((transformed[:, cross] ** 2).sum() / (transformed**2).sum().clamp_min(1e-30))


def edit_interaction(form: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> float:
    """q(left+right)-q(left)-q(right); zero for edits in distinct exact blocks."""
    quadratic = lambda vector: vector @ form @ vector
    return float(quadratic(left + right) - quadratic(left) - quadratic(right))


def planted_family(
    seed: int = 0,
    block_sizes: tuple[int, ...] = (2, 3, 2),
    consumers: int = 5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return gauged PSD forms, the hiding gauge, and original-coordinate labels."""
    dtype = torch.float64
    generator = torch.Generator().manual_seed(seed)
    dimension = sum(block_sizes)
    labels = torch.cat([
        torch.full((size,), index, dtype=torch.long)
        for index, size in enumerate(block_sizes)
    ])
    forms = []
    for _ in range(consumers):
        blocks = []
        for size in block_sizes:
            factor = torch.randn(size, size, generator=generator, dtype=dtype)
            blocks.append(factor @ factor.T + 0.4 * torch.eye(size, dtype=dtype))
        forms.append(torch.block_diag(*blocks))
    forms = torch.stack(forms)
    gauge = torch.linalg.qr(
        torch.randn(dimension, dimension, generator=generator, dtype=dtype)
    ).Q
    gauged = torch.einsum("ia,cij,jb->cab", gauge, forms, gauge)
    return gauged, gauge, labels


def dense_null_family(seed: int = 23, dimension: int = 7, consumers: int = 5) -> torch.Tensor:
    """Generic dense PSD forms; almost surely only scalar multiples of I commute."""
    generator = torch.Generator().manual_seed(seed)
    dtype = torch.float64
    forms = []
    for _ in range(consumers):
        factor = torch.randn(dimension, dimension, generator=generator, dtype=dtype)
        forms.append(factor @ factor.T + 0.4 * torch.eye(dimension, dtype=dtype))
    return torch.stack(forms)


def run_checks() -> dict[str, object]:
    forms, gauge, true_labels = planted_family()
    whitened = whiten_forms(forms)
    spectrum = commutant_spectrum(whitened)
    basis = commutant_basis(whitened)
    recovered_basis, recovered_labels = recover_block_basis(whitened)
    recovered_sizes = sorted(int((recovered_labels == value).sum()) for value in recovered_labels.unique())

    # Orthogonal coordinate changes must leave the commutator singular values fixed.
    generator = torch.Generator().manual_seed(41)
    second_gauge = torch.linalg.qr(torch.randn(7, 7, generator=generator, dtype=torch.float64)).Q
    regauged = torch.einsum("ia,cij,jb->cab", second_gauge, whitened, second_gauge)
    regauged_spectrum = commutant_spectrum(regauged)

    dense = whiten_forms(dense_null_family())
    dense_spectrum = commutant_spectrum(dense)
    dense_basis = commutant_basis(dense)

    # A small cross-block perturbation should turn the exact zero modes into small,
    # nonzero singular values instead of being silently rounded into exact structure.
    noise = torch.randn(forms.shape, generator=generator, dtype=torch.float64)
    noise = 0.5 * (noise + noise.transpose(-1, -2))
    perturbed = whiten_forms(forms + 0.015 * noise)
    perturbed_spectrum = commutant_spectrum(perturbed)

    # In the hidden block coordinates, edits in blocks zero and one have no quadratic
    # interaction.  A dense-null consumer generally couples the same two vectors.
    left_hidden = torch.zeros(7, dtype=torch.float64); left_hidden[0] = 0.8
    right_hidden = torch.zeros(7, dtype=torch.float64); right_hidden[3] = -0.6
    left, right = gauge.T @ left_hidden, gauge.T @ right_hidden
    planted_interactions = [edit_interaction(form, left, right) for form in forms]
    dense_interactions = [edit_interaction(form, left, right) for form in dense]

    passed = (
        len(basis) == 3
        and recovered_sizes == sorted([2, 3, 2])
        and offblock_fraction(whitened, recovered_basis, recovered_labels) < 1e-20
        and torch.allclose(spectrum, regauged_spectrum, atol=1e-10, rtol=1e-10)
        and len(dense_basis) == 1
        and max(abs(value) for value in planted_interactions) < 1e-12
        and max(abs(value) for value in dense_interactions) > 1e-2
        and perturbed_spectrum[1] > 1e-6
        and perturbed_spectrum[2] > 1e-6
        and perturbed_spectrum[2] < perturbed_spectrum[3]
    )
    return {
        "schema": "toy_consumer_commutant_blocks_v1",
        "purpose": "known-answer validation only; not real-model evidence",
        "planted_commutant_dimension": len(basis),
        "planted_smallest_singular_values": [float(x) for x in spectrum[:6]],
        "recovered_block_sizes": recovered_sizes,
        "recovered_offblock_energy_fraction": offblock_fraction(
            whitened, recovered_basis, recovered_labels
        ),
        "gauge_spectrum_max_abs_difference": float((spectrum - regauged_spectrum).abs().max()),
        "dense_null_commutant_dimension": len(dense_basis),
        "dense_null_smallest_singular_values": [float(x) for x in dense_spectrum[:4]],
        "perturbed_smallest_singular_values": [float(x) for x in perturbed_spectrum[:6]],
        "planted_cross_block_edit_interactions": planted_interactions,
        "dense_cross_block_edit_interactions": dense_interactions,
        "all_passed": bool(passed),
    }


def main() -> None:
    started = time.monotonic()
    result = run_checks()
    result["runtime_s"] = time.monotonic() - started
    RECEIPT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
