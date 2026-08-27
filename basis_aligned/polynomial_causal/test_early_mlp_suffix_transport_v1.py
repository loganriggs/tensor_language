from __future__ import annotations

import torch

import early_mlp_suffix_transport_v1 as contract


def _orthonormal(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    value = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    return torch.linalg.qr(value, mode="reduced").Q


def test_complete_gauge_rewrite_preserves_physical_operator_and_write() -> None:
    generator = torch.Generator().manual_seed(20260828)
    b0 = _orthonormal(contract.D_MODEL, contract.CODE_DIM, generator)
    b1 = _orthonormal(contract.D_MODEL, contract.CODE_DIM, generator)
    q0 = _orthonormal(contract.CODE_DIM, contract.CODE_DIM, generator)
    q1 = _orthonormal(contract.CODE_DIM, contract.CODE_DIM, generator)
    cross = torch.randn(
        contract.CODE_DIM, contract.CODE_DIM, generator=generator,
        dtype=torch.float64,
    )
    source = torch.randn(7, contract.D_MODEL, generator=generator, dtype=torch.float64)

    before = contract.physical_cross_map(b0, cross, b1)
    write_before = contract.transported_physical_write(source, b0, cross, b1)
    moved_b0, moved_cross, moved_b1 = contract.rewrite_cross_map_gauge(
        b0, cross, b1, q0, q1,
    )
    after = contract.physical_cross_map(moved_b0, moved_cross, moved_b1)
    write_after = contract.transported_physical_write(
        source, moved_b0, moved_cross, moved_b1,
    )

    assert torch.equal(before, after) or torch.allclose(before, after, atol=2e-13, rtol=0)
    assert torch.equal(write_before, write_after) or torch.allclose(
        write_before, write_after, atol=2e-12, rtol=0,
    )


def test_complete_gauge_rewrite_preserves_full_affine_transport_program() -> None:
    generator = torch.Generator().manual_seed(27182818)
    b0 = _orthonormal(contract.D_MODEL, contract.CODE_DIM, generator)
    b1 = _orthonormal(contract.D_MODEL, contract.CODE_DIM, generator)
    q0 = _orthonormal(contract.CODE_DIM, contract.CODE_DIM, generator)
    q1 = _orthonormal(contract.CODE_DIM, contract.CODE_DIM, generator)
    w0 = torch.randn(
        contract.D_MODEL, contract.CODE_DIM, generator=generator, dtype=torch.float64,
    )
    w1 = torch.randn(
        contract.D_MODEL, contract.CODE_DIM, generator=generator, dtype=torch.float64,
    )
    b0_aff = torch.randn(contract.CODE_DIM, generator=generator, dtype=torch.float64)
    b1_aff = torch.randn(contract.CODE_DIM, generator=generator, dtype=torch.float64)
    cross = torch.randn(
        contract.CODE_DIM, contract.CODE_DIM, generator=generator,
        dtype=torch.float64,
    )
    x0 = torch.randn(5, contract.D_MODEL, generator=generator, dtype=torch.float64)
    x1 = torch.randn(5, contract.D_MODEL, generator=generator, dtype=torch.float64)
    delta = torch.randn(5, contract.CODE_DIM, generator=generator, dtype=torch.float64)

    p0 = x0 @ w0 + b0_aff + delta
    p1 = x1 @ w1 + b1_aff + p0 @ cross
    physical_before = p1 @ b1.T

    moved_b0, moved_cross, moved_b1 = contract.rewrite_cross_map_gauge(
        b0, cross, b1, q0, q1,
    )
    moved_w0, moved_b0_aff = contract.rewrite_affine_output_gauge(w0, b0_aff, q0)
    moved_w1, moved_b1_aff = contract.rewrite_affine_output_gauge(w1, b1_aff, q1)
    moved_delta = contract.rewrite_code_gauge(delta, q0)
    moved_p0 = x0 @ moved_w0 + moved_b0_aff + moved_delta
    moved_p1 = x1 @ moved_w1 + moved_b1_aff + moved_p0 @ moved_cross
    physical_after = moved_p1 @ moved_b1.T
    teacher_label0 = torch.randn(
        5, contract.CODE_DIM, generator=generator, dtype=torch.float64,
    )
    moved_teacher_label0 = contract.rewrite_code_gauge(teacher_label0, q0)
    physical_edit_before = delta @ b0.T
    physical_edit_after = moved_delta @ moved_b0.T
    physical_label_before = teacher_label0 @ b0.T
    physical_label_after = moved_teacher_label0 @ moved_b0.T

    assert torch.allclose(moved_p0, contract.rewrite_code_gauge(p0, q0), atol=2e-12, rtol=0)
    assert torch.allclose(physical_before, physical_after, atol=5e-11, rtol=0)
    assert torch.allclose(physical_edit_before, physical_edit_after, atol=2e-12, rtol=0)
    assert torch.allclose(physical_label_before, physical_label_after, atol=2e-12, rtol=0)


def test_incomplete_gauge_rewrite_is_detected() -> None:
    generator = torch.Generator().manual_seed(314159)
    b0 = _orthonormal(contract.D_MODEL, contract.CODE_DIM, generator)
    b1 = _orthonormal(contract.D_MODEL, contract.CODE_DIM, generator)
    q0 = _orthonormal(contract.CODE_DIM, contract.CODE_DIM, generator)
    cross = torch.randn(
        contract.CODE_DIM, contract.CODE_DIM, generator=generator,
        dtype=torch.float64,
    )
    before = contract.physical_cross_map(b0, cross, b1)
    incomplete = contract.physical_cross_map(b0 @ q0, cross, b1)
    assert float((before - incomplete).abs().max()) > 1e-3


def test_price_is_coordinate_independent() -> None:
    assert contract.incremental_price() == contract.incremental_price()
    assert contract.incremental_price()["incremental_reals"] == 4096


def test_invalid_gauge_fails_closed() -> None:
    bad = torch.eye(contract.CODE_DIM, dtype=torch.float64)
    bad[0, 0] = 2
    try:
        contract.validate_orthogonal_gauge("bad", bad)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("nonorthogonal gauge was accepted")


def test_row_code_edit_is_a_physical_row_vector_write() -> None:
    generator = torch.Generator().manual_seed(1618033)
    basis = _orthonormal(contract.D_MODEL, contract.CODE_DIM, generator)
    residual = torch.randn(3, contract.D_MODEL, generator=generator, dtype=torch.float64)
    delta = torch.randn(3, contract.CODE_DIM, generator=generator, dtype=torch.float64)

    edited = contract.apply_physical_code_edit(residual, delta, basis)
    projected_change = (edited - residual) @ basis
    assert torch.allclose(projected_change, delta, atol=2e-12, rtol=0)


def test_covariance_shaped_direction_bank_is_reproducible_and_normalized() -> None:
    generator = torch.Generator().manual_seed(141421)
    codes = torch.randn(128, contract.CODE_DIM, generator=generator, dtype=torch.float64)
    first = contract.covariance_shaped_directions(codes)
    second = contract.covariance_shaped_directions(codes)

    assert torch.equal(first["raw_signs"], second["raw_signs"])
    assert torch.equal(first["directions"], second["directions"])
    assert tuple(first["directions"].shape) == (32, contract.CODE_DIM)
    rms = torch.sqrt(torch.mean(first["directions"].square(), dim=1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=2e-13, rtol=0)
    reconstructed = (
        first["raw_signs"] @ first["covariance_square_root"]
    )
    reconstructed /= torch.sqrt(torch.mean(reconstructed.square(), dim=1, keepdim=True))
    assert torch.allclose(first["directions"], reconstructed, atol=2e-13, rtol=0)


def test_pooled_response_metrics_use_ratio_of_sums() -> None:
    teacher = torch.tensor([[1.0, -1.0], [3.0, -3.0]], dtype=torch.float64)
    student = 0.5 * teacher
    metrics = contract.pooled_response_metrics(student, teacher)
    assert abs(metrics["nre"] - 0.5) < 1e-14
    assert abs(metrics["r2"] - 0.75) < 1e-14
    assert abs(metrics["cosine"] - 1.0) < 1e-14

    offset_student = student + 100
    centered = contract.pooled_response_metrics(
        offset_student, teacher, center_last_dimension=True,
    )
    assert abs(centered["nre"] - 0.5) < 1e-14


def test_finite_null_rank_counts_ties_against_primary() -> None:
    nulls = torch.tensor([0.1, 0.2, 0.3, 0.2], dtype=torch.float64)
    assert contract.finite_null_rank(0.2, nulls) == 4
    assert contract.finite_null_rank(0.4, nulls) == 1
