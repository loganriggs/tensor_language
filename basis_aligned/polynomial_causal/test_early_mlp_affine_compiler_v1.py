from __future__ import annotations

import torch

import early_mlp_affine_compiler_v1 as compiler


def test_complete_registered_lattice_and_names() -> None:
    assert len(compiler.ARM_STATES) == 18
    assert len(set(compiler.ARM_STATES)) == 18
    assert compiler.arm_name(("Q", "O", "E")) == "QOE"


def test_balanced_factorization_is_exact_at_full_rank_and_sign_canonical() -> None:
    generator = torch.Generator().manual_seed(7)
    weight = torch.randn(11, 5, generator=generator, dtype=torch.float64)
    left, right = compiler.balanced_factors(weight, 5)
    assert torch.allclose(left @ right, weight, atol=1e-10, rtol=1e-10)
    for column in range(left.shape[1]):
        pivot = int(left[:, column].abs().argmax())
        assert left[pivot, column] >= 0.0


def test_ridge_frontier_recovers_low_rank_affine_map() -> None:
    generator = torch.Generator().manual_seed(11)
    x = torch.randn(320, 12, generator=generator, dtype=torch.float64)
    left = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    right = torch.randn(3, 6, generator=generator, dtype=torch.float64)
    bias = torch.randn(6, generator=generator, dtype=torch.float64)
    y = x @ left @ right + bias
    state, frontier = compiler.fit_ridge_frontier(
        x[:240], y[:240], x[240:], y[240:],
        lambdas=(0.0, 1e-4), ranks=(2, 3, 6), selection_slack=1.01,
    )
    prediction = compiler.affine_predict(x[240:], state)
    assert state["rank"] == 3
    assert compiler.coefficient_metrics(prediction, y[240:])["nmse"] < 1e-18
    assert len(frontier) == 6


def _synthetic_rows() -> tuple[dict[tuple[str, str, str], list[float]], list[str]]:
    # Additive gains make every Q effect positive, Q retain 75% of O, and joint Q
    # beat the maximum singleton. E contributes a fixed background gain.
    rows = {}
    for arm in compiler.ARM_STATES:
        s0 = {"N": 0.0, "Q": 0.30, "O": 0.40}[arm[0]]
        s1 = {"N": 0.0, "Q": 0.30, "O": 0.40}[arm[1]]
        s2 = {"N": 0.0, "E": 0.10}[arm[2]]
        gain = s0 + s1 + s2
        rows[arm] = [3.0 - gain + 0.001 * (row % 3) for row in range(12)]
    documents = [f"doc-{row // 2}" for row in range(12)]
    return rows, documents


def test_lattice_analysis_recomputes_registered_non_linear_gates() -> None:
    rows, documents = _synthetic_rows()
    mean_rows = [2.80 + 0.001 * (row % 3) for row in range(12)]
    shuffle_rows = [2.75 + 0.001 * (row % 3) for row in range(12)]
    result = compiler.compiler_lattice_analysis(
        rows, documents, mean_control_rows=mean_rows,
        shuffle_control_rows=shuffle_rows, draws=200, seed=3,
    )
    assert len(result["core_no_free_rider"]) == 8
    assert len(result["oracle_neighbor_reuse"]) == 4
    assert len(result["same_background_fidelity"]) == 8
    assert result["decisions"]["all_statistical_gates"] is True


def test_lattice_analysis_rejects_incomplete_cube() -> None:
    rows, documents = _synthetic_rows()
    rows.pop(("O", "O", "E"))
    try:
        compiler.compiler_lattice_analysis(
            rows, documents, mean_control_rows=[3.0] * 12,
            shuffle_control_rows=[3.0] * 12, draws=10,
        )
    except ValueError as error:
        assert "complete 18-arm" in str(error)
    else:
        raise AssertionError("incomplete lattice was accepted")
