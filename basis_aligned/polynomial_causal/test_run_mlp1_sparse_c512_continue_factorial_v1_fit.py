import pytest

import run_mlp1_sparse_c512_continue_factorial_v1_fit as subject


def record(seed, final, values=None):
    values = values or [final - 0.002, final - 0.001, final]
    curve = [
        {"step": step, "select_r2": value, "train_mse": 1.0, "learning_rate": 0.1}
        for step, value in zip(range(200, 2401, 200), [values[0]] * 9 + values)
    ]
    return {
        "seed": seed,
        "final_select_r2": final,
        "curve": curve,
        "convergence": subject.convergence_metrics(curve),
    }


def test_selection_uses_only_final_r2_and_breaks_tie_by_seed():
    rows = [record(0, 0.7), record(1, 0.8), record(2, 0.8)]
    assert subject.select_seed(rows)["seed"] == 1


def test_convergence_and_admission_are_separate():
    rows = [record(0, 0.79), record(1, 0.80), record(2, 0.81)]
    gates = subject.selection_gates(rows, 0.91)
    assert gates["selected_seed"] == 2
    assert gates["selected_curve_converged"] is True
    assert gates["seed_final_select_r2_std_le_0p02"] is True
    assert gates["admitted_to_final"] is True
    failed = subject.selection_gates(rows, 0.899)
    assert failed["admitted_to_final"] is False


def test_convergence_rejects_late_instability_and_wrong_cadence():
    unstable = record(0, 0.80, [0.77, 0.82, 0.80])
    assert unstable["convergence"]["converged"] is False
    with pytest.raises(RuntimeError, match="cadence"):
        subject.convergence_metrics(unstable["curve"][:-1])
