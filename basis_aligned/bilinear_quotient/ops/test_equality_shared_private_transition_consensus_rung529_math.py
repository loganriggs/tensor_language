import math

import pytest
import torch

import equality_shared_private_transition_consensus_rung529_math as rung


def _deltas():
    base = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4) + 1
    return {action: base * (index + 1) for index, action in enumerate(rung.ACTIONS)}


def test_leave_one_out_consensus_omits_target_and_reconstructs_exactly():
    deltas = _deltas()
    result = rung.leave_one_out_decomposition(deltas)
    expected_z7 = torch.stack([
        rung.BETAS[action] * deltas[action] for action in ("N", "P", "Z8")
    ]).mean(0) / rung.BETAS["Z7"]
    assert torch.equal(result["Z7"]["consensus"], expected_z7)
    for action in rung.ACTIONS:
        assert torch.equal(result[action]["reconstruction"], deltas[action])


def test_single_donor_states_enumerate_all_twelve_frozen_substitutions():
    deltas = _deltas()
    result = rung.single_donor_states(deltas)
    assert sum(len(value) for value in result.values()) == 12
    expected = deltas["P"] * rung.BETAS["P"] / rung.BETAS["Z8"]
    assert torch.allclose(result["Z8"]["P"], expected, rtol=1e-15, atol=0.0)
    assert "Z8" not in result["Z8"]


def test_wrong_sign_controls_have_registered_six_target_control_pairs():
    deltas = _deltas()
    wrong = {"W7": -deltas["Z7"], "W8": -deltas["Z8"]}
    result = rung.wrong_sign_consensus_states(deltas, wrong)
    assert tuple(result["N"]) == ("W7", "W8")
    assert tuple(result["P"]) == ("W7", "W8")
    assert tuple(result["Z7"]) == ("W8",)
    assert tuple(result["Z8"]) == ("W7",)
    assert sum(len(value) for value in result.values()) == 6


def test_shape_key_and_nonfinite_checks_fail_closed():
    deltas = _deltas()
    with pytest.raises(ValueError, match="frozen order"):
        rung.aligned_states(dict(reversed(tuple(deltas.items()))))
    bad = {key: value.clone() for key, value in deltas.items()}
    bad["P"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        rung.leave_one_out_decomposition(bad)
    malformed = {key: value for key, value in deltas.items()}
    malformed["Z8"] = malformed["Z8"][..., :2]
    with pytest.raises(ValueError, match="share one shape"):
        rung.single_donor_states(malformed)


def test_discovery_rejects_missing_controls():
    good_circuit = torch.ones(2, 4, 8, dtype=torch.float64) * .01
    good_task = torch.ones(2, 4, 4, dtype=torch.float64) * .01
    with pytest.raises(ValueError, match="three single"):
        rung.score_discovery_target(
            good_circuit, good_task, good_circuit, good_task,
            {"P": good_circuit}, {"W8": -good_circuit}, [0.0] * 16)
    singles = {name: good_circuit for name in ("N", "P", "Z8")}
    with pytest.raises(ValueError, match="16 finite"):
        rung.score_discovery_target(
            good_circuit, good_task, good_circuit, good_task,
            singles, {"W8": -good_circuit}, [0.0] * 15)


def test_discovery_margin_is_strictly_against_every_single_donor():
    generator = torch.Generator().manual_seed(11)
    target = torch.randn(2, 4, 32, generator=generator, dtype=torch.float64) * .01
    task = torch.randn(2, 4, 4, generator=generator, dtype=torch.float64) * .01
    consensus = target + torch.randn(target.shape, generator=generator, dtype=torch.float64) * .0005
    task_consensus = task + torch.randn(task.shape, generator=generator, dtype=torch.float64) * .0005
    singles = {
        "N": target + torch.randn(target.shape, generator=generator, dtype=torch.float64) * .002,
        "P": target + torch.randn(target.shape, generator=generator, dtype=torch.float64) * .002,
        # An almost-perfect donor must veto the consensus advantage.
        "Z8": target + torch.randn(target.shape, generator=generator, dtype=torch.float64) * .0001,
    }
    result = rung.score_discovery_target(
        target, task, consensus, task_consensus, singles, {"W8": -target}, [0.0] * 16)
    assert not result["gates"]["d0_beats_every_single_by_005"]
    assert not result["passes_response_gates"]


def test_planted_suite_passes_all_response_gates_and_exact_split():
    result = rung.planted_suite()
    assert result["passes"]
    assert result["maximum_reconstruction_error"] <= 1e-12
    assert all(result["score"]["gates"].values())
    assert math.isfinite(result["score"]["permutation_q95"])
