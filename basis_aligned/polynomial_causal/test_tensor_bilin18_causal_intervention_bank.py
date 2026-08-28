from pathlib import Path

import torch

import tensor_bilin18_causal_intervention_bank as bank


def test_preregistration_and_create_only_result_path():
    assert bank.PREREG.exists()
    assert bank.OUTPUT.name == "tensor_bilin18_causal_intervention_bank_results.json"
    assert bank.RANKS == (512, 640)
    assert bank.N_BOOTSTRAP == 10_000
    assert bank.EVAL_BATCH == 4


def test_exact_candidate_prices():
    assert bank.EXPECTED_TOTALS[512] == 503_436_726
    assert bank.EXPECTED_TOTALS[640] == 516_707_766
    assert bank.EXPECTED_TOTALS[640] < bank.DENSE_TOTAL


def test_synthetic_fixtures_are_deterministic_and_distinct():
    first = bank.synthetic_tokens(0)
    second = bank.synthetic_tokens(1)
    assert first.shape == (256,)
    assert first.dtype == torch.long
    assert torch.equal(first, bank.synthetic_tokens(0))
    assert not torch.equal(first, second)


def test_delta_metrics_identity_and_zero_program():
    native = torch.tensor([[1.0, -2.0, 3.0]])
    identity = bank.delta_metrics(native, native.clone())
    assert identity["recovery"] == 1.0
    assert abs(identity["cosine"] - 1.0) < 1e-12
    zero = bank.delta_metrics(native, torch.zeros_like(native))
    assert zero["recovery"] == 0.0
    assert zero["cosine"] == 0.0


def test_distributional_gate_requires_all_three_conditions():
    passing = {
        "recovery_bootstrap_95_lcb": 0.90,
        "cosine_bootstrap_95_lcb": 0.95,
        "individual_joint_pass_fraction": 0.75,
        "all_native_and_program_signals_nonzero": True,
    }
    assert bank.robust_gate(passing)
    for key in (
        "recovery_bootstrap_95_lcb", "cosine_bootstrap_95_lcb",
        "individual_joint_pass_fraction",
    ):
        failing = dict(passing)
        failing[key] -= 1e-6
        assert not bank.robust_gate(failing)


def test_bootstrap_is_fixed_seed_and_below_mean_for_spread():
    values = [0.80, 0.85, 0.90, 0.95, 1.00]
    first = bank.bootstrap_lcb(values)
    second = bank.bootstrap_lcb(values)
    assert first == second
    assert first < sum(values) / len(values)


def test_sources_include_preregistration_and_test():
    names = {Path(path).name for path in bank.SOURCES}
    assert bank.PREREG.name in names
    assert Path(__file__).name in names
