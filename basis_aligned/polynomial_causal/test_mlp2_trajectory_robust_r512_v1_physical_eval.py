from __future__ import annotations

import torch

import prepare_mlp2_trajectory_robust_r512_v1_eval_rows as row_life
import run_mlp2_trajectory_robust_r512_v1_physical_eval as assay


def fake_ledger(dce: float, kl: float | None = None) -> torch.Tensor:
    value = torch.zeros(192, 9, dtype=torch.float64)
    value[:, 0] = 10.0
    value[:, 1] = 10.0 + dce * 192
    value[:, 2] = (dce if kl is None else kl) * 192
    value[:, 4] = 1.0
    value[:, 5:8] = 192
    value[:, 8] = 192
    return value


def test_call_census_replaces_exact_sites_and_programs() -> None:
    census = assay.expected_call_census()
    assert set(census) == set(assay.ARMS)
    assert census["NATIVE"]["native_mlp_sites"]["0"] == 48
    assert census["C512_ROBUST512"]["native_mlp_sites"]["0"] == 0
    assert census["C512_ROBUST512"]["native_mlp_sites"]["2"] == 0
    assert census["C512_ROBUST512"]["candidate_c512"] == 48
    assert census["C512_ROBUST512"]["candidate_mlp2"] == {
        "FULL512": 0, "CONTINUE512": 0, "ROBUST512": 48,
    }


def test_factorial_document_is_zero_for_additive_effects() -> None:
    ledgers = {arm: fake_ledger(0.0) for arm in assay.ARMS}
    ledgers["C512"] = fake_ledger(0.003)
    ledgers["FULL512"] = fake_ledger(0.05)
    ledgers["C512_FULL512"] = fake_ledger(0.053)
    assert torch.allclose(assay.factorial_document(ledgers, "FULL512"),
                          torch.zeros(192, dtype=torch.float64), atol=1e-12)


def test_simultaneous_contrasts_use_same_wave_values() -> None:
    ledgers = {arm: fake_ledger(0.0) for arm in assay.ARMS}
    ledgers.update({
        "C512": fake_ledger(0.003, 0.004),
        "FULL512": fake_ledger(0.050, 0.055),
        "C512_FULL512": fake_ledger(0.062, 0.068),
        "CONTINUE512": fake_ledger(0.049, 0.054),
        "C512_CONTINUE512": fake_ledger(0.060, 0.066),
        "ROBUST512": fake_ledger(0.048, 0.053),
        "C512_ROBUST512": fake_ledger(0.052, 0.058),
    })
    out = assay.simultaneous_contrasts(ledgers)
    assert abs(out["fresh_full_interaction"]["point"] - 0.009) < 1e-12
    assert abs(out["combined_dce_gain_vs_full"]["point"] - 0.010) < 1e-12
    assert abs(out["combined_dce_gain_vs_continue"]["point"] - 0.008) < 1e-12
    assert out["robust_absolute_interaction"]["point"] < 0.002


def test_optimization_inconclusive_rule_requires_every_gate() -> None:
    curve = []
    for step in range(0, 1201, 25):
        curve.append({"step": step, "worst_normalized_mse": 0.5 - step / 120000.0})
    bundle = {"curves": {"ROBUST512": curve}}
    result = assay.optimization_inconclusive(bundle)
    assert result["gates"]["last_four_strictly_decreasing"]
    assert not result["gates"]["step1200_improves_at_least_1pct_vs_1100"]
    assert not result["applies"]


def test_fresh_row_wrapper_configures_one_evaluation_role() -> None:
    names = (
        "PREREG", "FREEZER", "RUNNER", "TEST", "AUDIT", "SOURCE_PATHS",
        "START_DOCUMENT_INDEX", "DOCUMENTS_PER_ROLE", "TOTAL_DOCUMENTS",
        "ROLE_NAMES", "ROLE_AUTHORIZATIONS", "CACHE", "RECEIPT", "FAILURE",
        "LOCK", "RECEIPT_SCHEMA", "FAILURE_SCHEMA", "source_hashes",
        "validate_independent_audit",
    )
    before = {name: getattr(row_life.base, name) for name in names}
    try:
        row_life.configure()
        assert row_life.base.START_DOCUMENT_INDEX == 120_000
        assert row_life.base.TOTAL_DOCUMENTS == 192
        assert row_life.base.ROLE_NAMES == ("EVALUATION",)
        assert row_life.base.ROLE_AUTHORIZATIONS == {
            "EVALUATION": {"authorized_for_training": False,
                           "authorized_for_evaluation": True},
        }
    finally:
        for name, value in before.items():
            setattr(row_life.base, name, value)


def test_known_answer_passes_all_scientific_gates() -> None:
    ledgers = {arm: fake_ledger(0.0) for arm in assay.ARMS}
    ledgers.update({
        "C512": fake_ledger(0.003, 0.004),
        "FULL512": fake_ledger(0.050, 0.055),
        "C512_FULL512": fake_ledger(0.064, 0.069),
        "CONTINUE512": fake_ledger(0.049, 0.054),
        "C512_CONTINUE512": fake_ledger(0.060, 0.066),
        "ROBUST512": fake_ledger(0.049, 0.054),
        "C512_ROBUST512": fake_ledger(0.053, 0.059),
    })
    curve = [{"step": step, "worst_normalized_mse": 0.44}
             for step in range(0, 1201, 25)]
    result = assay.derive_result(
        ledgers, runtime=1.0, bundle={"curves": {"ROBUST512": curve}},
    )
    assert result["status"] == "trajectory_exposure_supported"
    assert all(result["decision_gates"].values())
