from __future__ import annotations

import torch

import run_mlp0_c512_mlp2_full512_composition_v1 as assay


def fake_ledger(dce: float) -> torch.Tensor:
    value = torch.zeros(192, 9, dtype=torch.float64)
    value[:, 0] = 10.0
    value[:, 1] = 10.0 + dce * 192
    value[:, 4] = 1.0
    value[:, 5:8] = 192
    value[:, 8] = 192
    return value


def test_factorial_interaction_zero_for_additive_effects() -> None:
    ledgers = {
        "NATIVE": fake_ledger(0.0),
        "C512": fake_ledger(0.002),
        "FULL512": fake_ledger(0.05),
        "BOTH": fake_ledger(0.052),
    }
    out = assay.interaction_from_ledgers(ledgers)
    assert abs(out["interaction_dce"]) < 1e-12
    assert abs(out["full_marginal_given_c512"] - 0.05) < 1e-12
    assert abs(out["c512_marginal_given_full"] - 0.002) < 1e-12


def test_factorial_interaction_detects_incompatibility() -> None:
    ledgers = {
        "NATIVE": fake_ledger(0.0),
        "C512": fake_ledger(0.002),
        "FULL512": fake_ledger(0.05),
        "BOTH": fake_ledger(0.09),
    }
    out = assay.interaction_from_ledgers(ledgers)
    assert abs(out["interaction_dce"] - 0.038) < 1e-12
    assert out["interaction_ci95"][0] > 0.01
