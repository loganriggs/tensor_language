from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace

import run_mlp0_c512_mlp2_full512_composition_v1 as assay
import bilin18_observed_model_facade as facade


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


def test_c512_write_is_exact_declared_formula_and_bias_once() -> None:
    torch.manual_seed(4)
    left = nn.Linear(3, 5, bias=False)
    right = nn.Linear(3, 5, bias=False)
    mlp = SimpleNamespace(Left=left, Right=right, Down_bias=torch.randn(3))
    state = torch.randn(2, 4, 3)
    tensors = {
        "right": torch.randn(2, 5),
        "left": torch.randn(3, 2),
        "intercept": torch.randn(3),
    }
    event = facade.EarlyMLPEvent(
        site=0, block=SimpleNamespace(mlp=mlp), state=state,
        attention_write=torch.zeros_like(state), tokens=torch.zeros(2, 4, dtype=torch.long),
        prior_writes=(),
    )
    hidden = left(state) * right(state)
    expected = torch.nn.functional.linear(
        torch.nn.functional.linear(hidden, tensors["right"]),
        tensors["left"], tensors["intercept"],
    ) + mlp.Down_bias
    assert torch.equal(assay.c512_write(event, tensors), expected)


def test_full_call_census_tracks_all_sites_and_returns() -> None:
    census = assay.expected_call_census()
    assert census["NATIVE"]["outer_calls"] == census["NATIVE"]["outer_returns"] == 48
    assert set(census["BOTH"]["attention_sites"]) == {str(i) for i in range(18)}
    assert all(v == 48 for v in census["BOTH"]["attention_sites"].values())
    assert census["BOTH"]["native_mlp_sites"]["0"] == 0
    assert census["BOTH"]["native_mlp_sites"]["2"] == 0
    assert census["BOTH"]["native_mlp_sites"]["1"] == 48
    assert census["BOTH"]["candidate_c512"] == 48
    assert census["BOTH"]["candidate_full512"] == 48


def test_failure_guard_rejects_receipt_created_during_artifact_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = tmp_path / "lock"
    receipt = tmp_path / "receipt.json"
    failure = tmp_path / "failure.json"
    artifact = tmp_path / "ledger.pt"
    artifact.write_bytes(b"bound-ledger")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    monkeypatch.setattr(assay, "LOCK", lock)
    monkeypatch.setattr(assay, "RECEIPT", receipt)
    monkeypatch.setattr(assay, "FAILURE", failure)
    original = assay.file_sha256

    def racing_hash(path: Path) -> str:
        observed = original(path)
        if path == artifact:
            receipt.write_text("{}")
        return observed

    monkeypatch.setattr(assay, "file_sha256", racing_hash)
    claim = assay.row_life.base.acquire_claim(lock)
    try:
        with pytest.raises(RuntimeError, match="during artifact replay"):
            assay.failure_terminal_guard(claim, {artifact: digest})
    finally:
        assay.row_life.base.release_claim(claim, lock)
