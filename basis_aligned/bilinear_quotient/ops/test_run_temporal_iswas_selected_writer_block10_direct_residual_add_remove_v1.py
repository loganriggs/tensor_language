import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import run_temporal_iswas_selected_writer_block10_direct_residual_add_remove_v1 as target


def test_bound_dryrun_is_authority_checked_and_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["static_authority_ok"] is True
    assert payload["authority_ok"] is True
    assert payload["status"] == "bound"
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 14


def test_inventory_and_price_match_frozen_prior():
    prior = json.loads(target.PRIOR.read_text())
    assert target.ARMS == ("writer", "R1M0_dynamic", "R0M1_dynamic",
                           "native_x18_self_patch", "direct_add", "direct_remove")
    assert prior["price"] == target.PRICE
    assert prior["fixed_writer"]["heads"] == ["L07H07", "L09H04"]
    assert len(target.PREDICTION_KEYS) == 5


def test_expected_static_authorities_match_bytes():
    assert {name: target.sha(path) for name, path in target.FILES.items()} == target.EXPECTED


def test_state_metric_uses_only_registered_positions():
    predicted = torch.zeros(2, 3, 2)
    observed = predicted.clone()
    observed[0, 2] = 1000
    observed[1, 0] = 1000
    assert target.state_relative_l2(observed, predicted, [[0, 1], [1, 2]]) == 0.0
    predicted[0, 0, 0] = 1.0
    observed[0, 0, 0] = 2.0
    assert target.state_relative_l2(observed, predicted, [[0, 1], [1, 2]]) == pytest.approx(1.0)


def test_binding_is_absent_until_parent_result_exists(monkeypatch, tmp_path):
    monkeypatch.setattr(target, "BINDING", tmp_path / "missing_binding.json")
    monkeypatch.setattr(target, "FACTORIAL_RESULT", tmp_path / "missing_result.json")
    observed = {name: target.sha(path) for name, path in target.FILES.items()}
    state, ok = target.binding_state(observed)
    assert state is None and ok is False
