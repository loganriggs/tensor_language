import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run_temporal_iswas_l11h3_source_writer_formula_mediation_v2 as target


def apply_rotary_emb(value, _cos, _sin):
    return value


def test_price_and_unbound_dryrun_are_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True, env=dict(os.environ, BQLIB_DRYRUN="1"))
    payload = json.loads(completed.stdout)
    assert payload["awaiting_binding"] is True
    assert payload["base_authority_ok"] is True
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["price"]["model_forwards"] == 18


def test_formula_zero_delta_is_zero():
    class Rotary:
        lamb = .2
        def rotary(self, value):
            shape = (value.shape[1], value.shape[-1] // 2)
            return torch.ones(shape), torch.zeros(shape)
    batch, tokens = 2, 3
    native = {name: torch.randn(batch, tokens, 9 * 128)
              for name in ("q", "k", "q2", "k2", "v")}
    result = target.recipient_native_source_formula(native, native["v"].clone(),
        [[0], [1]], Rotary(), torch, torch.nn.functional)
    assert result.shape == (batch, tokens, 128)
    assert torch.count_nonzero(result) == 0


def test_prediction_inventory_matches_gate_marker():
    assert len(target.PREDICTION_KEYS) == 5
    assert target.PRICE == json.loads(target.PRIOR_V2.read_text())["amended_price"]


def test_bound_authority_is_identical_for_dryrun_and_execution(tmp_path, monkeypatch):
    selected = {
        "temporal": {"arm": "P3", "length": 3,
                     "heads": ["L07H07", "L09H04", "L09H01"]},
        "iswas": {"arm": "P2", "length": 2, "heads": ["L07H07", "L09H04"]},
    }
    greedy = {"selected_prefixes": selected,
              "predictions": {key: index != 3 for index, key in
                              enumerate(target.greedy_contract.PREDICTION_KEYS)}}
    greedy_path = tmp_path / "greedy.json"
    greedy_path.write_text(json.dumps(greedy))
    monkeypatch.setattr(target, "GREEDY", greedy_path)
    binding = {
        "candidate_id": "cross_task.temporal_iswas.l11h3_source_writer_formula_mediation_v2",
        "conditional_prior_sha256": target.EXPECTED["prior_v1"],
        "price_amendment_sha256": target.EXPECTED["prior_v2"],
        "mediation_runner_sha256": target.sha(target.SELF),
        "greedy_result_sha256": target.sha(greedy_path),
        "selected_prefixes": selected,
    }
    assert target.bound_authority_ok(target.EXPECTED, binding, greedy)
    binding["selected_prefixes"] = {**selected, "temporal": selected["iswas"]}
    assert not target.bound_authority_ok(target.EXPECTED, binding, greedy)
