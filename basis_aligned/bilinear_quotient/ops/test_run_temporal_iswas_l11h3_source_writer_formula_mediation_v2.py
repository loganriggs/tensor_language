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
