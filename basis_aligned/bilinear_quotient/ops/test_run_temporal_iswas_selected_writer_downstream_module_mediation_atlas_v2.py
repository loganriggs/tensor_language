import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run_temporal_iswas_selected_writer_downstream_module_mediation_atlas_v2 as target


def test_dryrun_is_authority_bound_and_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["modules"] == list(target.MODULES)
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 19


def test_module_inventory_is_complete_and_ordered():
    assert len(target.MODULES) == 16
    assert target.MODULES[:4] == ("A10", "M10", "A11", "M11")
    assert target.MODULES[-2:] == ("A17", "M17")


def test_replace_positions_changes_only_locked_prefix():
    native = torch.zeros(2, 4, 3)
    writer = torch.arange(24).reshape(2, 4, 3).float()
    changed = target.replace_positions(native, writer, [[0, 1], [0, 1, 2]])
    assert torch.equal(changed[0, :2], writer[0, :2])
    assert torch.equal(changed[1, :3], writer[1, :3])
    assert torch.count_nonzero(changed[0, 2:]) == 0
    assert torch.count_nonzero(changed[1, 3:]) == 0


def test_price_amendment_and_authorities_match_bytes():
    amendment = json.loads(target.PRIOR_V2.read_text())
    assert amendment["amended_price"] == target.PRICE
    observed = {"prior_v1": target.sha(target.PRIOR_V1),
                "prior_v2": target.sha(target.PRIOR_V2),
                "route_result": target.sha(target.ROUTE_RESULT),
                "greedy_result": target.sha(target.GREEDY_RESULT),
                "mediation_runner": target.sha(target.MEDIATION_RUNNER)}
    assert observed == target.EXPECTED


def test_prediction_inventory_and_finiteness():
    assert len(target.PREDICTION_KEYS) == 5
    assert target.finite_reports([{"signed_recovery": 0.2, "cosine": 1.0}])
    assert not target.finite_reports([{"signed_recovery": float("inf")}])
