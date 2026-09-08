import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_l11h3_writer_route_closure_ladder_v1 as target


def test_dryrun_is_authority_bound_and_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["arms"] == list(target.ARMS)
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 18


def test_prefix_rows_include_every_position_through_query():
    assert target.prefix_rows([0, 2, 4]) == [[0], [0, 1, 2], [0, 1, 2, 3, 4]]


def test_prediction_inventory_matches_gate_and_prior_price():
    prior = json.loads(target.PRIOR.read_text())
    assert len(target.PREDICTION_KEYS) == 5
    assert prior["price"] == target.PRICE
    assert prior["locked_selected_prefixes"] == json.loads(
        target.GREEDY_RESULT.read_text())["selected_prefixes"]


def test_expected_authorities_match_bytes():
    observed = {
        "mediation_result": target.sha(target.MEDIATION_RESULT),
        "greedy_result": target.sha(target.GREEDY_RESULT),
        "binding": target.sha(target.BINDING),
        "mediation_runner": target.sha(target.MEDIATION_RUNNER),
    }
    assert observed == target.EXPECTED


def test_finite_reports_rejects_nonfinite_metric():
    assert target.finite_reports([{"signed_recovery": 0.8, "direction": 1.0}])
    assert not target.finite_reports([{"signed_recovery": float("nan")}])
