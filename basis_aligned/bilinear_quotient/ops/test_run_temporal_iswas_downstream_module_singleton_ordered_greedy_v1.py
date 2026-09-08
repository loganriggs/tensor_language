import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_downstream_module_singleton_ordered_greedy_v1 as target


def test_dryrun_is_authority_bound_and_model_free():
    completed = subprocess.run([sys.executable, str(Path(target.__file__))], check=True,
        capture_output=True, text=True,
        env=dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1"))
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] is True
    assert payload["order"] == list(target.ORDER)
    assert payload["gpu_accessed"] is False
    assert payload["model_loaded"] is False
    assert payload["queue_touched"] is False
    assert payload["price"]["model_forwards"] == 19


def test_order_is_exact_singleton_fit_ranking():
    result = json.loads(target.ATLAS_RESULT.read_text())
    fit = {row["module"]: row["signed_recovery"]
           for row in result["reports"] if row["phase"] == "FIT"}
    observed = tuple(sorted(fit, key=lambda name: (-fit[name], name)))
    assert observed == target.ORDER


def test_fit_and_holdout_qualification_are_distinct():
    row = {"signed_recovery": .55, "cosine": .92,
           "relative_residual": .6, "direction_agreement": .95}
    assert target.qualified(row, "FIT")
    assert target.qualified(row, "HOLDOUT")
    row["signed_recovery"] = .45
    assert not target.qualified(row, "FIT")
    assert target.qualified(row, "HOLDOUT")


def test_prediction_inventory_and_price_match_prior():
    prior = json.loads(target.PRIOR.read_text())
    assert len(target.PREDICTION_KEYS) == 5
    assert prior["price"] == target.PRICE
    assert prior["locked_module_order"] == list(target.ORDER)


def test_expected_authorities_match_bytes():
    observed = {"prior": target.sha(target.PRIOR),
                "atlas_result": target.sha(target.ATLAS_RESULT),
                "atlas_runner": target.sha(target.ATLAS_RUNNER),
                "greedy_result": target.sha(target.GREEDY_RESULT)}
    assert observed == target.EXPECTED
