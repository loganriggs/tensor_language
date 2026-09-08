import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1 as run


def test_v24_ood_confirmation_is_result_bound_and_unfiltered():
    assert run.EXPECTED_CAPABILITY_RESULT_SHA256 is None
    prior = json.loads(run.PRIOR.read_text())
    assert prior["frozen_design"]["population"].startswith("Forward all 64 v24 rows")
    assert prior["frozen_design"]["price"] == run.PRICE
    assert prior["frozen_design"]["routes"] == list(run.ROUTES)


def test_v24_ood_confirmation_prebound_authorities_match():
    paths = {"prior": run.PRIOR, "builder": run.BUILDER,
             "capability_prior": run.CAPABILITY_PRIOR,
             "capability_runner": run.CAPABILITY_RUNNER,
             "v23_result": run.V23_RESULT, "reference": run.REFERENCE}
    assert {name: run.sha(path) for name, path in paths.items()} == run.EXPECTED


def test_v24_ood_price_is_union_singletons_and_two_closures_only():
    assert run.PRICE["model_forwards_exact"] == 2 + 2 + 1 + len(run.ROUTES) == 9
    assert run.PRICE["sequence_evaluations_exact"] == 9 * 64 == 576
    assert len(run.PREDICTION_KEYS) == 4


def test_v24_ood_dryrun_waits_without_loading_model():
    env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
    completed = subprocess.run([sys.executable, str(Path(run.__file__))], env=env,
                               text=True, capture_output=True, check=True)
    payload = json.loads(completed.stdout)
    assert payload["status"] == "awaiting_hash_bound_v24_capability_result"
    assert payload["model_loaded"] is False and payload["gpu_accessed"] is False
    assert payload["target_rows_unfiltered"] == 32
    assert payload["capability_row_filter_used"] is False
