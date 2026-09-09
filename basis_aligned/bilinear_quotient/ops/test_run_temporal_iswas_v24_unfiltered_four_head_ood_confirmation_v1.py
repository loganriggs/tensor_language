import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_v24_unfiltered_four_head_ood_confirmation_v1 as run


def test_v24_ood_confirmation_is_result_bound_and_unfiltered():
    assert not run.CAPABILITY.exists()
    assert not run.BINDING.exists()
    prior = json.loads(run.PRIOR.read_text())
    assert prior["frozen_design"]["population"].startswith("Forward all 64 v24 rows")
    assert prior["frozen_design"]["price"] == run.PRICE
    assert prior["frozen_design"]["routes"] == list(run.ROUTES)


def test_v24_eligibility_binds_capability_and_confirmation_runners(monkeypatch):
    capability = {
        "terminal": "screen", "predictions": {"a": True},
        "causal_outcomes_opened": False, "rows_sha256": run.ROWS_SHA256,
        "jointly_capable_row_ids": {panel: [] for panel in run.PANELS},
    }
    binding = {
        "schema": "temporal_iswas_v24_ood_confirmation_binding_v1",
        "capability_result_sha256": "capability-result",
        "capability_runner_sha256": run.EXPECTED["capability_runner"],
        "confirmation_runner_sha256": "confirmation-runner",
    }
    observed = {run.CAPABILITY: "capability-result", run.SELF: "confirmation-runner"}
    monkeypatch.setattr(run, "sha", lambda path: observed[path])
    assert run.eligibility(binding, capability)
    capability["causal_outcomes_opened"] = True
    assert not run.eligibility(binding, capability)


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
