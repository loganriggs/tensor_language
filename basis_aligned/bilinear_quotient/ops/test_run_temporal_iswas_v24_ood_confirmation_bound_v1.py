import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_v24_ood_confirmation_bound_v1 as target


def test_launcher_delegates_exact_prediction_inventory_and_price():
    assert tuple(target.REGISTERED_PREDICTIONS) == target.confirmation.PREDICTION_KEYS
    assert target.confirmation.PRICE == {
        "checkpoint_loads": 1,
        "model_forwards_exact": 9,
        "sequence_evaluations_exact": 576,
        "transformer_backwards": 1,
        "model_updates": 0,
        "fit_parameters": 0,
    }


def test_dryrun_waits_without_creating_binding_or_loading_model():
    binding_existed = target.binding.BINDING.exists()
    env = {**os.environ, "BQLIB_NO_MODEL": "1"}
    completed = subprocess.run(
        [sys.executable, str(Path(target.__file__))],
        cwd=target.confirmation.ROOT.parents[1], env=env,
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["conditional_launcher"]
    assert not payload["gpu_accessed"] and not payload["model_loaded"]
    assert payload["binding_status"]["status"] in {"awaiting_result", "eligible"}
    assert target.binding.BINDING.exists() == binding_existed
