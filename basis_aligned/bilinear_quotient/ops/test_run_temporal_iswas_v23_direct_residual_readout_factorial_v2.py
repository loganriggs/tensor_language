import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_v23_direct_residual_readout_factorial_v2 as run


def test_v2_binds_invalid_v1_and_preserves_scientific_bars_and_price():
    failed = json.loads(run.FAILED_V1.read_text())
    assert failed["terminal"] == "invalid_instrument"
    assert failed["state_closure_max_abs"] == .0078125
    assert failed["reports"]["direct"]["controls"]["P"] > run.experiment.BARS["control"]
    assert run.experiment.PRICE["model_forwards_exact"] == 4
    assert run.experiment.BARS["control"] == .15


def test_v2_dryrun_is_hash_bound_and_side_effect_free():
    existed = run.experiment.OUT.exists()
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.experiment.ROOT.parents[1],
        env={**os.environ, "BQLIB_NO_MODEL": "1"}, check=True,
        text=True, capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["candidate_id"].endswith("_v2") and payload["dependency_ok"]
    assert not payload["gpu_accessed"] and not payload["model_loaded"]
    assert run.experiment.OUT.exists() == existed
