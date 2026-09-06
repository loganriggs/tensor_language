from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_auxiliary_will_had_block11h3_aligned_objective_cdas_v1 as runner


def test_static_authority_and_dryrun_price():
    rows, authority, axis = runner.validate_static()
    assert len(rows) == 128 and len(axis) == 128 and authority["terminal"] == "null"
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["lambda_grid"] == list(runner.LAMBDAS)
    assert payload["model_forwards"] == 1373
    assert payload["example_evaluations"] == 22176
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
