from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_path_conditioned_mlp_metric_derivation_v1 as runner


def test_development_authority_and_exact_dryrun():
    causal, rows = runner.validate_static()
    assert causal["price"]["causal_records"] == 288
    assert all(len(task_rows) == 16 for task_rows in rows.values())
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["model_forwards"] == runner.FORWARDS == 12
    assert payload["example_evaluations"] == runner.EVALUATIONS == 184
    assert payload["transformer_backwards"] == runner.BACKWARDS == 2
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
