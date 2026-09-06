from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_activation_conditioned_mlp_fresh_validation_v1 as runner


def test_fresh_authority_and_exact_dryrun() -> None:
    _subspaces, rows = runner.validate_static()
    assert all(len(task_rows) == 32 for task_rows in rows.values())
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["maximum_model_forwards"] == runner.MAX_FORWARDS == 34
    assert payload["maximum_example_evaluations"] == runner.MAX_EVALUATIONS == 544
    assert payload["maximum_causal_records"] == runner.MAX_RECORDS == 288
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
