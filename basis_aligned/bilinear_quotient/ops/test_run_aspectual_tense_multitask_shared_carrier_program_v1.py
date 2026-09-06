from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_multitask_shared_carrier_program_v1 as runner


def test_common_pool_and_exact_dryrun() -> None:
    _splits, _chosen, _pools, common = runner.validate_static()
    assert len(common) == 17
    assert {"MLP3", "MLP4", "MLP6", "MLP8"} <= set(common)
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["maximum_model_forwards"] == runner.MAX_FORWARDS == 270
    assert payload["maximum_example_evaluations"] == runner.MAX_EVALUATIONS == 3744
    assert payload["maximum_intervention_records"] == runner.MAX_RECORDS == 3562
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
