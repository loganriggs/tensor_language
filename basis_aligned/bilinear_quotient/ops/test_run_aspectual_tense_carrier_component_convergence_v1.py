from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_carrier_component_convergence_v1 as runner


def test_frozen_prefixes_and_exact_dryrun():
    splits, _chosen, pools, paths = runner.validate_static()
    assert len(splits["is_fit"]) == 8
    assert len(pools["is"]) == 20
    assert {task: len(path) for task, path in paths.items()} == {"has": 8, "is": 10}
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["marginal_improvement_threshold"] == 0.001
    assert payload["maximum_model_forwards"] == runner.MAX_FORWARDS == 70
    assert payload["maximum_example_evaluations"] == runner.MAX_EVALUATIONS == 665
    assert payload["maximum_records"] == runner.MAX_RECORDS == 515
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
