from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_aspectual_tense_carrier_program_backward_pruning_v1 as runner


def test_frozen_fit_only_selection_and_dryrun():
    _convergence, splits, _chosen, _pools, programs = runner.validate_static()
    assert {task: len(path) for task, path in programs.items()} == {"has": 8, "is": 11}
    assert len(splits["has_fit"]) == 16 and len(splits["is_fit"]) == 8
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["maximum_price"] == runner.MAX_PRICE
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
