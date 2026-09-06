from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_aspectual_tense_core_program_composition_v1 as runner


def test_frozen_factorization_and_dryrun():
    _necessity, _splits, _chosen, _pools, complete, core = runner.validate_static()
    assert set(core["has"]) & set(core["is"]) == set(runner.SHARED)
    assert {task: len(path) for task, path in complete.items()} == {"has": 8, "is": 11}
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert {key: payload[key] for key in runner.PRICE} == runner.PRICE
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
