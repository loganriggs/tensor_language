from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_aspectual_tense_carrier_program_leave_one_out_v1 as runner


def test_authority_and_exact_dryrun():
    _result, splits, _chosen, _pools, programs = runner.validate_static()
    assert {task: len(path) for task, path in programs.items()} == {"has": 8, "is": 11}
    assert {key: len(value) for key, value in splits.items() if key.endswith(("heldout", "a2"))} == {
        "has_heldout": 15, "has_a2": 31, "is_heldout": 6, "is_a2": 15}
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert {key: payload[key] for key in runner.PRICE} == runner.PRICE
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
