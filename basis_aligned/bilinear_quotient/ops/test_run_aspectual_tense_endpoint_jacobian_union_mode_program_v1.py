from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_aspectual_tense_endpoint_jacobian_union_mode_program_v1 as runner


def test_parent_null_and_dryrun():
    splits, _chosen, _pools, programs, _subspaces = runner.validate_static()
    assert len(splits["has_fit"]) == 16 and len(splits["is_fit"]) == 8
    assert {task: len(path) for task, path in programs.items()} == {"has": 7, "is": 10}
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert {key: payload[key] for key in runner.PRICE} == runner.PRICE
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
