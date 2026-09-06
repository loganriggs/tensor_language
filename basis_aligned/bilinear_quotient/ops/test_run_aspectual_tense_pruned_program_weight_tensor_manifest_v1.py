from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_aspectual_tense_pruned_program_weight_tensor_manifest_v1 as runner


def test_authority_and_zero_forward_dryrun():
    programs, subspaces = runner.validate_static()
    assert programs["terminal"] == "screen"
    assert [subspaces["subspaces"][task]["rank"] for task in ("has", "is")] == [18, 3]
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["model_forwards"] == payload["example_evaluations"] == 0
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
