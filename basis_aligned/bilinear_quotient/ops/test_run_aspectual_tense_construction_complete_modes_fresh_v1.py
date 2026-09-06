from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_aspectual_tense_construction_complete_modes_fresh_v1 as runner


def test_fresh_authority_and_dryrun():
    splits, _chosen, _pools, _programs, _subspaces, rows = runner.validate_static()
    assert len(splits["has_fit"]) + len(splits["has_heldout"]) + len(splits["has_a2"]) == 62
    assert {task: len(bank) for task, bank in rows.items()} == {"has_had": 64, "is_was": 64}
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["maximum_price"] == runner.MAX_PRICE
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
