import json
import os
from pathlib import Path
import subprocess
import sys

import run_tense_auxiliary_is_was_structural_holdout_v26_capability_v1 as run


def test_v26_runner_binds_v25_null_without_opening_causal_outcomes():
    observed = {name: run.sha(path) for name, path in run.PATHS.items()}
    assert observed == run.EXPECTED
    assert json.loads(run.V25_RESULT.read_text())["terminal"] == "null"


def test_v26_dryrun_freezes_population_and_price():
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.ROOT.parents[1],
        env={**os.environ, "BQLIB_NO_MODEL": "1"}, check=True,
        text=True, capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] and payload["rows"] == 64
    assert len(payload["target_constructions"]) == len(payload["control_constructions"]) == 4
    assert payload["model_forwards_exact"] == 2
    assert payload["sequence_evaluations_exact"] == 128
    assert payload["causal_outcomes_opened"] is False
    assert not payload["gpu_accessed"] and not payload["model_loaded"]


def test_v26_bars_are_construction_level_and_bidirectional():
    assert run.CELL_BAR == .75 and run.JOINT_BAR == 6
    assert run.PRICE["model_forwards_exact"] == 2
    assert run.PRICE["sequence_evaluations_exact"] == 128

