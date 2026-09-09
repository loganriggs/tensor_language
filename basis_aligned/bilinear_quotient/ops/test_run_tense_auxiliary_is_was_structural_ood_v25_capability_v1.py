import json
import os
from pathlib import Path
import subprocess
import sys

import run_tense_auxiliary_is_was_structural_ood_v25_capability_v1 as run


def test_v25_runner_binds_failed_v24_and_valid_v23_cells_without_opening_outcomes():
    observed = {"authority": run.sha(run.AUTHORITY), "prior": run.sha(run.PRIOR),
                "builder": run.sha(run.BUILDER), "v24_result": run.sha(run.V24_RESULT),
                "v23_cells": run.sha(run.V23_CELLS)}
    assert observed == run.EXPECTED
    assert json.loads(run.V24_RESULT.read_text())["terminal"] == "null"
    assert json.loads(run.V23_CELLS.read_text())["terminal"] == "shared_directed_cell_program_screen"


def test_v25_dryrun_freezes_structure_inventory_alignment_and_price():
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.ROOT.parents[1],
        env={**os.environ, "BQLIB_NO_MODEL": "1"}, check=True,
        text=True, capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["authority_ok"] and payload["rows"] == 64
    assert len(payload["target_constructions"]) == len(payload["control_constructions"]) == 8
    assert payload["token_length_range"] == [8, 18]
    assert payload["model_forwards_exact"] == 2
    assert payload["sequence_evaluations_exact"] == 128
    assert payload["causal_outcomes_opened"] is False
    assert not payload["gpu_accessed"] and not payload["model_loaded"]


def test_v25_bars_require_bidirectional_cells_and_three_joint_rows():
    assert run.CELL_BAR == .5 and run.JOINT_BAR == 3
    assert run.PRICE == {"model_forwards_exact": 2, "sequence_evaluations_exact": 128,
                         "interventions": 0, "transformer_backwards": 0,
                         "model_updates": 0}
