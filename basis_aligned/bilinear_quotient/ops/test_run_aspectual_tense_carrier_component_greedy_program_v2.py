from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_carrier_component_greedy_program_v2 as runner


def test_v2_configuration_and_exact_dryrun() -> None:
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["candidate_id"] == runner.CANDIDATE_ID
    assert payload["maximum_components"] == 10
    assert payload["maximum_model_forwards"] == 438
    assert payload["maximum_example_evaluations"] == 5845
    assert payload["maximum_intervention_records"] == 5663
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
