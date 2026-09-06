from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import run_aspectual_tense_weight_tensor_relative_instrument_audit_v1 as runner


def test_invalid_pattern_and_zero_forward_dryrun():
    _programs, _subspaces, old = runner.validate_static()
    assert old["terminal"] == "invalid" and not old["predictions"]["pred_a_authority_basis_rank_gauge_finiteness_and_exact_replay"]
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env, check=True,
                            capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["expected_hashes"] == 180 and payload["relative_tolerance"] == 1e-5
    assert payload["model_forwards"] == payload["example_evaluations"] == 0
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
