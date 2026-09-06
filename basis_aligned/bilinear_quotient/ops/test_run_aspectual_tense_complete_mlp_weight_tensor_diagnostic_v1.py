from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_complete_mlp_weight_tensor_diagnostic_v1 as runner


def test_development_authority_and_zero_forward_dryrun() -> None:
    _subspaces, causal = runner.validate_static()
    assert causal["scope_correction_passes"] == 3
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["task_layer_tensors"] == 18
    assert payload["selected_causal_matches"] == 16
    assert payload["model_forwards"] == 0 and payload["causal_records"] == 0
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
