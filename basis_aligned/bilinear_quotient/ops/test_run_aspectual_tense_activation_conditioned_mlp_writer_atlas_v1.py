from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_activation_conditioned_mlp_writer_atlas_v1 as runner


def test_development_authority_and_exact_dryrun() -> None:
    _subspaces, _causal, splits = runner.validate_static()
    assert len(splits["has_fit"]) == 16 and len(splits["is_fit"]) == 8
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["model_forwards"] == 4 and payload["example_evaluations"] == 48
    assert payload["task_layer_scores"] == 18 and payload["selected_causal_matches"] == 16
    assert payload["causal_interventions"] == 0
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
