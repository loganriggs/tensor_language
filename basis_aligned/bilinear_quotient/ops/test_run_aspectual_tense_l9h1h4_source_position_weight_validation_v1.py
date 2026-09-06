from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_l9h1h4_source_position_weight_validation_v1 as runner


def test_static_authority_and_exact_dryrun() -> None:
    splits, chosen, query_top = runner.validate_static()
    assert {name: len(rows) for name, rows in splits.items()} == {
        "has_fit": 16, "has_heldout": 15, "has_a2": 31,
        "is_fit": 8, "is_heldout": 6, "is_a2": 15,
    }
    assert all(len(chosen[task]["upstream_attention"]) == 12 for task in ("has", "is"))
    assert all(query_top[task][kind] > 0 for task in ("has", "is") for kind in runner.TYPES)
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["model_forwards"] == runner.FORWARDS == 88
    assert payload["example_evaluations"] == runner.EVALUATIONS == 1474
    assert payload["intervention_records"] == runner.RECORDS == 1340
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
