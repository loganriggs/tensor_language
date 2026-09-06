from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import run_aspectual_tense_carrier_component_greedy_program_v1 as runner


def test_static_pool_program_merge_and_dryrun() -> None:
    splits, _chosen, pools = runner.validate_static()
    assert len(splits["has_fit"]) == 16 and len(splits["is_fit"]) == 8
    assert all(len(pool) == 20 for pool in pools.values())
    has = pools["has"]
    specs = runner.program_specs(("L8H1", "L8H3", "MLP4"), has)
    assert len(specs) == 2
    assert specs[0].kind == "attention_heads" and specs[0].heads == (1, 3)
    assert specs[1].kind == "mlp" and specs[1].layer == 4
    env = dict(os.environ, BQLIB_DRYRUN="1", PYTHONPATH=str(Path(runner.__file__).parent))
    result = subprocess.run([sys.executable, runner.__file__], env=env,
                            check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    assert payload["maximum_model_forwards"] == runner.MAX_FORWARDS == 322
    assert payload["maximum_example_evaluations"] == runner.MAX_EVALUATIONS == 4377
    assert payload["maximum_intervention_records"] == runner.MAX_RECORDS == 4195
    assert not payload["gpu_accessed"] and not payload["queue_touched"]
