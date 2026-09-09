import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import torch

import run_temporal_iswas_v23_direct_residual_readout_factorial_v1 as run


def test_carry_coefficient_is_exact_product_of_blocks_12_through_17():
    blocks = [SimpleNamespace(lambdas=[torch.tensor(1.0)]) for _ in range(18)]
    expected = 1.0
    for layer in range(12, 18):
        blocks[layer].lambdas[0] = torch.tensor(.5 + layer / 100)
        expected *= .5 + layer / 100
    model = SimpleNamespace(transformer=SimpleNamespace(h=blocks))
    coefficient, factors = run.carry_coefficient(model)
    assert len(factors) == 6
    assert math.isclose(coefficient, expected, rel_tol=1e-6, abs_tol=1e-8)


def test_direct_readout_price_and_dependency_are_frozen():
    assert run.PRICE["model_forwards_exact"] == 4
    assert run.PRICE["sequence_evaluations_exact"] == 4 * 64
    assert json.loads(run.PARENT.read_text())["terminal"] == "direct_residual_readout_candidate"
    observed = {name: run.sha(path) for name, path in {
        "authority": run.AUTHORITY, "prior": run.PRIOR, "parent": run.PARENT,
        "reader": run.READER, "component": run.COMPONENT, "shared": run.SHARED,
    }.items()}
    assert observed == {name: run.EXPECTED[name] for name in observed}


def test_model_free_dryrun_has_no_side_effects_and_no_v25_access():
    existed = run.OUT.exists()
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.ROOT.parents[1],
        env={**os.environ, "BQLIB_NO_MODEL": "1"}, check=True,
        text=True, capture_output=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dependency_ok"] and payload["rows"] == 64
    assert payload["carry_layers"] == list(range(12, 18))
    assert payload["v25_accessed"] is False
    assert not payload["gpu_accessed"] and not payload["model_loaded"]
    assert run.OUT.exists() == existed
