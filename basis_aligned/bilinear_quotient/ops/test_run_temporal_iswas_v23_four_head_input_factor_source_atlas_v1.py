import json
import os
from pathlib import Path
import subprocess
import sys

import torch

import run_temporal_iswas_v23_four_head_input_factor_source_atlas_v1 as run


def test_enumeration_price_and_sites_are_frozen():
    assert run.ROUTES == ((8, 1), (9, 1), (9, 4), (11, 3))
    assert run.ROUTES_BY_LAYER == {8: (1,), 9: (1, 4), 11: (3,)}
    assert run.PRICE["model_forwards_exact"] == 2 + 6 + 31 + 6
    assert run.PRICE["sequence_evaluations_exact"] == 45 * 64
    assert len(run.source.SOURCE_FACTORS) == 5
    assert len(run.source.SOURCE_GROUPS) == 3


def test_same_layer_multi_head_replacements_are_both_retained():
    base = torch.zeros(2, 3, 6, 4)
    head1 = torch.full((2, 3, 4), 11.0)
    head4 = torch.full((2, 3, 4), 44.0)
    changed = run.assemble_layer_head_outputs(base, {1: head1, 4: head4})
    assert torch.equal(changed[:, :, 1], head1)
    assert torch.equal(changed[:, :, 4], head4)
    assert torch.count_nonzero(changed[:, :, (0, 2, 3, 5)]) == 0
    assert torch.count_nonzero(base) == 0


def test_shapley_reconstructs_an_additive_game_exactly():
    weights = [1.0, -2.0, 3.0]
    values = [sum(weights[index] for index in range(3) if mask & (1 << index))
              for mask in range(8)]
    report = run.shapley(values, ("a", "b", "c"))
    assert report["allocations"] == {"a": 1.0, "b": -2.0, "c": 3.0}
    assert report["efficiency_residual"] == 0.0


def test_selectivity_requires_target_and_both_controls():
    good = {
        "target": {"signed_projection": .6, "cosine": .95, "direction_fraction": 1.0},
        "controls": {"P": .1, "C": .01},
    }
    assert run.selective(good)
    assert not run.selective({**good, "controls": {"P": .16, "C": .01}})


def test_model_free_dryrun_has_no_result_or_queue_side_effect(tmp_path):
    before_queue = run.ROOT.joinpath("queue.txt").read_bytes()
    result_existed = run.OUT.exists()
    env = {**os.environ, "BQLIB_NO_MODEL": "1"}
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.ROOT.parents[1], env=env,
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dryrun"] and not payload["gpu_accessed"] and not payload["model_loaded"]
    assert payload["authority_ok"]
    assert payload["partition_ok"] and payload["target_rows"] == 30
    assert payload["capture_forwards"] == 6
    assert run.ROOT.joinpath("queue.txt").read_bytes() == before_queue
    assert run.OUT.exists() == result_existed
