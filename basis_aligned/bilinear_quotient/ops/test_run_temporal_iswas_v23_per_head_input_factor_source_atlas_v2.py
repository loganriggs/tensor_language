import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

import run_temporal_iswas_v23_per_head_input_factor_source_atlas_v2 as run


def test_price_and_per_head_enumeration_are_exact():
    assert run.ROUTES == ((8, 1), (9, 1), (9, 4), (11, 3))
    assert run.PRICE["model_forwards_exact"] == 2 + 6 + 4 * (31 + 6)
    assert run.PRICE["sequence_evaluations_exact"] == 156 * 64


def test_profile_normalization_uses_l1_and_preserves_sign():
    profile = run.normalized_profile({"a": 2.0, "b": -1.0, "c": 1.0}, ("a", "b", "c"))
    assert np.allclose(profile, np.asarray([.5, -.25, .25]))
    assert np.isclose(np.abs(profile).sum(), 1.0)


def test_pair_distances_require_the_same_pair_in_both_halves():
    names = ("a", "b")
    games = {
        "h1": {"allocations": {"a": 1.0, "b": 0.0}},
        "h2": {"allocations": {"a": 0.0, "b": 1.0}},
    }
    halves = {
        "h1": {half: games["h1"] for half in ("first", "second")},
        "h2": {half: games["h2"] for half in ("first", "second")},
    }
    report = run.pair_profile_distances(games, halves, names)["h1__h2"]
    assert report == {"overall_l1": 2.0, "halves": {"first": 2.0, "second": 2.0}}


def test_head_relative_selectivity_checks_both_controls():
    report = {
        "target": {"signed_projection": .8, "cosine": .95, "direction_fraction": 1.0},
        "controls": {"P": .1, "C": .2},
    }
    assert run.selective(report)
    assert not run.selective({**report, "controls": {"P": .1, "C": .3}})


def test_model_free_dryrun_accepts_pending_or_valid_dependency_without_side_effects():
    before_queue = run.ROOT.joinpath("queue.txt").read_bytes()
    result_existed = run.OUT.exists()
    env = {**os.environ, "BQLIB_NO_MODEL": "1"}
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.ROOT.parents[1], env=env,
        check=True, capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dryrun"] and not payload["gpu_accessed"] and not payload["model_loaded"]
    assert payload["static_authority_ok"]
    assert payload["dependency"]["status"] in {"pending", "valid"}
    assert payload["capture_forwards"] == 6 and payload["target_rows"] == 30
    assert run.ROOT.joinpath("queue.txt").read_bytes() == before_queue
    assert run.OUT.exists() == result_existed
