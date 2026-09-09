import json
import os
from pathlib import Path
import subprocess
import sys

import run_temporal_iswas_v23_source_destination_cell_atlas_v3 as run


def metric(projection=.2, cosine=.8, direction=.8):
    return {
        "signed_projection": projection, "cosine": cosine,
        "direction_fraction": direction,
    }


def reciprocal_report(**overrides):
    arm = {
        "target": metric(), "controls": {"P": .1, "C": .1},
        "halves": {"first": metric(.1), "second": metric(.1)},
    }
    report = {"sufficiency": arm, "removed_by_reset": json.loads(json.dumps(arm))}
    report.update(overrides)
    return report


def test_price_role_cross_product_and_route_count_are_exact():
    assert run.ROLE_NAMES == ("changed", "unchanged_prefix", "matched_suffix")
    assert len(run.ROUTES) == 4
    assert run.PRICE["model_forwards_exact"] == 2 + 6 + 4 * (9 + 9 + 1)
    assert run.PRICE["sequence_evaluations_exact"] == 84 * 64
    assert {run.cell_label(d, s) for d in range(3) for s in range(3)} == {
        f"{destination}<-{source}"
        for destination in run.ROLE_NAMES for source in run.ROLE_NAMES
    }


def test_reciprocal_requires_both_directions_halves_and_controls():
    report = reciprocal_report()
    assert run.is_reciprocal(report)
    failed_reset = json.loads(json.dumps(report))
    failed_reset["removed_by_reset"]["target"]["signed_projection"] = .14
    assert not run.is_reciprocal(failed_reset)
    failed_half = json.loads(json.dumps(report))
    failed_half["sufficiency"]["halves"]["second"]["signed_projection"] = 0.0
    assert not run.is_reciprocal(failed_half)
    failed_control = json.loads(json.dumps(report))
    failed_control["removed_by_reset"]["controls"]["C"] = .26
    assert not run.is_reciprocal(failed_control)


def test_dependency_is_exactly_hash_bound_and_valid_now():
    dependency = run.dependency_status()
    assert dependency["status"] == "valid"
    assert dependency["sha256"] == run.EXPECTED["parent_result"]


def test_model_free_dryrun_has_no_result_or_queue_side_effects():
    result_existed = run.OUT.exists()
    queue_paths = sorted(run.ROOT.joinpath("ops").glob("queue*.txt"))
    before = {path: path.read_bytes() for path in queue_paths}
    completed = subprocess.run(
        [sys.executable, str(Path(run.__file__))], cwd=run.ROOT.parents[1],
        env={**os.environ, "BQLIB_NO_MODEL": "1"}, check=True,
        capture_output=True, text=True,
    )
    payload = json.loads(completed.stdout)
    assert payload["dryrun"] and not payload["gpu_accessed"] and not payload["model_loaded"]
    assert payload["static_authority_ok"] and payload["dependency"]["status"] == "valid"
    assert payload["cells_per_head"] == 9 and payload["capture_forwards"] == 6
    assert run.OUT.exists() == result_existed
    assert {path: path.read_bytes() for path in queue_paths} == before
