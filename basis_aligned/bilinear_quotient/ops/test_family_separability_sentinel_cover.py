import json
from pathlib import Path

import pytest

import family_separability_sentinel_cover as target


ROOT = Path(__file__).resolve().parents[1]


def test_collision_graph_is_bar_strict_and_ignores_incomplete_members():
    members = {
        "a": {"arms": {"own": {"siblings": {"x": .051, "y": -.05, "z": -.2}}}},
        "b": {"error": "missing"},
    }
    assert target.collision_graph(members) == {"a": {"x", "z"}}


def test_greedy_covers_are_deterministic_and_control_disjoint():
    graph = {"a": {"x", "y"}, "b": {"x", "y"}, "c": {"x", "y"}}
    covers = target.greedy_disjoint_covers(graph, maximum=2)
    assert not covers[0]["uncovered"]
    assert not covers[1]["uncovered"]
    assert set(covers[0]["controls"]).isdisjoint(covers[1]["controls"])
    assert covers == target.greedy_disjoint_covers(dict(reversed(list(graph.items()))), maximum=2)


def test_completed_family_receipts_have_three_disjoint_known_collision_covers():
    cases = {
        "unit_family_separability_spec_v281_result.json": (22, [3, 3, 3]),
        "unit_family_separability_spec_v260_result.json": (15, [2, 3, 3]),
    }
    for filename, (universe_size, widths) in cases.items():
        receipt = json.loads((ROOT / "circuits/followups" / filename).read_text())
        graph = target.collision_graph(
            receipt["members"], cross_bar=receipt["bars"]["cross_max"])
        covers = target.greedy_disjoint_covers(graph, maximum=3)
        assert sum(bool(controls) for controls in graph.values()) == universe_size
        assert [len(cover["controls"]) for cover in covers] == widths
        assert all(not cover["uncovered"] for cover in covers)


def test_incremental_cost_model_meets_ten_minute_target_for_ten_by_three():
    assert target.estimated_seconds(targets=10, controls=3) == pytest.approx(469.0)
    assert target.estimated_seconds(targets=44, controls=44) == pytest.approx(18840.8)
