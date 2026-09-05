import json

import analyze_task14_ood_mlp6_7_prompt_level_composition_crossvalidation as run


def test_plan_is_hash_bound_and_zero_gpu():
    plan = run.compile_plan()
    assert plan["folds"] == 16
    assert plan["predicted_intermediate_conditions"] == 224
    assert all(value == 0 for value in plan["price"].values())


def test_row_lattices_are_complete():
    cells = run._row_lattices(json.loads(run.SOURCE_RESULT.read_text()))
    assert sorted(len(rows) for rows in cells.values()) == [8, 8]
    assert all(set(q) == set(run.SUBSETS) for rows in cells.values() for q in rows.values())


def test_profile_recovers_additive_coefficients():
    expected = {"E": .4, "A": .3, "U": .2, "W": .1}
    q = {s: 1.0 + sum(expected[f] for f in s) for s in run.SUBSETS}
    profile = run._profile([q, q])
    assert all(abs(profile[f] - expected[f]) < 1e-12 for f in expected)


def test_analysis_scores_all_rows_and_registered_predictions():
    score = run.analyze()
    assert score["total_row_count"] == 16
    assert len(score["rows"]) == 16
    assert len(score["predictions"]) == 6


def test_dry_run_does_not_write(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "OUT", tmp_path / "forbidden.json")
    run.main(["--dry-run"])
    assert not run.OUT.exists()
