import json

import run_temporal_auxiliary_will_had_block8h1_fresh_cue_path_v1 as runner


def test_static_fresh_authority():
    rows = runner.validate_static()
    assert len(rows) == 64
    assert runner.ARMS == (
        "full_attention", "h1_complete", "h1_all_sources", "h1_cue", "h1_noncue"
    )


def test_dryrun_exact_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 14
    assert receipt["example_evaluations"] == 448
    assert receipt["intervention_records"] == 320
    assert receipt["gpu_accessed"] is False


def test_capability_cells_cover_both_sides_and_directions():
    class Output:
        answer_foil = tuple((2.0, 1.0) for _ in range(32))
    rows = runner.validate_static()
    cells = runner.capability_cells(rows, {
        family: {side: Output() for side in ("base", "donor")}
        for family in ("A1", "A2")
    })
    assert len(cells) == 8
    assert all(cell["passed"] for cell in cells)
