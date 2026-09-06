import json

import run_temporal_auxiliary_will_had_block8h1_subject_source_groups_v1 as runner


def test_static_authorities_and_population():
    rows, fast_screen, capability = runner.validate_static()
    assert len(rows) == 64
    assert fast_screen["terminal"] == "screen"
    assert len(capability) == 4
    assert runner.ARMS == ("full_h1", "all_sources", "prefix", "cue", "local")


def test_dryrun_exact_price(monkeypatch, capsys):
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    runner.main()
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["model_forwards"] == 14
    assert receipt["example_evaluations"] == 448
    assert receipt["intervention_records"] == 320
    assert receipt["selected_heads"] == [1]
    assert receipt["gpu_accessed"] is False
